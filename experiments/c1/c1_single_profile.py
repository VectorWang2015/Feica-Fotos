#!/usr/bin/env python3
"""Local-only C1 single-camera-ICC diagnostics, never an official RAW renderer.

Build one affine PCS order probe and compare bounded-domain Look proposals.
No installation, proprietary application execution, or network operation.
All outputs are create-only. Numerical tests use generic math, not C1 itself.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import struct
import sys

import numpy as np
from PIL import Image, ImageCms

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from reproduction.ios_looks.renderer import TextureLUT, sample_texture


from experiments.c1 import inspect_profile as audit
from experiments.c1 import assess_vivid_domain as color

# User-supplied native camera profile is deliberately not bundled.
SOURCE = ROOT/'inputs/LeicaQTyp116-Generic.icm'
SOURCE_SHA = 'efed3f9554cb8da91de1fffb0e4a93b2ae9f98acc47d2cd4078133b38f99b969'
VIVID = ROOT/'filters/looks/Leica_Vivid_sRGB_sRGB_Release.cube'
# A modest, invertible affine perturbation, NOT a Look or calibration correction.
PROBE_MATRIX = np.array([[.90, 0, 0], [.04, .98, 0], [-.03, 0, .98]])
PROBE_NAME = 'LocalLooks-C1Single-OrderProbe-v1'


def configure_inputs(profile=None, cube=None):
    """Select explicit local inputs without reading or copying either file."""
    global SOURCE, VIVID
    if profile is not None:
        SOURCE = Path(profile)
    if cube is not None:
        VIVID = Path(cube)


def input_label(path):
    """Keep portable report identifiers instead of recording a host home path."""
    path = Path(path)
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return path.name


def sha(data):
    return hashlib.sha256(data).hexdigest()


def stats(values):
    x = np.asarray(values, dtype=float)
    if not x.size:
        return {'count': 0}
    return {'count': int(x.size), 'mean': float(x.mean()), 'max': float(x.max()),
            'p50': float(np.quantile(x, .5)), 'p95': float(np.quantile(x, .95)),
            'p99': float(np.quantile(x, .99))}


def read_tags(data):
    if len(data) < 132 or data[36:40] != b'acsp' or audit.u32(data, 0) != len(data):
        raise ValueError('Invalid ICC header/length')
    count = audit.u32(data, 128)
    end = 132+12*count
    if end > len(data):
        raise ValueError('Truncated directory')
    tags = []
    for i in range(count):
        k = 132+12*i
        sig, off, length = data[k:k+4], audit.u32(data, k+4), audit.u32(data, k+8)
        if off < end or off % 4 or length < 8 or off+length > len(data):
            raise ValueError('Invalid tag bounds/alignment')
        if sig in [t[0] for t in tags]:
            raise ValueError('Duplicate tag')
        tags.append((sig, data[off:off+length]))
    return tags


def native_lut(data):
    if data[8:12] != bytes.fromhex('02100000') or data[12:24] != b'scnrRGB Lab ':
        raise ValueError('This bounded tool expects the observed Q1 v2.1 RGB->Lab profile')
    lut = audit.Lut16(dict(read_tags(data))[b'A2B0'])
    if (lut.ni, lut.no, lut.grid, lut.nin, lut.nout) != (3, 3, 33, 256, 2):
        raise ValueError('Unexpected native LUT structure')
    if lut.outputs != [[0, 65535]]*3 or lut.matrix != [1., 0., 0., 0., 1., 0., 0., 0., 1.]:
        raise ValueError('Expected identity matrix/output tables')
    return lut


def codes_lab(codes):
    x = np.asarray(codes, dtype=float)
    return np.stack((x[..., 0]/652.8, x[..., 1]/256-128, x[..., 2]/256-128), axis=-1)


def lab_codes(lab):
    x = np.asarray(lab, dtype=float)
    return np.stack((x[..., 0]*652.8, (x[..., 1]+128)*256, (x[..., 2]+128)*256), axis=-1)


def probe(lab):
    return np.asarray(lab, dtype=float) @ PROBE_MATRIX.T


def description(text):
    raw = text.encode('ascii')+b'\0'
    if len(raw) > 67:
        raise ValueError('Description exceeds ScriptCode field')
    return b'desc'+bytes(4)+struct.pack('>I', len(raw))+raw+bytes(8)+struct.pack('>HB', 0, len(raw))+raw.ljust(67, b'\0')


def replace_clut(lut, values):
    codes = lab_codes(values)
    if not np.isfinite(codes).all() or np.any(codes < 0) or np.any(codes > 65535):
        raise ValueError('PCS is not representable; refusing any silent clipping')
    result = bytearray(lut.raw)
    result[lut.cb:lut.ce] = np.floor(codes+.5).astype('>u2').tobytes()
    return bytes(result)


def serialize_probe(data):
    lut = native_lut(data)
    nodes = codes_lab(np.asarray(lut.clut).reshape(-1, 3))
    payload = replace_clut(lut, probe(nodes))
    tags = read_tags(data)
    result = bytearray(data[:128]+struct.pack('>I', len(tags))+bytes(12*len(tags)))
    for i, (sig, old) in enumerate(tags):
        new = description(PROBE_NAME) if sig == b'desc' else payload if sig == b'A2B0' else old
        result.extend(bytes((-len(result)) % 4))
        off = len(result)
        struct.pack_into('>4sII', result, 132+12*i, sig, off, len(new))
        result.extend(new)
    struct.pack_into('>I', result, 0, len(result))
    return bytes(result)


def smoothstep(x):
    t = np.clip(x, 0, 1)
    return t*t*(3-2*t)


def bounded_look(lab, table, margin=.08):
    """PCS residual policy: exact identity outside sRGB, full Look in core.

    This is a new declared approximation, not official intensity, gamut mapping,
    or a reconstruction of C1's/Leica's Standard rendering. No RGB is clipped.
    """
    if not 0 < margin < .5:
        raise ValueError('margin must be inside (0,.5)')
    x = np.asarray(lab, dtype=float)
    rgb = color.lab_rgb(x)
    distance = np.minimum(rgb, 1-rgb).min(axis=-1)
    active = distance > 0
    w = smoothstep(distance/margin)
    out = x.copy()
    if active.any():
        transformed = color.rgb_lab(sample_texture(table, rgb[active]))
        out[active] += w[active, None]*(transformed-x[active])
    return out, w


def compress(rgb, margin=.08):
    """C1, reversible R->(0,1), identity in [m,1-m]; diagnostic only."""
    x = np.asarray(rgb, dtype=float)
    out = x.copy()
    lower, upper = x < margin, x > 1-margin
    out[lower] = margin*np.exp((x[lower]-margin)/margin)
    out[upper] = 1-margin*np.exp((1-margin-x[upper])/margin)
    return out


def decompress(rgb, margin=.08):
    x = np.asarray(rgb, dtype=float)
    if np.any(x <= 0) or np.any(x >= 1) or not np.isfinite(x).all():
        raise ValueError('Look reached closed-cube endpoint: inverse compression undefined')
    out = x.copy()
    lower, upper = x < margin, x > 1-margin
    out[lower] = margin+margin*np.log(x[lower]/margin)
    out[upper] = 1-margin-margin*np.log((1-x[upper])/margin)
    return out


def evaluate_lut(lut, q):
    return np.array([lut.evaluate(list(p))['pcs_Lab'] for p in q])


def domain_comparison(lab, table):
    rgb = color.lab_rgb(lab)
    outside = np.any((rgb < -1e-6) | (rgb > 1+1e-6), axis=1)
    clipped = color.rgb_lab(np.clip(rgb, 0, 1))
    bounded, weights = bounded_look(lab, table)
    compressed = compress(rgb)
    looked = sample_texture(table, compressed)
    inverse_valid = np.all((looked > 0) & (looked < 1), axis=1)
    reconstructed = decompress(looked[inverse_valid]) if inverse_valid.any() else np.empty((0, 3))
    inverse_lab = color.rgb_lab(reconstructed)
    represented = lab_codes(inverse_lab)
    return {
        'count': len(lab), 'out_of_srgb_count': int(outside.sum()),
        'hardclip_identity_bridge_deltaE76': stats(np.linalg.norm(clipped-lab, axis=1)),
        'reversible_compression': {
            'compressed_then_inverse_rgb_error': float(np.max(np.abs(decompress(compressed)-rgb))),
            'look_output_endpoint_rows_inverse_undefined': int((~inverse_valid).sum()),
            'valid_inverse_lab16_unrepresentable_rows': int(np.any((represented < 0) | (represented > 65535), axis=1).sum()),
            'valid_inverse_effect_deltaE76': stats(np.linalg.norm(inverse_lab-lab[inverse_valid], axis=1)),
            'note': 'No endpoint clamp to hide inverse singularity; inverse-valid subset only.'},
        'bounded_residual': {
            'margin_encoded_rgb': .08, 'identity_rows': int((weights == 0).sum()),
            'full_look_core_rows': int((weights == 1).sum()),
            'feather_rows': int(((weights > 0) & (weights < 1)).sum()),
            'out_of_srgb_max_abs_Lab_change': float(np.max(np.abs(bounded[outside]-lab[outside]))) if outside.any() else 0.,
            'effect_deltaE76': stats(np.linalg.norm(bounded-lab, axis=1)),
            'lab16_unrepresentable_rows': int(np.any((lab_codes(bounded) < 0) | (lab_codes(bounded) > 65535), axis=1).sum())},
        'note': 'Abstract profile-coordinate coverage; not affected fraction of actual photographs.'}


def refined_payload(lut, grid, transform):
    """Resample in native SHAPED coordinates; keep original input/output curves.

    Diagnostic in-memory payload only. A nested 65-grid is tested, not assumed to
    match a Capture One-supported grid or interpolation implementation.
    """
    if grid not in (33, 65):
        raise ValueError('Only controlled 33/65 nested grids supported')
    coords = np.stack(np.meshgrid(*([np.linspace(0, 1, grid)]*3), indexing='ij'), axis=-1).reshape(-1, 3)
    base = codes_lab(np.array([lut.clut_eval(p, 'tetrahedral') for p in coords])*65535)
    codes = lab_codes(transform(base))
    if not np.isfinite(codes).all() or np.any(codes < 0) or np.any(codes > 65535):
        raise ValueError('Unrepresentable refined PCS: no implicit clipping')
    prefix = bytearray(lut.raw[:lut.cb]); prefix[10] = grid
    return bytes(prefix)+np.floor(codes+.5).astype('>u2').tobytes()+lut.raw[lut.ce:]


def refinement_study(report_path, grid):
    if Path(report_path).exists():
        raise FileExistsError('Refusing report overwrite')
    data, cube = SOURCE.read_bytes(), VIVID.read_bytes()
    if sha(data) != SOURCE_SHA:
        raise ValueError('Native profile hash mismatch')
    lut = native_lut(data)
    raw_table = TextureLUT.from_cube(VIVID)
    table = TextureLUT(raw_table.values.astype(np.float16).astype(float))
    identity = audit.Lut16(refined_payload(lut, grid, lambda lab: lab))
    gated = audit.Lut16(refined_payload(lut, grid, lambda lab: bounded_look(lab, table)[0]))
    rng = np.random.default_rng(20261003)
    q = np.concatenate((rng.random((8192, 3)), np.repeat(np.linspace(0, 1, 257)[:, None], 3, axis=1)))
    base = evaluate_lut(lut, q)
    out = evaluate_lut(gated, q)
    expected, _ = bounded_look(base, table)
    outside = np.any((color.lab_rgb(base) < -1e-6) | (color.lab_rgb(base) > 1+1e-6), axis=1)
    r = {'version': 1, 'grid': grid, 'samples': len(q), 'icc_written': False,
         'sampling': 'Native shaped coordinates, original input/output curves; tetrahedral assumption',
         'identity_resampling_deltaE76': stats(np.linalg.norm(evaluate_lut(identity, q)-base, axis=1)),
         'bounded_residual_resampling_deltaE76': stats(np.linalg.norm(out-expected, axis=1)),
         'outside_domain_leak_deltaE76': stats(np.linalg.norm(out[outside]-base[outside], axis=1)),
         'native_input_tables_preserved': identity.raw[identity.ib:identity.ie] == lut.raw[lut.ib:lut.ie],
         'source_sha256': sha(data), 'cube_sha256': sha(cube),
         'source_unchanged': SOURCE.read_bytes() == data, 'cube_unchanged': VIVID.read_bytes() == cube,
         'limits': ['No claim of exact global zero leak', 'Higher grid acceptance/interpolation in C1 untested',
                    'A bounded Look at PCS still does not identify or invert downstream Standard rendering']}
    if not r['source_unchanged'] or not r['cube_unchanged']:
        raise AssertionError('Source changed')
    create_json(report_path, r)
    print(json.dumps(r, indent=2))


def create_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write('\n')


def build(out_dir):
    out_dir = Path(out_dir)
    if out_dir.exists():
        raise FileExistsError('Refusing existing output directory; choose a new version/run')
    data, cube_data = SOURCE.read_bytes(), VIVID.read_bytes()
    if sha(data) != SOURCE_SHA:
        raise ValueError('Native profile hash mismatch')
    original = native_lut(data)
    new_data = serialize_probe(data)
    modified = native_lut(new_data)
    old_tags, new_tags = dict(read_tags(data)), dict(read_tags(new_data))
    nodes = codes_lab(np.asarray(original.clut).reshape(-1, 3))
    raw_table = TextureLUT.from_cube(VIVID)
    table = TextureLUT(raw_table.values.astype(np.float16).astype(float), raw_table.name)
    rng = np.random.default_rng(20261003)
    q = np.concatenate((rng.random((8192, 3)), np.repeat(np.linspace(0, 1, 257)[:, None], 3, axis=1)))
    base = evaluate_lut(original, q)
    actual = evaluate_lut(modified, q)
    error = np.linalg.norm(actual-probe(base), axis=1)
    # Lab-code quantization + convex interpolation gives this global bound.
    quant_bound = float(np.linalg.norm([.5/652.8, .5/256, .5/256]))
    if error.max() > quant_bound+1e-10:
        raise AssertionError('Affine probe exceeded proven quantization error bound')
    lcms = audit.lcms_compare(new_data, q[:64].tolist())
    if not lcms.get('available'):
        raise RuntimeError('Installed LCMS required for independent probe parse/evaluation')
    lcms_error = np.linalg.norm(np.asarray(lcms['intents']['perceptual']['pcs_Lab'])-actual[:64], axis=1)
    # Quantify why sampling the bounded nonlinear H at native nodes is insufficient.
    gated_nodes, _ = bounded_look(nodes, table)
    gated_lut = audit.Lut16(replace_clut(original, gated_nodes))
    gated_actual = evaluate_lut(gated_lut, q)
    gated_ideal, w = bounded_look(base, table)
    outside = np.any((color.lab_rgb(base) < -1e-6) | (color.lab_rgb(base) > 1+1e-6), axis=1)
    neutral_L = np.linspace(1, 99, 99)
    neutral = np.column_stack((neutral_L, np.zeros((99, 2))))
    def synthetic_curve(x):
        y = x.copy(); y[:, 0] = 100*(y[:, 0]/100)**.8; return y
    report = {
        'version': 1, 'purpose': 'Single native camera ICC placement diagnostic; NOT a Vivid/Standard RAW deliverable',
        'source': input_label(SOURCE), 'source_sha256': sha(data),
        'cube': input_label(VIVID), 'cube_sha256': sha(cube_data),
        'vivid_samples_restored_binary16': True,
        'probe': {
            'file': 'LeicaQTyp116-LocalLooks-C1Single-OrderProbe-v1.icm',
            'description': PROBE_NAME, 'sha256': sha(new_data), 'bytes': len(new_data),
            'matrix_physical_Lab': PROBE_MATRIX.tolist(), 'determinant': float(np.linalg.det(PROBE_MATRIX)),
            'native_input_shapers_byte_identical': original.raw[original.ib:original.ie] == modified.raw[modified.ib:modified.ie],
            'native_output_tables_byte_identical': original.raw[original.ob:] == modified.raw[modified.ob:],
            'mft2_fixed_header_byte_identical': original.raw[:52] == modified.raw[:52],
            'icc_header_4_127_byte_identical': data[4:128] == new_data[4:128],
            'tag_signatures_and_order_unchanged': list(old_tags) == list(new_tags),
            'unchanged_tags': [sig.decode() for sig in old_tags if old_tags[sig] == new_tags[sig]],
            'analytic_max_deltaE76_bound_under_native_tetrahedral_or_trilinear': quant_bound,
            'off_grid_probe_vs_H_of_B_deltaE76': stats(error),
            'independent_lcms_vs_python_64_samples_deltaE76': stats(lcms_error),
            'lcms_version': lcms['encoded_version'], 'host_tested': False},
        'domain_options': {'native_33_cubed_nodes': domain_comparison(nodes, table),
                           'uniform_device_and_gray_8449': domain_comparison(base, table)},
        'bounded_residual_native_33_grid': {
            'icc_written': False, 'samples': len(q),
            'resampled_vs_ideal_deltaE76': stats(np.linalg.norm(gated_actual-gated_ideal, axis=1)),
            'outside_domain_interpolation_leak_deltaE76': stats(np.linalg.norm(gated_actual[outside]-base[outside], axis=1)),
            'reason_not_product': 'Pointwise H is exact identity outside sRGB; interpolation of changed neighboring CLUT nodes need not preserve this property.'},
        'order_negative_control': {
            'synthetic_downstream': 'D(L,a,b)=(100*(L/100)^0.8,a,b), not a claim about Capture One',
            'D_H_vs_H_D_neutral_axis_deltaE76': stats(np.linalg.norm(synthetic_curve(probe(neutral))-probe(synthetic_curve(neutral)), axis=1))},
        'host_contract': 'RAW -> N -> q -> native B -> D -> encoded sRGB export. Define Dlab = standard sRGB-to-Lab composed with D. Probe changes B to H(B). Compare Dlab(H(B)) vs H(Dlab(B)) only under controlled same q/D assumptions.',
        'no_vendor_application_execution': True, 'installed': False, 'published': False,
        'source_unchanged': SOURCE.read_bytes() == data, 'cube_unchanged': VIVID.read_bytes() == cube_data}
    if not report['source_unchanged'] or not report['cube_unchanged']:
        raise AssertionError('Read-only source changed')
    out_dir.mkdir(parents=True)
    with (out_dir/report['probe']['file']).open('xb') as f:
        f.write(new_data)
    create_json(out_dir/'diagnostic.json', report)
    print(json.dumps({'output': str(out_dir), 'probe_quantization_error': stats(error),
                      'bounded_33_grid_error': report['bounded_residual_native_33_grid']}, indent=2))


def read_tiff(path, stride=1):
    """Read the observed uncompressed RGB16 layout without Pillow's RGB8 downgrade.

    This bounded comparison rejects other layouts rather than silently converting.
    Output values must be encoded sRGB; caller explicitly acknowledges that.
    """
    path = Path(path)
    with Image.open(path) as image:
        tags = dict(image.tag_v2)
        width, height = image.size
        icc = image.info.get('icc_profile', b'')
        if image.format != 'TIFF' or tuple(tags.get(258, ())) != (16, 16, 16):
            raise ValueError('Expected RGB16 TIFF')
        if (tags.get(259), tags.get(262), tags.get(277), tags.get(284, 1), tags.get(274, 1)) != (1, 2, 3, 1, 1):
            raise ValueError('Require uncompressed chunky RGB TIFF orientation1')
        if tags.get(339, (1, 1, 1)) not in ((1, 1, 1), (1,), 1):
            raise ValueError('Unsigned integer samples required')
        if not icc:
            raise ValueError('Embedded ICC required')
        desc = ImageCms.getProfileDescription(ImageCms.ImageCmsProfile(io.BytesIO(icc))).strip()
        if 'srgb' not in desc.lower():
            raise ValueError('Embedded profile is not named sRGB; refusing to assume transfer')
    out = np.empty(((height+stride-1)//stride, (width+stride-1)//stride, 3), dtype=np.uint16)
    size = path.stat().st_size
    with path.open('rb') as f:
        endian = f.read(2)
        if endian not in (b'II', b'MM'):
            raise ValueError('Invalid byte order')
        offsets, lengths = tags[273], tags[279]
        if len(offsets) != len(lengths):
            raise ValueError('Strip metadata mismatch')
        y = 0
        for off, length in zip(offsets, lengths):
            rows = min(tags[278], height-y)
            if rows <= 0 or length != rows*width*6 or off+length > size:
                raise ValueError('Invalid strip')
            f.seek(off); raw = f.read(length)
            if len(raw) != length:
                raise ValueError('Truncated strip')
            arr = np.frombuffer(raw, dtype='<u2' if endian == b'II' else '>u2').reshape(rows, width, 3)
            indices = np.arange(y, y+rows)
            chosen = indices % stride == 0
            out[indices[chosen]//stride] = arr[chosen, ::stride]
            y += rows
        if y != height:
            raise ValueError('Missing rows')
    return out, {'path': str(path), 'sha256': sha(path.read_bytes()), 'size': [width, height],
                 'sample_stride': stride, 'icc_sha256': sha(icc), 'icc_description': desc}


def compare(baseline, candidate, report_path, stride, repeat=None):
    if Path(report_path).exists():
        raise FileExistsError('Refusing report overwrite')
    b, bm = read_tiff(baseline, stride)
    c, cm = read_tiff(candidate, stride)
    if bm['size'] != cm['size'] or bm['icc_sha256'] != cm['icc_sha256']:
        raise ValueError('Dimensions/output ICC differ; controlled comparison required')
    x = b.reshape(-1, 3).astype(float)/65535
    y = c.reshape(-1, 3).astype(float)/65535
    base_lab, actual_lab = color.rgb_lab(x), color.rgb_lab(y)
    predicted_lab = probe(base_lab)
    predicted_rgb = color.lab_rgb(predicted_lab)
    mask = np.all((x > .02) & (x < .98) & (y > .02) & (y < .98)
                  & (predicted_rgb > .02) & (predicted_rgb < .98), axis=1)
    delta = np.linalg.norm(actual_lab-predicted_lab, axis=1)
    r = {'version': 1, 'baseline': bm, 'candidate': cm,
         'hypothesis': 'Dlab(H(B(q))) = H(Dlab(B(q))); Dlab includes standard encoded-sRGB-export to Lab conversion; sampled non-clipped values only',
         'sampled_pixels': len(x), 'nonclipped_compared_pixels': int(mask.sum()),
         'candidate_vs_post_export_H_deltaE76_all': stats(delta),
         'candidate_vs_post_export_H_deltaE76_nonclipped': stats(delta[mask]),
         'candidate_vs_baseline_deltaE76': stats(np.linalg.norm(actual_lab-base_lab, axis=1)),
         'no_automatic_pass_fail': True,
         'limits': ['This is a commutation/identity-dispatch diagnostic, not complete pipeline identification.',
                    'Mismatch can arise from downstream D, profile-sensitive preprocessing q, interpolation or mismatched host settings.',
                    'Agreement on one photograph does not establish arbitrary-Look placement or local-processing invertibility.',
                    'Use explicit same Curve; Auto identity effects are a separate control.',
                    'sRGB transfer is explicitly user-acknowledged; profile description alone is not a cryptographic standard-space proof.']}
    if repeat:
        z, zm = read_tiff(repeat, stride)
        if zm['size'] != bm['size'] or zm['icc_sha256'] != bm['icc_sha256']:
            raise ValueError('Repeat dimensions/output ICC mismatch')
        r['repeat'] = zm
        r['baseline_repeat_deltaE76'] = stats(np.linalg.norm(color.rgb_lab(z.reshape(-1, 3).astype(float)/65535)-base_lab, axis=1))
    create_json(report_path, r)
    print(json.dumps({'report': str(report_path), 'nonclipped': r['candidate_vs_post_export_H_deltaE76_nonclipped']}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('build-diagnostic')
    p.add_argument('--profile', type=Path, default=SOURCE, help='User-provided original Q1 Generic; exact pinned profile required')
    p.add_argument('--cube', type=Path, default=VIVID, help='Vivid CUBE input')
    p.add_argument('--out-dir', type=Path, required=True)
    p = sub.add_parser('refine-domain')
    p.add_argument('--profile', type=Path, default=SOURCE, help='User-provided original Q1 Generic; exact pinned profile required')
    p.add_argument('--cube', type=Path, default=VIVID, help='Vivid CUBE input')
    p.add_argument('--report', type=Path, required=True)
    p.add_argument('--grid', type=int, choices=(33, 65), default=65)
    p = sub.add_parser('compare-host')
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--probe', type=Path, required=True)
    p.add_argument('--repeat', type=Path)
    p.add_argument('--report', type=Path, required=True)
    p.add_argument('--stride', type=int, default=4)
    p.add_argument('--acknowledge-encoded-srgb', action='store_true', required=True)
    args = parser.parse_args()
    configure_inputs(getattr(args, 'profile', None), getattr(args, 'cube', None))
    if args.command == 'build-diagnostic':
        build(args.out_dir)
    elif args.command == 'refine-domain':
        refinement_study(args.report, args.grid)
    else:
        if args.stride < 1:
            parser.error('stride must be >=1')
        compare(args.baseline, args.probe, args.report, args.stride, args.repeat)


if __name__ == '__main__':
    main()
