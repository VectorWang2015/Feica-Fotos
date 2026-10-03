#!/usr/bin/env python3
"""Private trial-ready native Q1 camera ICC with an explicit limited-domain Look.

This is an engineering preview, NOT Leica RAW Standard or a C1 host match claim.
Original profiles/resources are read-only. Every output is create-only.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import struct
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.c1 import c1_single_profile as c1

MARGIN = .08
FEATHER_RINGS = 2
NAME = 'Feica Fotos-Q1-VividPreview-Native33-v1'
PRIMARY_GRID = 33
UNSAFE_NODE_BUDGET = .75  # Explicit DeltaE76 magnitude, not an official gamut/strength rule.
BITS = tuple(itertools.product((0, 1), repeat=3))
LAB_HALF_CODE = np.array([.5/652.8, .5/256, .5/256])


def lattice(grid):
    if grid not in (33, 65):
        raise ValueError('Only nested33/65 supported')
    return np.stack(np.meshgrid(*([np.linspace(0, 1, grid)]*3), indexing='ij'), axis=-1).reshape(-1, 3)


def interpolate_nodes(nodes, positions, method='tetrahedral'):
    """Vector evaluator over ICC [R,G,B,channel], not the Look texture convention."""
    x = np.asarray(positions, dtype=float)
    if x.ndim != 2 or x.shape[1] != 3 or not np.isfinite(x).all():
        raise ValueError('Need finite Nx3 coordinates')
    if np.any(x < 0) or np.any(x > 1):
        raise ValueError('Profile coordinates outside [0,1]')
    n = nodes.shape[0]
    if nodes.shape != (n, n, n, 3):
        raise ValueError('Need cubic RGB output lattice')
    t = x*(n-1); lo = np.minimum(np.floor(t).astype(int), n-2); f = t-lo
    out = np.zeros_like(x)
    if method == 'tetrahedral':
        axes = np.argsort(-f, axis=1, kind='stable')
        fs = np.take_along_axis(f, axes, axis=1)
        ws = np.column_stack((1-fs[:, 0], fs[:, 0]-fs[:, 1], fs[:, 1]-fs[:, 2], fs[:, 2]))
        q = lo.copy(); rows = np.arange(len(x))
        for i in range(4):
            out += ws[:, i, None]*nodes[q[:, 0], q[:, 1], q[:, 2]]
            if i < 3:
                q[rows, axes[:, i]] += 1
    elif method == 'trilinear':
        for bits in BITS:
            q = lo+bits
            w = np.prod(np.where(np.array(bits), f, 1-f), axis=1)
            out += w[:, None]*nodes[q[:, 0], q[:, 1], q[:, 2]]
    else:
        raise ValueError('Unknown interpolation')
    return out


def shaped_coordinates(lut, q):
    q = np.asarray(q, dtype=float)
    if not np.isfinite(q).all() or np.any(q < 0) or np.any(q > 1):
        raise ValueError('Device RGB outside [0,1]')
    axis = np.linspace(0, 1, lut.nin)
    return np.column_stack([np.interp(q[:, k], axis, np.array(lut.inputs[k])/65535) for k in range(3)])


def quantize_lab(lab):
    values = c1.lab_codes(lab)
    if not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 65535):
        raise ValueError('Lab16 cannot represent candidate; no silent clipping')
    return c1.codes_lab(np.floor(values+.5))


def cell_linear_rgb_bounds(nodes, expansion=0.):
    """Conservative interval enclosure of Lab->linear-sRGB for every cell hull.

    Interpolated Lab is a convex combination of cell vertices. The three fXYZ
    coordinates are affine in Lab; inverse Lab f is monotone; XYZ->linearRGB is
    linear. Interval propagation therefore encloses every convex interpolation.
    This does not assume arbitrary curved Lab gamut has convex corner tests.
    """
    n = nodes.shape[0]
    # Quantized baseline vertices; bounds can optionally include per-Lab slack.
    lab = nodes
    fy = (lab[..., 0]+16)/116
    f = np.stack((fy+lab[..., 1]/500, fy, fy-lab[..., 2]/200), axis=-1)
    lo = np.full((n-1, n-1, n-1, 3), np.inf); hi = -lo
    for bits in BITS:
        cut = tuple(slice(b, b+n-1) for b in bits)+(slice(None),)
        lo = np.minimum(lo, f[cut]); hi = np.maximum(hi, f[cut])
    e = np.broadcast_to(np.asarray(expansion, dtype=float), (3,))
    fe = np.array([e[0]/116+e[1]/500, e[0]/116, e[0]/116+e[2]/200])
    lo -= fe; hi += fe
    delta = 6/29
    def inv(v):
        return np.where(v > delta, v**3, 3*delta*delta*(v-4/29))*c1.color.WHITE50
    xyz_lo, xyz_hi = inv(lo), inv(hi)
    matrix = np.linalg.inv(c1.color.RGB_XYZ65)@c1.color.CAT
    pos, neg = np.maximum(matrix, 0), np.minimum(matrix, 0)
    return xyz_lo@pos.T+xyz_hi@neg.T, xyz_hi@pos.T+xyz_lo@neg.T


def incident_safe_nodes(safe_cells):
    """A node may change only if every cell incident on that node is certified."""
    n = safe_cells.shape[0]+1
    if safe_cells.shape != (n-1,)*3:
        raise ValueError('Need cubic cell mask')
    permitted = np.ones((n, n, n), dtype=bool)
    for bits in BITS:
        cut = tuple(slice(b, b+n-1) for b in bits)
        permitted[cut] &= safe_cells
    return permitted


def erode_nodes(mask):
    """One 26-neighbor erosion; outside the lattice is inactive."""
    n = mask.shape[0]
    padded = np.pad(mask, 1, constant_values=False)
    out = np.ones_like(mask)
    for a, b, c in itertools.product(range(3), repeat=3):
        out &= padded[a:a+n, b:b+n, c:c+n]
    return out


def support_feather(permitted, rings=FEATHER_RINGS):
    if not isinstance(rings, int) or rings < 1 or rings > 4:
        raise ValueError('rings must be1..4')
    distance = np.zeros(permitted.shape, dtype=float)
    level = permitted.copy()
    for _ in range(rings):
        distance += level
        level = erode_nodes(level)
    return c1.smoothstep(distance/rings)


def prepare(lut, table, grid=65, margin=MARGIN, rings=FEATHER_RINGS, unsafe_budget=0.):
    coords = lattice(grid)
    old_nodes = c1.codes_lab(np.array(lut.clut).reshape(lut.grid, lut.grid, lut.grid, 3))
    raw_base = interpolate_nodes(old_nodes, coords).reshape(grid, grid, grid, 3)
    base = quantize_lab(raw_base)
    if not np.isfinite(unsafe_budget) or not 0 <= unsafe_budget <= 2:
        raise ValueError('unsafe_budget must be0..2DeltaE76')
    # For65 nested tetrahedral resampling, include native-node rounding slack.
    # This does NOT claim equivalence if a host uses native33 trilinear instead.
    low, high = cell_linear_rgb_bounds(base, LAB_HALF_CODE if grid == 65 else 0.)
    # Strict positive guard handles floating numerical enclosure tolerance.
    safe_cells = np.all((low >= 1e-12) & (high <= 1-1e-12), axis=-1)
    allowed = incident_safe_nodes(safe_cells)
    feather = support_feather(allowed, rings)
    ideal, domain_w = c1.bounded_look(base.reshape(-1, 3), table, margin)
    residual = (ideal-base.reshape(-1, 3)).reshape(base.shape)
    magnitude = np.linalg.norm(residual, axis=-1)
    if unsafe_budget > 0:
        cap = np.minimum(1., unsafe_budget/np.maximum(magnitude, 1e-30))
        feather = np.where(allowed, feather, cap)
    nodes = quantize_lab(base+feather[..., None]*residual)
    # All8vertices of an uncertified cell either remain unchanged (strict mode)
    # or have residual<=declared budget+half-code rounding (budget mode). Convex
    # interpolation preserves that norm bound, independent of tetra vs tri.
    change = np.linalg.norm(nodes-base, axis=-1)
    allowed_error = 0. if unsafe_budget == 0 else unsafe_budget+np.linalg.norm(LAB_HALF_CODE)+1e-10
    for bits in BITS:
        cut = tuple(slice(b, b+grid-1) for b in bits)
        if np.any(change[cut][~safe_cells] > allowed_error):
            raise AssertionError('Unsafe cell residual exceeds declared budget')
    return {'nodes': nodes, 'base': base, 'raw_base': raw_base,
            'safe_cells': safe_cells, 'allowed_nodes': allowed,
            'support_weight': feather, 'domain_weight': domain_w.reshape((grid,)*3),
            'total_weight': feather*domain_w.reshape((grid,)*3),
            'unsafe_budget_deltaE76': unsafe_budget,
            'unsafe_convex_interpolation_bound_deltaE76': allowed_error}


def serialize(data, nodes, name=NAME):
    """One native camera ICC; preserve all original shapers and non-Look tags."""
    lut = c1.native_lut(data); grid = nodes.shape[0]
    if grid not in (33, 65) or nodes.shape != (grid, grid, grid, 3):
        raise ValueError('Only33/65 RGB lattice permitted')
    codes = c1.lab_codes(nodes)
    if not np.isfinite(codes).all() or np.any(codes < 0) or np.any(codes > 65535):
        raise ValueError('Lab16 cannot represent candidate')
    prefix = bytearray(lut.raw[:lut.cb]); prefix[10] = grid
    payload = bytes(prefix)+np.floor(codes+.5).astype('>u2').tobytes()+lut.raw[lut.ce:]
    tags = c1.read_tags(data)
    result = bytearray(data[:128]+struct.pack('>I', len(tags))+bytes(12*len(tags)))
    for i, (sig, old) in enumerate(tags):
        new = c1.description(name) if sig == b'desc' else payload if sig == b'A2B0' else old
        result.extend(bytes((-len(result)) % 4)); offset = len(result)
        struct.pack_into('>4sII', result, 132+12*i, sig, offset, len(new)); result.extend(new)
    struct.pack_into('>I', result, 0, len(result))
    return bytes(result)


def coverage_report(prepared):
    w = prepared['total_weight']; sw = prepared['support_weight']
    return {'cells': int(prepared['safe_cells'].size), 'certified_cells': int(prepared['safe_cells'].sum()),
            'nodes': int(w.size), 'permitted_nodes': int(prepared['allowed_nodes'].sum()),
            'nonzero_effect_weight_nodes': int((w > 0).sum()),
            'full_effect_weight_nodes': int((w == 1).sum()),
            'domain_active_nodes_discarded_by_support': int(((prepared['domain_weight'] > 0) & (sw == 0)).sum()),
            'total_weight_stats': c1.stats(w[w > 0]),
            'note': 'Abstract ICC coordinate coverage, not fraction of real photographs.'}


def evaluate_candidate(lut, table, prepared, q, margin=MARGIN, method='tetrahedral'):
    shaped = shaped_coordinates(lut, q)
    old = c1.codes_lab(np.array(lut.clut).reshape(lut.grid, lut.grid, lut.grid, 3))
    native = interpolate_nodes(old, shaped, method)
    base = interpolate_nodes(prepared['base'], shaped, method)
    actual = interpolate_nodes(prepared['nodes'], shaped, method)
    ideal, weights = c1.bounded_look(native, table, margin)
    rgb = c1.color.lab_rgb(base)
    oog = np.any((rgb < 0) | (rgb > 1), axis=-1)
    lo = np.minimum(np.floor(shaped*(len(prepared['base'])-1)).astype(int), len(prepared['base'])-2)
    safe = prepared['safe_cells'][lo[:, 0], lo[:, 1], lo[:, 2]]
    if np.any(safe & oog):
        raise AssertionError('Interval certification violated by sample')
    effect = np.linalg.norm(actual-base, axis=-1)
    full = np.linalg.norm(ideal-native, axis=-1)
    useful = full > .25
    projection = np.zeros(len(q))
    d = ideal-native
    projection[useful] = np.sum((actual-base)[useful]*d[useful], axis=-1)/(full[useful]**2)
    return {'samples': len(q), 'interpolation': method,
            'declared_unsafe_cell_bound_deltaE76': prepared['unsafe_convex_interpolation_bound_deltaE76'],
            'sampled_safe_cells': int(safe.sum()),
            'sampled_outside_srgb': int(oog.sum()),
            'native_resample_deltaE76': c1.stats(np.linalg.norm(base-native, axis=-1)),
            'preview_vs_native_deltaE76': c1.stats(effect),
            'preview_vs_pointwise_bounded_look_deltaE76': c1.stats(np.linalg.norm(actual-ideal, axis=-1)),
            'outside_srgb_preview_vs_resampled_base_deltaE76': c1.stats(effect[oog]),
            'outside_srgb_preview_vs_native_deltaE76': c1.stats(np.linalg.norm(actual[oog]-native[oog], axis=-1)),
            'uncertified_cell_preview_vs_base_max': float(np.max(effect[~safe], initial=0)),
            'meaningful_ideal_effect_gt_quarter_deltaE': int(useful.sum()),
            'retained_effect_projection_on_meaningful_subset': c1.stats(projection[useful]),
            'effect_at_least_half_ideal_on_meaningful_subset': int((projection[useful] >= .5).sum()),
            'effect_fully_suppressed_on_meaningful_subset': int((effect[useful] == 0).sum())}


def proxy_preimages(nodes, target_lab, iterations=12):
    """Find abstract shaped-ICC coordinates for colors, never infer true RAW q.

    Nearest lattice node initializes a bounded damped Newton solve. A returned
    residual is mandatory; non-invertible/uncovered target colors remain failures.
    No target clipping is used. Coordinate bounds constrain this inverse search,
    not the color transform written into the profile.
    """
    n = nodes.shape[0]; flat = nodes.reshape(-1, 3)
    targets = np.asarray(target_lab, dtype=float)
    positions = np.empty_like(targets)
    squared = np.sum(flat*flat, axis=1)
    for start in range(0, len(targets), 128):
        t = targets[start:start+128]
        dist = np.sum(t*t, axis=1)[:, None]+squared[None, :]-2*t@flat.T
        nearest = np.argmin(dist, axis=1)
        positions[start:start+len(t)] = np.column_stack(np.unravel_index(nearest, (n,)*3))/(n-1)
    value = interpolate_nodes(nodes, positions)
    error = np.linalg.norm(value-targets, axis=1)
    for _ in range(iterations):
        jacobian = np.empty((len(targets), 3, 3))
        for axis in range(3):
            high = positions.copy(); low = positions.copy()
            high[:, axis] = np.minimum(1, high[:, axis]+1e-5)
            low[:, axis] = np.maximum(0, low[:, axis]-1e-5)
            jacobian[:, :, axis] = (interpolate_nodes(nodes, high)-interpolate_nodes(nodes, low))/(high[:, axis]-low[:, axis])[:, None]
        step = np.einsum('nij,nj->ni', np.linalg.pinv(jacobian, rcond=1e-8), targets-value)
        step = np.clip(step, -.08, .08)
        best = positions.copy(); best_value = value.copy(); best_error = error.copy()
        for scale in (1., .5, .25, .125):
            candidate = np.clip(positions+scale*step, 0, 1)
            v = interpolate_nodes(nodes, candidate)
            e = np.linalg.norm(v-targets, axis=1)
            improve = e < best_error
            best[improve] = candidate[improve]; best_value[improve] = v[improve]; best_error[improve] = e[improve]
        positions, value, error = best, best_value, best_error
    return positions, value, error


def existing_photo_diagnostic(table, lut, prepared, reference_tiff):
    image, metadata = c1.read_tiff(reference_tiff, stride=64)
    rgb = image.reshape(-1, 3).astype(float)/65535
    lab = c1.color.rgb_lab(rgb)
    full = c1.color.rgb_lab(c1.sample_texture(table, rgb))
    bounded, weights = c1.bounded_look(lab, table)
    native = c1.codes_lab(np.array(lut.clut).reshape(33, 33, 33, 3))
    shaped, _, _ = proxy_preimages(native, lab)
    # Recheck native33-grid preimages against each studied baseline; native33
    # remains exact, while65 studies include their resampling/quantization error.
    base = interpolate_nodes(prepared['base'], shaped)
    fit = np.linalg.norm(base-lab, axis=1)
    accepted = fit <= .05
    actual = interpolate_nodes(prepared['nodes'], shaped)
    effect = actual-base; desired = full-lab
    amplitude = np.linalg.norm(effect, axis=1); desired_amplitude = np.linalg.norm(desired, axis=1)
    meaningful = accepted & (desired_amplitude > .25)
    projection = np.sum(effect[meaningful]*desired[meaningful], axis=1)/(desired_amplitude[meaningful]**2)
    return {'input': metadata, 'purpose': 'Existing rendered C1 TIFF colors mapped to abstract native PCS preimages only; NOT true RAW q, NOT C1 host execution/prediction',
            'pointwise_margin': MARGIN, 'samples': len(rgb),
            'full_vivid_deltaE76': c1.stats(desired_amplitude),
            'pointwise_bounded_vivid_deltaE76': c1.stats(np.linalg.norm(bounded-lab, axis=-1)),
            'weight_stats': c1.stats(weights), 'full_weight_pixels': int((weights == 1).sum()),
            'zero_weight_pixels': int((weights == 0).sum()),
            'abstract_preimage_fit_deltaE76': c1.stats(fit), 'fit_threshold_deltaE76': .05,
            'accepted_proxy_colors': int(accepted.sum()), 'uncovered_proxy_colors': int((~accepted).sum()),
            'delivered_profile_effect_on_accepted_colors_deltaE76': c1.stats(amplitude[accepted]),
            'accepted_colors_effect_gt_quarter_deltaE': int((accepted & (amplitude > .25)).sum()),
            'accepted_meaningful_vivid_colors': int(meaningful.sum()),
            'retained_projection_on_meaningful_vivid': c1.stats(projection),
            'at_least_half_vivid_effect_on_meaningful_colors': int((projection >= .5).sum()),
            'suppressed_meaningful_colors': int((amplitude[meaningful] == 0).sum()),
            'missing': 'Export colors are only a plausibility surrogate. They are not the actual PCS before C1 tone processing, so these figures do not predict photograph coverage in C1.'}


def build(out_dir, release_dir, reference_tiff=None):
    out_dir, release_dir = Path(out_dir), Path(release_dir)
    if out_dir.exists() or release_dir.exists():
        raise FileExistsError('Choose new analysis and release directories; no overwrite')
    source, cube = c1.SOURCE.read_bytes(), c1.VIVID.read_bytes()
    if c1.sha(source) != c1.SOURCE_SHA:
        raise ValueError('Native profile hash mismatch')
    lut = c1.native_lut(source)
    raw = c1.TextureLUT.from_cube(c1.VIVID)
    table = c1.TextureLUT(raw.values.astype(np.float16).astype(float))
    rng = np.random.default_rng(20261021)
    q = np.concatenate((rng.random((32768, 3)), np.repeat(np.linspace(0, 1, 1025)[:, None], 3, axis=1)))
    studies = {}; selected = None
    for grid, rings, budget in [(33, 1, 0.), (33, 1, UNSAFE_NODE_BUDGET), (65, 1, 0.), (65, 2, 0.)]:
        p = prepare(lut, table, grid=grid, rings=rings, unsafe_budget=budget)
        key = f'grid{grid}_rings{rings}_budget{budget:g}'
        studies[key] = {'grid': grid, 'rings': rings, 'unsafe_node_budget_deltaE76': budget,
                        'coverage': coverage_report(p), 'holdout': evaluate_candidate(lut, table, p, q),
                        'trilinear_holdout': evaluate_candidate(lut, table, p, q[:4096], method='trilinear')}
        if grid == 33 or (grid == 65 and rings == 2):
            studies[key]['rendered_photo_proxy'] = (
                existing_photo_diagnostic(table, lut, p, reference_tiff)
                if reference_tiff is not None else
                {'status': 'NOT_RUN', 'reason': 'No optional --reference-tiff supplied; profile construction needs no photo.'})
        if grid == PRIMARY_GRID and budget == UNSAFE_NODE_BUDGET:
            selected = p; selected_key = key
    candidate = serialize(source, selected['nodes'])
    parsed = c1.audit.Lut16(dict(c1.read_tags(candidate))[b'A2B0'])
    cm = c1.audit.lcms_compare(candidate, q[:256].tolist())
    if not cm.get('available') or not cm['intents']['perceptual']['transform_created']:
        raise RuntimeError('LCMS interoperability required locally')
    pred = interpolate_nodes(selected['nodes'], shaped_coordinates(lut, q[:256]))
    cm_error = np.linalg.norm(np.array(cm['intents']['perceptual']['pcs_Lab'])-pred, axis=-1)
    if cm_error.max() > .1:
        raise AssertionError('Unexpected CMM discrepancy beyond preview validation budget')
    oldtags, tags = dict(c1.read_tags(source)), dict(c1.read_tags(candidate))
    report = {'version': 1, 'status': 'TRIAL-READY EXPERIMENTAL LOOK, NOT HOST-VERIFIED',
              'name': NAME, 'filename': 'LeicaQTyp116-FeicaFotos-VividPreview-Native33-v1.icm',
              'bytes': len(candidate), 'sha256': c1.sha(candidate),
              'source_sha256': c1.sha(source), 'vivid_cube_sha256': c1.sha(cube),
              'algorithm': {'look': 'Vivid100 original binary16 table, x*N-.5 sampling',
                            'base': 'native Q1 camera ICC, preserved input shapers and original33grid', 'grid': PRIMARY_GRID,
                            'domain_margin_encoded_srgb': MARGIN, 'support_feather_rings': 1,
                            'unsafe_node_residual_budget_deltaE76': UNSAFE_NODE_BUDGET,
                            'policy': 'Vivid endpoint100 at PCS; full bounded residual in certified interior; elsewhere magnitude attenuated to0.75DeltaE at nodes, giving convex-cell bound0.752867 including quantization; no color clipping',
                            'location': 'ICC Lab PCS approximation, not demonstrated Leica Standard or post-C1-curve stage'},
              'preservation': {'header4_127': source[4:128] == candidate[4:128],
                               'input_shapers': lut.raw[lut.ib:lut.ie] == parsed.raw[parsed.ib:parsed.ie],
                               'output_tables': lut.raw[lut.ob:] == parsed.raw[parsed.ob:],
                               'matrix': lut.matrix == parsed.matrix,
                               'unchanged_tags': [k.decode() for k in oldtags if oldtags[k] == tags[k]]},
              'studies': studies,
              'independent_lcms_256_deltaE76': c1.stats(cm_error), 'lcms_version': cm['encoded_version'],
              'selected_study': selected_key,
              'existing_rendered_tiff_surrogate': studies[selected_key]['rendered_photo_proxy'],
              'engineering_budget': {'outside_uncertified_cells': 'DELIVERED native33: originalbase unchanged at identical interpolation; Look effect<=0.75+0.002866365DeltaE76 for every convex interpolation in uncertified cells. Not zero; explicitly accepted approximation.',
                                     '65_study_limit': '65baseline uses native33TETRA subdivision. Quantization-only comparison to original applies only to matching nestedtetra; original33TRILINEAR can differ materially.65studies not delivered.',
                                     'lcms_vs_float_max_deltaE76_gate': .1,
                                     'full_look_preservation': 'not guaranteed; edge/unsafe regions intentionally keep native base; metrics explicitly reported'},
              'host_tested': False, 'installed': False, 'public_distribution_authorized': False,
              'source_unchanged': c1.SOURCE.read_bytes() == source, 'cube_unchanged': c1.VIVID.read_bytes() == cube}
    if not all(report['preservation'][k] for k in ('header4_127', 'input_shapers', 'output_tables', 'matrix')):
        raise AssertionError('Native structure changed')
    if not report['source_unchanged'] or not report['cube_unchanged']:
        raise AssertionError('Original changed')
    out_dir.mkdir(parents=True); release_dir.mkdir(parents=True)
    for folder in (out_dir, release_dir):
        with (folder/report['filename']).open('xb') as f:
            f.write(candidate)
    c1.create_json(out_dir/'validation.json', report)
    np.savez_compressed(out_dir/'support.npz', safe_cells=selected['safe_cells'], node_weight=selected['total_weight'])
    print(json.dumps({'report': str(out_dir/'validation.json'), 'candidate': str(release_dir/report['filename']),
                      'selected': studies[selected_key], 'lcms': report['independent_lcms_256_deltaE76']}, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profile', type=Path, default=c1.SOURCE, help='User-provided Q1 Generic; exact pinned source required')
    p.add_argument('--cube', type=Path, default=c1.VIVID, help='Vivid CUBE input')
    p.add_argument('--reference-tiff', type=Path, help='Optional rendered sRGB16 TIFF for a surrogate diagnostic, never required to build')
    p.add_argument('--out-dir', type=Path, required=True)
    p.add_argument('--release-dir', type=Path, required=True)
    a = p.parse_args()
    c1.configure_inputs(a.profile, a.cube)
    build(a.out_dir, a.release_dir, a.reference_tiff)


if __name__ == '__main__':
    main()
