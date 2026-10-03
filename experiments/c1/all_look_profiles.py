#!/usr/bin/env python3
"""Build the complete Q1 Look collection from one native camera ICC.

Finite-gamut rendering is explicit: preserve in-gamut linear RGB, project
out-of-gamut chroma toward neutral, execute the selected float Look recipe,
and store the resulting PCS values in the native 33-cube. Output files are new.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments.c1 import c1_single_profile as c1
from experiments.c1 import c1_single_look_preview as native
from experiments.c1.look_composition import FloatLookCompositor, build_variant_plan
from reproduction.ios_looks import color_spaces

POLICY_VERSION = 'finite-gamut-native33-v2'
PCS_L_MAX = 65535 / 652.8
PCS_AB_MAX = 65535 / 256 - 128


def rgb_xyz_matrix(space):
    if space == 'srgb':
        return c1.color.RGB_XYZ65
    if space == 'display-p3':
        primaries = np.array([[.68/.32, .265/.69, .15/.06],
                              [1., 1., 1.],
                              [0., .045/.69, .79/.06]])
        return primaries @ np.diag(np.linalg.solve(primaries, c1.color.WHITE65))
    raise ValueError('Unknown RGB space')


def project_linear_rgb(rgb, luminance_weights):
    """Project along a neutral-to-color ray; retain every in-gamut input exactly."""
    x = np.asarray(rgb, dtype=np.float64)
    if x.shape[-1:] != (3,) or not np.isfinite(x).all():
        raise ValueError('Finite RGB vectors required')
    weights = np.asarray(luminance_weights, dtype=float)
    if weights.shape != (3,) or np.any(weights < 0) or not np.isclose(weights.sum(), 1, atol=1e-12):
        raise ValueError('Normalized nonnegative luminance weights required')
    inside = np.all((x >= 0) & (x <= 1), axis=-1)
    neutral = np.clip(x @ weights, 0, 1)
    direction = x - neutral[..., None]
    bounds = np.full_like(direction, np.inf)
    np.divide(1-neutral[..., None], direction, out=bounds, where=direction > 0)
    lower = np.full_like(direction, np.inf)
    np.divide(-neutral[..., None], direction, out=lower, where=direction < 0)
    factor = np.minimum(1., np.minimum(bounds, lower).min(axis=-1))
    factor = np.maximum(factor, 0.)
    mapped = neutral[..., None] + factor[..., None]*direction
    # Roundoff only: the ray construction already bounds the mathematical value.
    if np.any(mapped < -2e-12) or np.any(mapped > 1+2e-12):
        raise AssertionError('Radial gamut bound violated')
    mapped = np.clip(mapped, 0, 1)
    mapped = np.where(inside[..., None], x, mapped)
    return mapped, ~inside


def lab_to_source(lab, space):
    """Native D50 PCS -> finite working RGB -> encoded-sRGB compositor operand."""
    x = np.asarray(lab, dtype=float).reshape(-1, 3)
    matrix = rgb_xyz_matrix(space)
    linear = c1.color.lab_xyz(x) @ c1.color.CAT.T @ np.linalg.inv(matrix).T
    mapped, changed = project_linear_rgb(linear, matrix[1])
    encoded = c1.color.encode(mapped)
    source = color_spaces.convert_rgb(encoded, space, 'srgb', clip=False)
    return source, changed


def fit_lab16(lab):
    """Keep L and chroma direction when fitting the ICC v2 Lab code range."""
    x = np.asarray(lab, dtype=float)
    if x.shape[-1:] != (3,) or not np.isfinite(x).all():
        raise ValueError('Finite Lab vectors required')
    if np.any(x[..., 0] < -1e-8) or np.any(x[..., 0] > 100+1e-8):
        raise ValueError('Finite RGB render produced invalid luminance')
    out = x.copy()
    out[..., 0] = np.clip(out[..., 0], 0, 100)
    ab = x[..., 1:]
    upper = np.full_like(ab, np.inf)
    lower = np.full_like(ab, np.inf)
    np.divide(PCS_AB_MAX, ab, out=upper, where=ab > 0)
    np.divide(-128., ab, out=lower, where=ab < 0)
    factor = np.minimum(1., np.minimum(upper, lower).min(axis=-1))
    out[..., 1:] *= factor[..., None]
    # Keep conversion roundoff inside the exact representable code range.
    out[..., 1:] = np.clip(out[..., 1:], -128., PCS_AB_MAX)
    return out, factor < 1


def rendered_to_lab(rgb, space):
    matrix = rgb_xyz_matrix(space)
    working = color_spaces.convert_rgb(np.asarray(rgb, dtype=float), 'srgb', space, clip=False)
    linear = c1.color.decode(working)
    mapped, projected = project_linear_rgb(linear, matrix[1])
    lab = c1.color.xyz_lab(mapped @ matrix.T @ np.linalg.inv(c1.color.CAT).T)
    representable, reduced = fit_lab16(lab)
    return representable, projected, reduced


def render_pcs(lab, compositor, variant):
    binding = compositor.bindings[variant.look_id]
    source, input_projection = lab_to_source(lab, binding['input_space'])
    result = compositor.apply(source, variant.look_id, variant.strength, variant.color_filter)
    output, output_projection, lab_reduction = rendered_to_lab(result, binding['output_space'])
    return output, {'input_projection_count': int(input_projection.sum()),
                    'output_projection_count': int(output_projection.sum()),
                    'lab16_chroma_reduction_count': int(lab_reduction.sum())}


def preservation(source, output):
    a, b = c1.native_lut(source), c1.native_lut(output)
    original, current = dict(c1.read_tags(source)), dict(c1.read_tags(output))
    return {'header_except_size': source[4:128] == output[4:128],
            'input_shapers': a.raw[a.ib:a.ie] == b.raw[b.ib:b.ie],
            'matrix': a.matrix == b.matrix,
            'output_tables': a.raw[a.ob:] == b.raw[b.ob:],
            'native_grid': b.grid == 33,
            'tag_set': set(original) == set(current),
            'non_color_tags': all(original[k] == current[k] for k in original if k not in (b'A2B0', b'desc'))}


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def build(profile_path, resource_dir, out_dir, validation_path, *, require_lcms=True):
    out_dir, validation_path = Path(out_dir), Path(validation_path)
    if out_dir.exists() or validation_path.exists():
        raise FileExistsError('Use new output and validation paths')
    profile_path = Path(profile_path)
    source = profile_path.read_bytes()
    if sha_bytes(source) != c1.SOURCE_SHA:
        raise ValueError('Q1 source profile SHA-256 differs from the supported base')
    lut = c1.native_lut(source)
    base_nodes = c1.codes_lab(np.array(lut.clut).reshape(33, 33, 33, 3))
    compositor = FloatLookCompositor(resource_dir)
    plan = build_variant_plan()
    rng = np.random.default_rng(20261004)
    q = np.concatenate([rng.random((2048, 3)), np.repeat(np.linspace(0, 1, 257)[:, None], 3, axis=1)])
    shaped = native.shaped_coordinates(lut, q)
    base_tetra = native.interpolate_nodes(base_nodes, shaped)
    base_tri = native.interpolate_nodes(base_nodes, shaped, 'trilinear')
    cm_q = np.concatenate([rng.uniform(.01, .99, (96, 3)), [[0., 0., 0.], [1., 1., 1.]]])
    cm_shaped = native.shaped_coordinates(lut, cm_q)
    out_dir.mkdir(parents=True)
    records = []
    for index, variant in enumerate(plan, 1):
        transformed, projections = render_pcs(base_nodes.reshape(-1, 3), compositor, variant)
        nodes = native.quantize_lab(transformed).reshape(base_nodes.shape)
        output = native.serialize(source, nodes, variant.description)
        kept = preservation(source, output)
        if not all(kept.values()):
            raise AssertionError('Native structure preservation failed')
        node_quant_error = float(np.max(np.linalg.norm(nodes.reshape(-1, 3)-transformed, axis=1)))
        parsed = c1.native_lut(output)
        if parsed.raw[parsed.cb:parsed.ce] == lut.raw[lut.cb:lut.ce]:
            raise AssertionError('Look unexpectedly produced an unchanged native CLUT')
        continuous_tetra, _ = render_pcs(base_tetra, compositor, variant)
        continuous_tri, _ = render_pcs(base_tri, compositor, variant)
        approximate_tetra = native.interpolate_nodes(nodes, shaped)
        approximate_tri = native.interpolate_nodes(nodes, shaped, 'trilinear')
        cm = c1.audit.lcms_compare(output, cm_q.tolist())
        cm_stats = {'available': cm['available']}
        if not cm['available'] and require_lcms:
            raise RuntimeError('LittleCMS2 required for release validation')
        if cm['available']:
            # Relative colorimetric is the direct PCS comparison. Perceptual
            # intent can adapt a lifted-black v2 input to the Lab destination;
            # record that separately rather than calling it a CLUT mismatch.
            expected = native.interpolate_nodes(nodes, cm_shaped)
            cm_stats.update({'version': cm['encoded_version'],
                             'reference_intent': 'relative_colorimetric',
                             'intent_deltaE76': {}})
            for intent, result in cm['intents'].items():
                if not result['transform_created']:
                    raise AssertionError('LittleCMS could not create an intent transform')
                got = np.asarray(result['pcs_Lab'])
                if not np.isfinite(got).all():
                    raise AssertionError('LittleCMS produced a nonfinite PCS value')
                cm_stats['intent_deltaE76'][intent] = c1.stats(np.linalg.norm(got-expected, axis=1))
            cm_stats['deltaE76'] = cm_stats['intent_deltaE76']['relative_colorimetric']
            perceptual = np.array(cm['intents']['perceptual']['pcs_Lab'])
            relative = np.array(cm['intents']['relative_colorimetric']['pcs_Lab'])
            cm_stats['perceptual_vs_relative_deltaE76'] = c1.stats(np.linalg.norm(perceptual-relative, axis=1))
            if cm_stats['deltaE76']['max'] > .5:
                raise AssertionError('LittleCMS relative PCS and ICC evaluator disagree')
        path = out_dir / variant.relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as f:
            f.write(output)
        record = variant.to_dict()
        record.update({'bytes': len(output), 'sha256': sha_bytes(output), 'preservation': kept,
                       'node_quantization_max_deltaE76': node_quant_error,
                       'node_projections': projections,
                       'holdout_samples': len(q),
                       'holdout_tetra_error_deltaE76': c1.stats(np.linalg.norm(approximate_tetra-continuous_tetra, axis=1)),
                       'holdout_trilinear_error_deltaE76': c1.stats(np.linalg.norm(approximate_tri-continuous_tri, axis=1)),
                       'effect_vs_base_deltaE76': c1.stats(np.linalg.norm(approximate_tetra-base_tetra, axis=1)),
                       'lcms': cm_stats})
        records.append(record)
        print(f'[{index}/{len(plan)}] {variant.relative_path}', flush=True)
    if profile_path.read_bytes() != source:
        raise AssertionError('Source profile changed')
    manifest = {'schema': 1, 'created_utc': datetime.now(timezone.utc).isoformat(),
                'policy': POLICY_VERSION, 'source_profile_sha256': sha_bytes(source),
                'profile_count': len(records), 'main_look_count': 21,
                'main_strengths': [25, 50, 75, 100], 'dual_additional_strength': 0,
                'attachment_looks': 6, 'attachment_colors': 5, 'attachment_strengths': [25, 50, 75, 100],
                'host_tested': False, 'source_unchanged': True,
                'domain_policy': 'Radial linear working-RGB projection toward clamped luminance-neutral, complete float Look then output gamut projection; Lab chroma scaled only for ICC v2 representability.',
                'base_policy': 'Original 33-cube structure, input shapers, matrix, output curves and non-color tags retained; output colors replaced by selected Look.',
                'profiles': [{k: v for k, v in r.items() if k not in ('preservation', 'node_projections', 'holdout_tetra_error_deltaE76', 'holdout_trilinear_error_deltaE76', 'effect_vs_base_deltaE76', 'lcms')} for r in records]}
    with (out_dir/'MANIFEST.json').open('x', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2); f.write('\n')
    validation_path.parent.mkdir(parents=True, exist_ok=True)
    with validation_path.open('x', encoding='utf-8') as f:
        json.dump({'manifest': manifest, 'validation': records}, f, ensure_ascii=False, indent=2); f.write('\n')
    print(json.dumps({'output': str(out_dir), 'profiles': len(records), 'validation': str(validation_path)}))
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profile', type=Path, required=True)
    p.add_argument('--resource-dir', type=Path, default=ROOT/'filters/looks')
    p.add_argument('--out-dir', type=Path, required=True)
    p.add_argument('--validation', type=Path, required=True)
    p.add_argument('--allow-missing-lcms', action='store_true', help='Development only; release builds require LittleCMS')
    a = p.parse_args()
    build(a.profile, a.resource_dir, a.out_dir, a.validation, require_lcms=not a.allow_missing_lcms)


if __name__ == '__main__':
    main()
