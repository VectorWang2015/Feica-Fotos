#!/usr/bin/env python3
"""Independent all-family validation, not a request for per-Look device goldens.

The default suite loads CUBE values independently and uses synthetic RGB only.
Optional external half-resource and photo-fixture checks are opt-in. The oracle
sums eight barycentric corner contributions and uses fixed
rational standard-primary matrices. Production imports occur only in run_suite
as the implementation under test. No vendor code is executed and no images are
written; reports contain aggregate data, resource hashes, and source identities.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import itertools
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
CATALOG = ROOT / 'apps/feica_fotos/assets/look-catalog.json'
CUBES = ROOT / 'filters/looks'
STRENGTHS = (0, 25, 37.25, 50, 75, 100)
# Exact rational standard-primary matrices, independent of production's xy solve.
SRGB_XYZ = np.array([[506752/1228815, 87881/245763, 12673/70218],
                     [87098/409605, 175762/245763, 12673/175545],
                     [7918/409605, 87881/737289, 1001167/1053270]])
P3_XYZ = np.array([[608311/1250200, 189793/714400, 198249/1000160],
                   [35783/156275, 247089/357200, 198249/2500400],
                   [0, 32229/714400, 5220557/5000800]])


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_cube(path, restore_half=False):
    """Separate CUBE parser: numeric rows to R-fast flat table, no production API."""
    size = None
    numbers = []
    for line in Path(path).read_text(encoding='utf-8-sig').splitlines():
        record = line.partition('#')[0].strip().split()
        if not record:
            continue
        if record[0] == 'TITLE':
            continue
        if record[0] == 'LUT_3D_SIZE':
            if size is not None:
                raise ValueError('duplicate dimension')
            size = int(record[1])
        elif record[0] in ('DOMAIN_MIN', 'DOMAIN_MAX'):
            if [float(v) for v in record[1:]] != ([0.] * 3 if record[0] == 'DOMAIN_MIN' else [1.] * 3):
                raise ValueError('nonunit domain')
        else:
            if len(record) != 3:
                raise ValueError('invalid numeric row')
            numbers.append(tuple(map(float, record)))
    if size is None or size < 2 or len(numbers) != size**3:
        raise ValueError('wrong number of nodes')
    values = np.array(numbers, dtype=np.float64).reshape(size, size, size, 3)
    if not np.isfinite(values).all():
        raise ValueError('nonfinite node')
    return values.astype(np.float16).astype(np.float64) if restore_half else values


def texture(table, rgb):
    """Eight-corner weighted sum; intentionally not production's nested lerp."""
    x = np.asarray(rgb, dtype=np.float64)
    if x.shape[-1] != 3 or not np.isfinite(x).all():
        raise ValueError('finite RGB required')
    n = table.shape[0]
    coordinates = np.maximum(0., np.minimum(n - 1., x * n - .5))
    cell = coordinates.astype(np.int64)
    fraction = coordinates - cell
    result = np.zeros_like(x)
    flat_table = table.reshape(-1, 3)
    for bits in itertools.product((0, 1), repeat=3):
        index = np.minimum(cell + np.array(bits), n - 1)
        weight = np.prod(np.where(bits, fraction, 1 - fraction), axis=-1)
        flat_index = index[..., 0] + n * index[..., 1] + n*n * index[..., 2]
        result += flat_table[flat_index] * weight[..., None]
    return result


def transfer(x, decode):
    x = np.asarray(x, dtype=np.float64)
    absolute = np.abs(x)
    if decode:
        value = np.where(absolute <= .04045, absolute / 12.92,
                         ((absolute + .055) / 1.055)**2.4)
    else:
        value = np.where(absolute <= .0031308, absolute * 12.92,
                         1.055 * absolute**(1/2.4) - .055)
    return np.copysign(value, x)


def convert(x, source, target):
    if source == target:
        return np.array(x, dtype=np.float64, copy=True)
    matrices = {'srgb': SRGB_XYZ, 'display-p3': P3_XYZ}
    # Separate XYZ leg rather than production's single precombined matrix.
    xyz = transfer(x, True) @ matrices[source].T
    linear = xyz @ np.linalg.inv(matrices[target]).T
    return transfer(linear, False)


def quantize(x):
    clipped = np.maximum(0., np.minimum(1., np.asarray(x)))
    return np.trunc(clipped * 255 + .5).astype(np.uint8)


def compose(source, binding, tables, strength, attachment=None):
    """Independent declared-recipe float pipeline; no production helpers."""
    source = np.asarray(source, dtype=np.float64)
    recipe = binding['recipe']
    if recipe == 'identity' or (recipe == 'source_to_primary' and strength == 0):
        return source.copy()
    working = convert(source, 'srgb', binding['input_space'])
    if attachment is not None:
        working = texture(tables[attachment['cube']['filename']], working)
    primary = texture(tables[binding['primary_cube']['filename']], working)
    weight = strength / 100
    if recipe == 'secondary_to_primary':
        secondary = texture(tables[binding['secondary_cube']['filename']], working)
        tone = secondary * (1-weight) + primary * weight
    elif recipe == 'source_to_primary':
        tone = primary
    else:
        raise ValueError('unknown recipe')
    result = convert(tone, binding['output_space'], 'srgb')
    if recipe == 'source_to_primary' and strength != 100:
        result = source * (1-weight) + result * weight
    return result


def family(binding, attachment=False):
    if binding['recipe'] == 'identity':
        return 'identity'
    base = ('dual' if binding['recipe'] == 'secondary_to_primary' else 'single')
    domain = 'p3' if binding['input_space'] == 'display-p3' else 'srgb'
    return f'{base}-{domain}' + ('-prefilter' if attachment else '')


def synthetic_rgb8():
    ramp = np.arange(256, dtype=np.uint8)
    records = [np.array(list(itertools.product((0, 255), repeat=3)), dtype=np.uint8),
               np.repeat(ramp[:, None], 3, axis=1)]
    for channel in range(3):
        for background in (0, 255):
            row = np.full((256, 3), background, dtype=np.uint8)
            row[:, channel] = ramp
            records.append(row)
    nodes = np.rint(np.linspace(0, 255, 17)).astype(np.uint8)
    records.append(np.array(list(itertools.product(nodes, repeat=3)), dtype=np.uint8))
    records.append(np.random.default_rng(2601002).integers(0, 256, (4096, 3), dtype=np.uint8))
    return np.concatenate(records)


def boundary_points(n):
    """Around clamp edges, every texel center, and random extended coordinates."""
    centers = (np.arange(n) + .5) / n
    values = np.unique(np.r_[[-.1, 0., 1., 1.1], centers,
                            centers - 1e-12, centers + 1e-12])
    records = []
    for channel in range(3):
        points = np.full((len(values), 3), .37)
        points[:, channel] = values
        records.append(points)
    records.append(np.random.default_rng(n).uniform(-.1, 1.1, (1024, 3)))
    return np.concatenate(records)


def sample_image(path, limit=4096, orientation_override=None):
    with Image.open(path) as image:
        if orientation_override is not None:
            if orientation_override != 6 or image.getexif().get(274) not in (None, 1):
                raise ValueError('only explicit orientation6 for metadata-less extracted preview is supported')
            upright = image.transpose(Image.Transpose.ROTATE_270).convert('RGB')
        else:
            upright = ImageOps.exif_transpose(image).convert('RGB')
        pixels = np.asarray(upright).reshape(-1, 3)
        indices = np.linspace(0, len(pixels)-1, min(limit, len(pixels)), dtype=np.int64)
        return pixels[indices].copy(), {'path': str(path), 'sha256': sha(path),
                                     'upright_size': list(upright.size),
                                     'orientation_override': orientation_override,
                                     'sample_pixels': len(indices)}


def error_stats(actual, expected):
    d = np.abs(np.asarray(actual, dtype=float) - np.asarray(expected, dtype=float))
    return {'mae': float(d.mean()), 'max': float(d.max()),
            'p99': float(np.percentile(d, 99))}


def load_photo_fixtures(manifest_path):
    """Read an explicit external manifest; paths resolve beside the manifest.

    Schema: {"images": [{"path": "input.jpg", "orientation_override": 6}],
    "anchors": [{"look": "vivid", "strength": 100, "source": "input.jpg",
    "reference": "reference.jpg", "source_orientation_override": 6}]}.
    Overrides are optional. References are user-supplied comparison images:
    supplying one does not establish official provenance or host equivalence.
    No file contents or photographic pixels are copied into the repository.
    """
    if manifest_path is None:
        return [], []
    manifest_path = Path(manifest_path)
    raw = json.loads(manifest_path.read_text(encoding='utf-8'))
    if not isinstance(raw, dict) or set(raw) - {'images', 'anchors'}:
        raise ValueError('Photo manifest must contain only images/anchors arrays')
    def path(value):
        if not isinstance(value, str) or not value:
            raise ValueError('Fixture paths must be nonempty strings')
        p = Path(value)
        return p if p.is_absolute() else manifest_path.parent / p
    images, anchors = [], []
    for row in raw.get('images', []):
        images.append((path(row['path']), row.get('orientation_override')))
    for row in raw.get('anchors', []):
        strength = float(row['strength'])
        if not np.isfinite(strength) or not 0 <= strength <= 100:
            raise ValueError('Anchor strength must be finite and within 0..100')
        anchors.append((row['look'], strength, path(row['source']),
                        path(row['reference']), row.get('source_orientation_override')))
    return images, anchors


def run_suite(out_dir, resource_dir=CUBES, source_half_dir=None, photo_fixtures=None):
    """Exercise app+sampler as observed implementation, oracle above as reference."""
    from apps.feica_fotos.engine import ImageEngine
    from reproduction.ios_looks import renderer, color_spaces
    resource_dir = Path(resource_dir)
    source_half_dir = Path(source_half_dir) if source_half_dir is not None else None
    external_images, external_anchors = load_photo_fixtures(photo_fixtures)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=False)
    raw = json.loads(CATALOG.read_text())
    catalog_sha = sha(CATALOG)
    specs = {row['id']: row for row in raw['looks']}
    filters = {row['id']: row for row in raw['color_filters']}
    cube_specs = {}
    for row in raw['looks']:
        for key in ('primary_cube', 'secondary_cube'):
            item = row[key]
            if item:
                if item['filename'] in cube_specs and cube_specs[item['filename']] != item:
                    raise ValueError('conflicting table policy')
                cube_specs[item['filename']] = item
    for row in filters.values():
        cube_specs[row['cube']['filename']] = row['cube']
    tables, table_results = {}, []
    for name, spec in cube_specs.items():
        path = resource_dir/name
        if sha(path) != spec['sha256']:
            raise ValueError('resource mismatch: '+name)
        table = load_cube(path, spec['restore_half'])
        n = spec['grid_size']
        if table.shape != (n, n, n, 3):
            raise ValueError('grid mismatch: '+name)
        source_check = {'status': 'NOT_RUN', 'reason': 'No optional --source-half-dir supplied'}
        if source_half_dir is not None:
            data_path = source_half_dir/Path(name).with_suffix('.data')
            data = np.frombuffer(data_path.read_bytes(), dtype='<f2').reshape(n,n,n,4)
            if not np.isfinite(data).all() or not (data[..., 3] == 1).all():
                raise ValueError('source RGBA16F contract failed')
            half = data[..., :3].astype(np.float64)
            delta = float(np.max(np.abs(table-half)))
            if (spec['restore_half'] and delta != 0) or delta > 5.1e-10:
                raise ValueError('recovered sample/source mismatch')
            source_check = {'status': 'PASS', 'source_sha256': sha(data_path),
                            'max_source_half_difference': delta}
        points = boundary_points(n)
        expected = texture(table, points)
        observed = renderer.sample_texture(renderer.TextureLUT(table), points)
        numeric_error = float(np.max(np.abs(expected-observed)))
        tables[name] = table
        table_results.append({'filename': name, 'cube_sha256': sha(path),
                              'source_half_crosscheck': source_check, 'grid_size': n,
                              'restore_half': spec['restore_half'],
                              'boundary_points': len(points),
                              'sampler_float_max_error': numeric_error,
                              'pass': numeric_error < 3e-14})
    groups = [('synthetic', synthetic_rgb8())]
    image_records = []
    for n, (path, orientation) in enumerate(external_images):
        data, record = sample_image(path, orientation_override=orientation)
        groups.append((f'external_{n}', data)); image_records.append(record)
    source = np.concatenate([data for _, data in groups])
    frozen_source = source.copy()
    source_float = source.astype(np.float64)/255
    engine = ImageEngine(resource_dir)
    cases = []
    for row in raw['looks']:
        strengths = (100,) if row['recipe'] == 'identity' else STRENGTHS
        for attachment_id in (None, *row['filter_options']):
            attachment = None if attachment_id is None else filters[attachment_id]
            for strength in strengths:
                expected_float = compose(source_float, row, tables, strength, attachment)
                expected = quantize(expected_float)
                observed = engine.render(source[None, ...], row['id'], strength,
                                         color_filter=attachment_id)[0]
                difference = np.abs(observed.astype(np.int16)-expected.astype(np.int16))
                # Different floating association may move an exactly-half byte
                # boundary by ~1e-14. Count and bound these; never hide a whole
                # byte difference away from a quantizer tie.
                codes = np.clip(expected_float, 0, 1)*255
                tie_distance = np.abs(codes-(np.floor(codes)+.5))
                unjustified = (difference != 0) & (tie_distance > 1e-9)
                cases.append({'look':row['id'], 'family':family(row, attachment is not None),
                              'attachment':attachment_id, 'strength':strength,
                              'pixels':len(source), 'max_byte_error':int(difference.max()),
                              'different_channels':int(np.count_nonzero(difference)),
                              'quantizer_tie_differences':int(np.count_nonzero((difference != 0)&~unjustified)),
                              'non_tie_differences':int(np.count_nonzero(unjustified)),
                              'pass':int(difference.max()) <= 1 and not unjustified.any()})
    # Independent conversion comparison with extended values: clipping belongs
    # to final byte encoding, not the P3 return leg before source blending.
    conversion_points = np.random.default_rng(50).uniform(-.2, 1.2, (4096, 3))
    conversion_checks = []
    for a,b in [('srgb','display-p3'), ('display-p3','srgb')]:
        oracle = convert(conversion_points,a,b)
        actual = color_spaces.convert_rgb(conversion_points,a,b,clip=False)
        conversion_checks.append({'source':a,'target':b,'float_max_error':float(np.max(np.abs(oracle-actual))),
                                  'pass':bool(np.allclose(oracle,actual,atol=2e-13,rtol=0))})
    # Quantify wrong-family shortcuts instead of requiring every sibling's photo.
    negative = []
    for key in ('classic','contemporary','natural','greg','monochrome_high_contrast'):
        row = specs[key]
        correct = compose(source_float,row,tables,50, filters['red'] if key=='monochrome_high_contrast' else None)
        if key in ('classic','contemporary'):
            full = compose(source_float,row,tables,100)
            wrong = source_float + (full-source_float)*(.5*.65)
            label = 'erroneous extra opacity65 multiplier'
        elif key=='natural':
            copied=dict(row,input_space='srgb',output_space='srgb')
            wrong=compose(source_float,copied,tables,50);label='omit P3 conversions'
        elif key=='greg':
            copied=dict(row,primary_cube=row['secondary_cube'],secondary_cube=row['primary_cube'])
            correct=compose(source_float,row,tables,25)
            wrong=compose(source_float,copied,tables,25);label='reverse endpoints at25'
        else:
            # Wrong order: Tone before red rather than red before Tone.
            y=texture(tables[row['primary_cube']['filename']],source_float)
            wrong=source_float*.5+texture(tables[filters['red']['cube']['filename']],y)*.5
            label='Tone before prefilter'
        negative.append({'look':key,'wrong_shortcut':label,'difference_in_byte_units':error_stats(wrong*255,correct*255)})
    # Optional user-supplied photo comparisons; provenance is not inferred.
    anchors=[]
    for key,strength,src,dst,orientation in external_anchors:
        if key not in specs:
            raise ValueError('Unknown external anchor Look: '+str(key))
        x,xi=sample_image(src,16384,orientation_override=orientation)
        y,yi=sample_image(dst,16384)
        if xi['upright_size'] != yi['upright_size']:
            raise ValueError('anchor dimensions differ')
        prediction=quantize(compose(x.astype(float)/255,specs[key],tables,strength))
        anchors.append({'look':key,'strength':strength,'source':xi,'reference_output':yi,
                        'oracle_vs_reference_rgb8':error_stats(prediction,y),
                        'meaning':'user-supplied photo comparison; official provenance and bit identity not established'})
    families=defaultdict(lambda:{'looks':set(),'cases':0,'passes':0})
    for c in cases:
        families[c['family']]['looks'].add(c['look'])
        families[c['family']]['cases']+=1
        families[c['family']]['passes']+=int(c['pass'])
    for f in families.values():f['looks']=sorted(f['looks'])
    report={'schema':1,'catalog_sha256':catalog_sha,
            'independence':'oracle independent eight-corner sum, CUBE parser, exact rational matrices; production used only as observed SUT',
            'optional_checks': {
                'source_half_crosscheck': 'PASS' if source_half_dir is not None else 'NOT_RUN',
                'photo_samples': 'RUN' if external_images else 'NOT_RUN',
                'photo_anchors': 'RUN' if external_anchors else 'NOT_RUN',
                'official_host_validation': 'NOT_ESTABLISHED'},
            'coverage':{'main_looks':len(raw['looks'])-1,'original':1,'tables':len(tables),
                        'attachments':len(filters),'strengths':list(STRENGTHS),'cases':len(cases),
                        'points_per_case':len(source),'point_groups':{name:len(x) for name,x in groups},
                        'total_rendered_pixels_compared':sum(c['pixels'] for c in cases)},
            'families':dict(families),'image_inputs':image_records,'table_checks':table_results,'conversion_checks':conversion_checks,
            'cases':cases,'external_photo_anchors':anchors,'negative_controls':negative,
            'source_unchanged':bool(np.array_equal(source,frozen_source)),
            'catalog_unchanged':sha(CATALOG)==catalog_sha,
            'all_pass':all(c['pass'] for c in cases+table_results+conversion_checks)}
    with (out/'results.json').open('x',encoding='utf-8') as f:json.dump(report,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps({'results':str(out/'results.json'),'coverage':report['coverage'],'all_pass':report['all_pass'],
                      'families':report['families']},ensure_ascii=False,indent=2))
    return 0 if report['all_pass'] else 1


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir',type=Path,required=True)
    parser.add_argument('--resource-dir',type=Path,default=CUBES,help='CUBE directory; defaults to filters/looks')
    parser.add_argument('--source-half-dir',type=Path,help='Optional external RGBA16F .data directory; omitted by default')
    parser.add_argument('--photo-fixtures',type=Path,help='Optional external JSON manifest with images/anchors; see load_photo_fixtures schema')
    args=parser.parse_args()
    return run_suite(args.out_dir, args.resource_dir, args.source_half_dir, args.photo_fixtures)


if __name__=='__main__':
    raise SystemExit(main())
