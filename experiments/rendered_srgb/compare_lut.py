#!/usr/bin/env python3
"""Test a LUT against a supplied before/after image pair using host FFmpeg.

Diagnostic only: this performs numerical RGB LUT evaluation, NOT an ICC-managed
RAW or iOS rendering emulation. Does not resize or align images, overwrite input,
change gamma/gamut, or call a network service. Error metrics are RGB code-value
errors, not Delta E and not a percentage of visual similarity.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess


def run(argv, cwd=None):
    p = subprocess.run(argv, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, timeout=90)
    if p.returncode:
        raise RuntimeError(f'Command failed ({p.returncode}): {argv!r}\n{p.stderr.decode(errors="replace")}')
    return p.stdout


def probe(path):
    data = run(['ffprobe', '-v', 'error', '-show_entries',
                'stream=width,height,pix_fmt,color_space,color_transfer,color_primaries', '-of', 'json', str(path)])
    return json.loads(data)['streams'][0]


def rgb8(path):
    return run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-threads', '1', '-i', str(path),
                '-frames:v', '1', '-pix_fmt', 'rgb24', '-f', 'rawvideo', '-threads', '1', 'pipe:1'])


def metrics(actual, target):
    if len(actual) != len(target) or not actual:
        raise ValueError('Image sample counts differ')
    try:
        import numpy as np  # optional; host system Python already provides it
    except ImportError:
        histogram = Counter(abs(a - b) for a, b in zip(actual, target))
    else:
        left = np.frombuffer(actual, dtype=np.uint8)
        right = np.frombuffer(target, dtype=np.uint8)
        counts = np.zeros(256, dtype=np.int64)
        for start in range(0, len(left), 3_000_000):
            delta = np.abs(np.subtract(left[start:start + 3_000_000], right[start:start + 3_000_000], dtype=np.int16))
            counts += np.bincount(delta, minlength=256)
        histogram = {i: int(n) for i, n in enumerate(counts) if n}
    count = len(actual)
    mae = sum(d * n for d, n in histogram.items()) / count
    mse = sum(d * d * n for d, n in histogram.items()) / count
    return {'mae_rgb_8bit': mae, 'rmse_rgb_8bit': math.sqrt(mse),
            'psnr_db': 10 * math.log10(255 ** 2 / mse) if mse else None,
            'max_channel_error_8bit': max(histogram),
            'fraction_channel_samples_within_2': sum(n for d, n in histogram.items() if d <= 2) / count,
            'fraction_channel_samples_within_5': sum(n for d, n in histogram.items() if d <= 5) / count}


def fingerprint(path):
    return {'path': str(path), 'bytes': path.stat().st_size,
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def compare(before, after, lut, out):
    before, after, lut, out = [Path(p).resolve() for p in (before, after, lut, out)]
    if out.exists():
        raise FileExistsError(f'Refusing to reuse existing experiment directory: {out}')
    before_info, after_info = probe(before), probe(after)
    if (before_info['width'], before_info['height']) != (after_info['width'], after_info['height']):
        raise ValueError('Source and reference dimensions differ; no automatic resizing or alignment is performed')
    a, b = rgb8(before), rgb8(after)
    if len(a) != before_info['width'] * before_info['height'] * 3:
        raise ValueError('Unexpected decoded RGB buffer size')
    out.mkdir(parents=True)
    shutil.copyfile(lut, out / 'look.cube')
    result = {'before': fingerprint(before), 'reference_after': fingerprint(after),
              'lut': fingerprint(lut), 'before_stream': before_info, 'reference_stream': after_info,
              'ffmpeg_version': run(['ffmpeg', '-version']).decode().splitlines()[0],
              'no_lut_baseline': metrics(a, b), 'renders': {},
              'limitations': ['No source ICC profile transform or gamma/gamut conversion performed',
                              'Decoded RGB numbers are used as supplied; this does not establish a LUT input colour space',
                              'Same dimensions are checked, but alignment and image correspondence need human verification',
                              'Reference compression, resampling or independent retouching can cause error',
                              'This is not a test of an iOS FOTOS installation or Capture One RAW rendering']}
    for interpolation in ('trilinear', 'tetrahedral'):
        filename = f'rendered-{interpolation}.png'
        vf = f'format=gbrpf32le,lut3d=file=look.cube:interp={interpolation}'
        command = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-threads', '1', '-i', str(before),
                   '-vf', vf, '-frames:v', '1', '-pix_fmt', 'rgb48be', '-threads', '1', '-n', filename]
        run(command, cwd=out)
        render = out / filename
        result['renders'][interpolation] = {'filter': vf, 'output': fingerprint(render),
                                           'metrics': metrics(rgb8(render), b)}
    (out / 'comparison.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('before', type=Path)
    ap.add_argument('after', type=Path)
    ap.add_argument('lut', type=Path)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    result = compare(args.before, args.after, args.lut, args.out)
    print(json.dumps({'baseline': result['no_lut_baseline'],
                      'renders': {key: value['metrics'] for key, value in result['renders'].items()}}, indent=2))


if __name__ == '__main__':
    main()
