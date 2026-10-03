#!/usr/bin/env python3
"""Migrate prepared ICC branding while preserving every color payload byte.

The supported migration changes filenames and the v2 description tag. Originals
and a machine-readable audit are stored in a caller-selected ignored work folder.
No native camera input, LUT resampling, or external application is required.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import struct
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments.c1 import c1_single_profile as c1

LEGACY_MACHINE = 'LocalLooks'
MACHINE_BRAND = 'FeicaFotos'
DISPLAY_BRAND = 'Feica Fotos'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def display_name(payload):
    """Read the terminated ASCII portion of an ICC v2 descType."""
    if len(payload) < 13 or payload[:4] != b'desc':
        raise ValueError('ICC v2 descType required')
    count = struct.unpack_from('>I', payload, 8)[0]
    if count < 1 or 12+count > len(payload) or payload[11+count] != 0:
        raise ValueError('Invalid description ASCII bounds')
    return payload[12:11+count].decode('ascii')


def renamed_description(text):
    if text.startswith(LEGACY_MACHINE+'-'):
        return DISPLAY_BRAND+text[len(LEGACY_MACHINE):]
    if text.startswith(DISPLAY_BRAND+'-'):
        return text
    raise ValueError('Description is outside the supported product family')


def renamed_path(value):
    p = PurePosixPath(value)
    if p.is_absolute() or '..' in p.parts or '\\' in value:
        raise ValueError('A safe relative profile path is required')
    prefix = 'LeicaQTyp116-'+LEGACY_MACHINE+'-'
    if not p.name.startswith(prefix) or p.suffix != '.icm':
        raise ValueError('Filename is outside the supported product family')
    return str(p.with_name(p.name.replace(LEGACY_MACHINE, MACHINE_BRAND, 1)))


def replace_description(data, text):
    """Rewrite directory offsets and length; no other tag/header data changes."""
    tags = c1.read_tags(data)
    if b'desc' not in dict(tags):
        raise ValueError('Description tag missing')
    replacement = c1.description(text)
    result = bytearray(data[:128]+struct.pack('>I', len(tags))+bytes(12*len(tags)))
    for index, (signature, payload) in enumerate(tags):
        new = replacement if signature == b'desc' else payload
        result.extend(bytes((-len(result)) % 4))
        struct.pack_into('>4sII', result, 132+12*index, signature, len(result), len(new))
        result.extend(new)
    struct.pack_into('>I', result, 0, len(result))
    rewritten = bytes(result)
    checks = unchanged_payloads(data, rewritten)
    if not all(checks.values()):
        raise AssertionError('Profile data changed outside the description')
    if display_name(dict(c1.read_tags(rewritten))[b'desc']) != text:
        raise AssertionError('Rewritten description differs')
    return rewritten


def unchanged_payloads(before, after):
    a, b = c1.read_tags(before), c1.read_tags(after)
    return {'header_except_length': before[4:128] == after[4:128],
            'tag_order': [s for s, _ in a] == [s for s, _ in b],
            'all_non_description_payloads': all(x == y for (s, x), (t, y) in zip(a, b) if s != b'desc') and len(a) == len(b),
            'A2B0_exact': dict(a).get(b'A2B0') == dict(b).get(b'A2B0')}


def _json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2)+'\n').encode('utf-8')


def migrate(root, audit_dir, *, apply=False):
    root, audit_dir = Path(root).resolve(), Path(audit_dir).resolve()
    top_path = root/'filters/MANIFEST.json'
    all_dir = root/'filters/c1/all-looks'
    all_path = all_dir/'MANIFEST.json'
    top = json.loads(top_path.read_text(encoding='utf-8'))
    all_manifest = json.loads(all_path.read_text(encoding='utf-8'))
    if len(top['camera_profiles']) != 3 or all_manifest['profile_count'] != 206 or len(all_manifest['profiles']) != 206:
        raise ValueError('Expected the current 3 plus 206 prepared profile inventory')
    new_top, new_all = copy.deepcopy(top), copy.deepcopy(all_manifest)
    inventory = []
    for old_row, new_row in zip(top['camera_profiles'], new_top['camera_profiles']):
        inventory.append((old_row, new_row, 'path', ''))
    for old_row, new_row in zip(all_manifest['profiles'], new_all['profiles']):
        inventory.append((old_row, new_row, 'relative_path', 'filters/c1/all-looks/'))
    planned, seen = [], set()
    for old, new, key, prefix in inventory:
        old_path = prefix+old[key]
        new_value = renamed_path(old[key]); new_path = prefix+new_value
        data = (root/old_path).read_bytes()
        if len(data) != old['bytes'] or sha(data) != old['sha256']:
            raise ValueError('Prepared profile hash/size mismatch: '+old_path)
        if (root/new_path).exists() or new_path.casefold() in seen:
            raise FileExistsError('Migration target exists or collides: '+new_path)
        seen.add(new_path.casefold())
        text = display_name(dict(c1.read_tags(data))[b'desc'])
        description = renamed_description(text)
        output = replace_description(data, description)
        new[key] = new_value; new['bytes'] = len(output); new['sha256'] = sha(output)
        if 'description' in new:
            if old['description'] != text:
                raise ValueError('Manifest description differs from ICC')
            new['description'] = description
        planned.append({'old_path': old_path, 'new_path': new_path,
                        'old_description': text, 'new_description': description,
                        'old_sha256': sha(data), 'new_sha256': sha(output),
                        'old_bytes': len(data), 'new_bytes': len(output),
                        'preservation': unchanged_payloads(data, output),
                        '_before': data, '_after': output})
    metadata_paths = [top_path, all_path, all_dir/'FILTERS.csv', all_dir/'SHA256SUMS.txt', all_dir/'README.md']
    metadata_before = {p: p.read_bytes() for p in metadata_paths}
    csv_rows = list(csv.DictReader(io.StringIO(metadata_before[all_dir/'FILTERS.csv'].decode('utf-8-sig'))))
    by_old = {a['relative_path']: b for a, b in zip(all_manifest['profiles'], new_all['profiles'])}
    if len(csv_rows) != len(by_old):
        raise ValueError('CSV row inventory differs from manifest')
    for row in csv_rows:
        new = by_old[row['File']]
        row['ICC name'], row['File'] = new['description'], new['relative_path']
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]), lineterminator='\n')
    writer.writeheader(); writer.writerows(csv_rows)
    readme = metadata_before[all_dir/'README.md'].decode('utf-8').replace(LEGACY_MACHINE, DISPLAY_BRAND)
    readme = readme.replace('# Capture One · Leica Q Typ116 全套滤镜', '# Feica Fotos · Leica Q Typ116 全套滤镜')
    metadata_after = {top_path: _json_bytes(new_top), all_path: _json_bytes(new_all),
                      all_dir/'FILTERS.csv': ('\ufeff'+stream.getvalue()).encode('utf-8'),
                      all_dir/'SHA256SUMS.txt': ''.join(f"{r['sha256']}  {r['relative_path']}\n" for r in new_all['profiles']).encode('utf-8'),
                      all_dir/'README.md': readme.encode('utf-8')}
    validation_path = all_dir/'VALIDATION.json'
    validation_sha = sha(validation_path.read_bytes())
    report = {'profiles': len(planned), 'applied': apply, 'color_payloads_rerendered': False,
              'all_preservation_checks_pass': all(all(p['preservation'].values()) for p in planned),
              'validation_metrics_sha256_before': validation_sha,
              'mapping': [{k: v for k, v in r.items() if not k.startswith('_')} for r in planned]}
    if not apply:
        return report
    if not audit_dir.is_relative_to(root/'work') or audit_dir.exists():
        raise ValueError('Use a new audit directory under the repository ignored work/ folder')
    audit_dir.mkdir(parents=True)
    # The backup is complete before changing active profile names or manifests.
    for row in planned:
        p = audit_dir/'originals'/row['old_path']; p.parent.mkdir(parents=True, exist_ok=True)
        with p.open('xb') as f: f.write(row['_before'])
    for p, data in metadata_before.items():
        copy_path = audit_dir/'originals'/p.relative_to(root); copy_path.parent.mkdir(parents=True, exist_ok=True)
        with copy_path.open('xb') as f: f.write(data)
    with (audit_dir/'planned.json').open('xb') as f: f.write(_json_bytes(report))
    for row in planned:
        target = root/row['new_path']
        with target.open('xb') as f: f.write(row['_after'])
        if target.read_bytes() != row['_after']:
            raise AssertionError('New profile write mismatch')
    for p, data in metadata_after.items():
        if p.read_bytes() != metadata_before[p]:
            raise RuntimeError('Metadata changed concurrently: '+str(p))
        p.write_bytes(data)
    for row in planned:
        old = root/row['old_path']
        if old.read_bytes() != row['_before']:
            raise RuntimeError('Original profile changed concurrently')
        old.unlink()
    report['validation_metrics_sha256_after'] = sha(validation_path.read_bytes())
    if report['validation_metrics_sha256_after'] != validation_sha:
        raise AssertionError('Numerical validation metadata changed')
    with (audit_dir/'audit.json').open('xb') as f: f.write(_json_bytes(report))
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=ROOT)
    p.add_argument('--audit-dir', type=Path, required=True)
    p.add_argument('--apply', action='store_true')
    args = p.parse_args()
    report = migrate(args.root, args.audit_dir, apply=args.apply)
    print(json.dumps({k: v for k, v in report.items() if k != 'mapping'}, indent=2))


if __name__ == '__main__':
    main()
