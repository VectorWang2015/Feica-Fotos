#!/usr/bin/env python3
"""Explicit, bounded classic-TIFF/DNG JPEG-preview container adapter.

No CFA development, automatic preview selection, ICC conversion, or resizing.
This is separate from the recovered RGB Look stage and makes no claim about
FOTOS's unknown image-source API or automatic preview-selection policy.
"""
from dataclasses import dataclass, asdict
from pathlib import Path
import argparse
import hashlib
import io
import json
import os
import stat
import struct
import sys
import warnings

import numpy as np
from PIL import Image, ImageFile, PngImagePlugin

if __package__:
    from . import renderer
else:  # Direct script execution, like renderer.py.
    import renderer


class PreviewError(ValueError):
    """Malformed container, unsupported preview, or unsafe operation."""


@dataclass(frozen=True)
class Limits:
    max_file_bytes: int = 512 * 1024 * 1024
    max_ifds: int = 64
    max_tags_per_ifd: int = 4096
    max_values_per_tag: int = 1_000_000
    max_tag_bytes: int = 16 * 1024 * 1024
    max_decoded_tag_values: int = 1_000_000
    max_jpeg_bytes: int = 128 * 1024 * 1024
    max_pixels: int = 64_000_000
    max_jpeg_markers: int = 1_000_000

    def __post_init__(self):
        for key, value in asdict(self).items():
            _integer(value, key, 1)
        if self.max_ifds > 256:
            raise PreviewError('max_ifds must be <=256 (bounded graph traversal)')


def _integer(value, name, minimum=0):
    # Reject bool, float, NaN, coercion and unsafe JSON-sized integer inputs.
    if type(value) is not int or not minimum <= value <= (1 << 53) - 1:
        raise PreviewError(f'{name} must be an exact integer in {minimum}..2**53-1')
    return value


# Classic TIFF field widths. IFD (13) is the 32-bit SubIFD pointer extension;
# BigTIFF LONG8/SLONG8/IFD8 (16..18) are intentionally unsupported.
TYPE_WIDTH = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 6: 1, 7: 1,
              8: 2, 9: 4, 10: 8, 11: 4, 12: 8, 13: 4}
# Relevant tags must have their specified TIFF metadata types, not merely
# numerically convenient bytes. Unrelated tags are still type/range bounded.
TYPES = {254: (4,), 256: (3, 4), 257: (3, 4), 258: (3,), 259: (3,),
         262: (3,), 273: (3, 4), 274: (3,), 277: (3,), 278: (3, 4),
         279: (3, 4), 284: (3,), 322: (3, 4), 323: (3, 4),
         324: (4,), 325: (3, 4), 330: (4, 13), 339: (3,), 34665: (4, 13),
         34675: (7,), 40961: (3,), 50706: (1,), 50970: (4,)}
SCALARS = {254, 256, 257, 259, 262, 274, 277, 278, 284,
           322, 323, 34665, 40961, 50970}
NAMES = {254: 'new_subfile_type', 256: 'width', 257: 'height',
         258: 'bits_per_sample', 259: 'compression', 262: 'photometric',
         273: 'strip_offsets', 274: 'orientation', 277: 'samples_per_pixel',
         278: 'rows_per_strip', 279: 'strip_byte_counts', 284: 'planar_configuration',
         322: 'tile_width', 323: 'tile_height', 324: 'tile_offsets',
         325: 'tile_byte_counts', 330: 'sub_ifds', 339: 'sample_format',
         50970: 'preview_color_space'}


def _read_source(path, limits):
    path = Path(path)
    # Nonblocking open prevents an accidental FIFO/device input from hanging
    # before the regular-file check; regular files still use normal reads.
    fd = os.open(path, os.O_RDONLY | getattr(os, 'O_NONBLOCK', 0))
    with os.fdopen(fd, 'rb') as handle:
        before = os.fstat(handle.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise PreviewError('Input must be a regular file')
        if before.st_size < 8 or before.st_size > limits.max_file_bytes:
            raise PreviewError('Input size outside configured 8-byte..max_file_bytes bound')
        raw = handle.read(limits.max_file_bytes + 1)
        after = os.fstat(handle.fileno())
    stamp = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    if len(raw) != before.st_size or stamp(before) != stamp(after):
        raise PreviewError('Input changed during read')
    return raw


class _TIFF:
    def __init__(self, raw, limits):
        self.raw, self.limits, self.directories = raw, limits, {}
        self.decoded_values = 0
        if raw[:2] not in (b'II', b'MM'):
            raise PreviewError('Expected classic TIFF II/MM byte order marker')
        self.endian = '<' if raw[:2] == b'II' else '>'
        if self.unpack('H', 2)[0] != 42:
            raise PreviewError('Only classic TIFF/DNG magic42 supported; not BigTIFF')
        self.first = self.unpack('I', 4)[0]
        if not self.first:
            raise PreviewError('Missing IFD0')
        self.image_ifds = []
        active, visited = set(), set()

        def visit(offset):
            if offset in active:
                raise PreviewError(f'Cycle in image IFD/SubIFD/next graph at {offset}')
            if offset in visited:
                return
            active.add(offset)
            directory = self.directory(offset)
            self.image_ifds.append(directory)
            for target in directory['tags'].get(330, []):
                if not target:
                    raise PreviewError('Null SubIFD pointer is unsupported')
                visit(target)
            if directory['next_ifd']:
                visit(directory['next_ifd'])
            active.remove(offset)
            visited.add(offset)
        visit(self.first)

    def bound(self, offset, size, label):
        _integer(offset, label + ' offset')
        _integer(size, label + ' size')
        if offset > len(self.raw) or size > len(self.raw) - offset:
            raise PreviewError(f'{label} outside file bounds')

    def unpack(self, fmt, offset):
        self.bound(offset, struct.calcsize(self.endian + fmt), 'TIFF field')
        return struct.unpack_from(self.endian + fmt, self.raw, offset)

    def directory(self, offset):
        if offset in self.directories:
            return self.directories[offset]
        if len(self.directories) >= self.limits.max_ifds:
            raise PreviewError('Exceeded max_ifds limit')
        if offset < 8:
            raise PreviewError('IFD offset overlaps TIFF header')
        count = self.unpack('H', offset)[0]
        if count > self.limits.max_tags_per_ifd:
            raise PreviewError('Exceeded max_tags_per_ifd limit')
        self.bound(offset, 2 + count * 12 + 4, 'IFD table')
        tags, fields = {}, {}
        for index in range(count):
            start = offset + 2 + 12 * index
            tag, dtype, n = self.unpack('HHI', start)
            if tag in fields:
                raise PreviewError(f'Duplicate tag {tag} in IFD {offset}')
            if dtype not in TYPE_WIDTH:
                raise PreviewError(f'Invalid/unsupported TIFF dtype {dtype} for tag {tag}')
            if n < 1 or n > self.limits.max_values_per_tag:
                raise PreviewError(f'Tag {tag} count outside configured bounds')
            size = n * TYPE_WIDTH[dtype]
            if size > self.limits.max_tag_bytes:
                raise PreviewError(f'Tag {tag} exceeds max_tag_bytes')
            pos = start + 8 if size <= 4 else self.unpack('I', start + 8)[0]
            if size > 4 and pos < 8:
                raise PreviewError(f'Tag {tag} payload overlaps TIFF header')
            self.bound(pos, size, f'Tag {tag} payload')
            fields[tag] = {'dtype': dtype, 'count': n, 'offset': pos, 'length': size}
            if tag not in TYPES:
                continue
            if dtype not in TYPES[tag]:
                raise PreviewError(f'Wrong metadata dtype {dtype} for tag {tag}; expected {TYPES[tag]}')
            if tag in SCALARS and n != 1:
                raise PreviewError(f'Tag {tag} must have scalar count1')
            if tag == 50706 and n != 4:
                raise PreviewError('DNGVersion must have BYTE count4')
            if tag == 330 and n > self.limits.max_ifds:
                raise PreviewError('SubIFD pointer count exceeds max_ifds')
            if tag == 34675:  # ICC payload is handled without TIFF numeric decoding.
                continue
            if n > self.limits.max_decoded_tag_values - self.decoded_values:
                raise PreviewError('Exceeded aggregate max_decoded_tag_values budget')
            self.decoded_values += n
            fmt = {1: 'B', 3: 'H', 4: 'I', 13: 'I'}[dtype]
            tags[tag] = list(self.unpack(str(n) + fmt, pos))
        directory = {'ifd_offset': offset, 'tags': tags, 'fields': fields,
                     'next_ifd': self.unpack('I', offset + 2 + count * 12)[0]}
        self.directories[offset] = directory
        return directory

    def jpeg_range(self, directory):
        t = directory['tags']
        if t.get(262) == [32803]:
            raise PreviewError('CFA IFD explicitly forbidden; no RAW decode')
        if t.get(254) != [1]:
            raise PreviewError('Require explicit NewSubfileType=1 reduced image only')
        if t.get(262) != [6] or t.get(259) != [7]:
            raise PreviewError('Require PhotometricInterpretation=6 YCbCr and Compression=7 JPEG')
        if any(tag in directory['fields'] for tag in (322, 323, 324, 325)):
            raise PreviewError('Tiled previews unsupported; single-strip only')
        if t.get(277) != [3] or t.get(258) != [8, 8, 8]:
            raise PreviewError('Require SamplesPerPixel=3 and BitsPerSample=8,8,8 (not16-bit)')
        if t.get(284, [1]) != [1]:
            raise PreviewError('Separate planar samples unsupported')
        if t.get(339, [1, 1, 1]) != [1, 1, 1]:
            raise PreviewError('Require unsigned integer SampleFormat=1,1,1 or absent')
        if len(t.get(273, [])) != 1 or len(t.get(279, [])) != 1:
            raise PreviewError('Multistrip/missing strip unsupported; require one offset/count')
        width, height = t.get(256, [0])[0], t.get(257, [0])[0]
        if width < 1 or height < 1 or width * height > self.limits.max_pixels:
            raise PreviewError('IFD dimensions outside max_pixels bound')
        if t.get(278, [0xffffffff])[0] < height:
            raise PreviewError('RowsPerStrip smaller than image height')
        start, size = t[273][0], t[279][0]
        if start < 8 or size < 4 or size > self.limits.max_jpeg_bytes:
            raise PreviewError('JPEG range outside configured bounds')
        self.bound(start, size, 'JPEG strip')
        return start, size, width, height


def _jpeg_frame(data, limits):
    """Walk the selected strip's marker grammar, never search DNG for FFD8.

    Bound segment lengths, require exactly one 8-bit/3-component Huffman SOF,
    >=1 scan and final EOI with no trailing payload. Entropy byte stuffing and
    restart markers are handled so an interior EOI is not mistaken for an end.
    Pillow subsequently performs the actual JPEG entropy decode. ICC APP2
    ranges/hashes come from the whole marker walk, including after SOS, not
    from Pillow's header-only info. Profile payload contents remain opaque.
    """
    if not data.startswith(b'\xff\xd8') or not data.endswith(b'\xff\xd9'):
        raise PreviewError('Strip must be a complete standalone JPEG SOI/EOI stream')
    pos, frame, scans, markers = 2, None, 0, 0
    icc_parts, icc_count = {}, None
    sof = {0xc0, 0xc1, 0xc2, 0xc3, 0xc5, 0xc6, 0xc7,
           0xc9, 0xca, 0xcb, 0xcd, 0xce, 0xcf}
    while pos < len(data):
        markers += 1
        if markers > limits.max_jpeg_markers:
            raise PreviewError('Exceeded max_jpeg_markers')
        if data[pos] != 0xff:
            raise PreviewError('Malformed JPEG marker boundary')
        while pos < len(data) and data[pos] == 0xff:
            pos += 1
        if pos == len(data):
            raise PreviewError('Truncated JPEG marker')
        code = data[pos]; pos += 1
        if code == 0xd9:
            if pos != len(data) or frame is None or not scans:
                raise PreviewError('Premature/trailing JPEG EOI')
            if icc_count is not None and set(icc_parts) != set(range(1, icc_count + 1)):
                raise PreviewError('Incomplete JPEG ICC APP2 sequence; cannot silently discard tags')
            chunks = [icc_parts[sequence] for sequence in sorted(icc_parts)]
            digest = hashlib.sha256()
            for chunk in chunks:
                start = chunk['payload_offset']
                digest.update(data[start:start + chunk['payload_length']])
            frame['icc_app2'] = {
                'present': bool(chunks), 'chunk_count': len(chunks),
                'payload_bytes': sum(chunk['payload_length'] for chunk in chunks),
                'icc_sha256': digest.hexdigest() if chunks else None,
                'chunks': chunks, 'range_domain': 'selected JPEG-relative bytes',
                'payload_interpretation': 'opaque; no ICC profile validation/conversion',
            }
            return frame
        if code in (0x00, 0x01, 0xd8) or 0xd0 <= code <= 0xd7:
            raise PreviewError('Unexpected standalone JPEG marker')
        if pos + 2 > len(data):
            raise PreviewError('Truncated JPEG segment length')
        size = int.from_bytes(data[pos:pos + 2], 'big')
        if size < 2 or size > len(data) - pos:
            raise PreviewError('JPEG segment outside strip bounds')
        if code == 0xe2 and data[pos + 2:pos + 14] == b'ICC_PROFILE\x00':
            if size <= 16:
                raise PreviewError('Truncated/empty JPEG ICC APP2 payload')
            sequence, total = data[pos + 14], data[pos + 15]
            if not 1 <= sequence <= total or sequence in icc_parts or icc_count not in (None, total):
                raise PreviewError('Invalid JPEG ICC APP2 sequence; cannot silently discard tags')
            payload_start, payload_length = pos + 16, size - 16
            icc_parts[sequence] = {
                'sequence': sequence, 'declared_total': total,
                'marker_offset': pos - 2, 'payload_offset': payload_start,
                'payload_length': payload_length,
                'payload_sha256': hashlib.sha256(data[payload_start:pos + size]).hexdigest(),
                'after_sos': scans > 0,
            }
            icc_count = total
        if code in sof:
            if frame is not None or size < 8:
                raise PreviewError('Duplicate/truncated JPEG SOF')
            precision = data[pos + 2]
            height = int.from_bytes(data[pos + 3:pos + 5], 'big')
            width = int.from_bytes(data[pos + 5:pos + 7], 'big')
            components = data[pos + 7]
            if precision != 8 or components != 3 or code not in (0xc0, 0xc2):
                raise PreviewError('Only actualJPEG8, 3-component baseline/progressive Huffman JPEG supported')
            if size != 8 + 3 * components or not width or not height or width * height > limits.max_pixels:
                raise PreviewError('Invalid JPEG SOF dimensions/component count')
            frame = {'width': width, 'height': height, 'precision_bits': precision,
                     'components': components, 'sof_marker': hex(code)}
        pos += size
        if code == 0xda:
            if frame is None:
                raise PreviewError('JPEG SOS before SOF')
            scans += 1
            # Only search for marker escapes inside THIS explicitly bounded strip.
            while True:
                marker = data.find(b'\xff', pos)
                if marker < 0:
                    raise PreviewError('JPEG entropy scan has no terminating marker')
                cursor = marker + 1
                while cursor < len(data) and data[cursor] == 0xff:
                    cursor += 1
                if cursor == len(data):
                    raise PreviewError('Truncated JPEG entropy marker')
                escaped = data[cursor]
                if escaped == 0 or 0xd0 <= escaped <= 0xd7:
                    pos = cursor + 1
                    continue
                pos = marker
                break
    raise PreviewError('Missing JPEG EOI')


ORIENTATION = {1: (None, 'identity'),
               2: (Image.Transpose.FLIP_LEFT_RIGHT, 'FLIP_LEFT_RIGHT'),
               3: (Image.Transpose.ROTATE_180, 'ROTATE_180'),
               4: (Image.Transpose.FLIP_TOP_BOTTOM, 'FLIP_TOP_BOTTOM'),
               5: (Image.Transpose.TRANSPOSE, 'TRANSPOSE'),
               6: (Image.Transpose.ROTATE_270, 'ROTATE_270'),
               7: (Image.Transpose.TRANSVERSE, 'TRANSVERSE'),
               8: (Image.Transpose.ROTATE_90, 'ROTATE_90')}


def _orientation(tiff, selected):
    root = tiff.directories[tiff.first]
    if 274 in selected['tags']:
        value, source = selected['tags'][274][0], 'selected_ifd'
        offset = selected['ifd_offset']
    elif 274 in root['tags']:
        value, source, offset = root['tags'][274][0], 'ifd0_inherited', tiff.first
    else:
        value, source, offset = 1, 'default_absent', None
    if value not in ORIENTATION:
        raise PreviewError('Orientation must be1..8; present invalid value is not absent')
    return {'value': value, 'source': source, 'source_ifd_offset': offset,
            'action': ORIENTATION[value][1], 'method': 'PIL.Image.transpose exact pixel permutation',
            'output_orientation': 1, 'resized': False, 'interpolation': 'none'}


def _source_audit(path, raw, tiff):
    return {'schema': 'dng-preview-container-adapter-v1', 'input': str(Path(path).resolve()),
            'input_sha256': hashlib.sha256(raw).hexdigest(), 'input_bytes': len(raw),
            'container': 'classic TIFF/DNG', 'byte_order': 'II' if tiff.endian == '<' else 'MM',
            'ifd0_offset': tiff.first, 'limits': asdict(tiff.limits),
            'noRAWdecode': True, 'raw_developed': False, 'resized': False,
            'selection_policy': 'explicit caller selection only; no inferred FOTOS automatic policy'}


def inspect_dng(path, *, limits=None):
    """Read-only image-IFD inventory; unsupported images have rejection reasons.

    Structural TIFF errors reject the whole file. Eligibility checks include
    selected-strip JPEG marker/SOF validation, not full JPEG entropy decoding.
    """
    limits = limits or Limits()
    raw = _read_source(path, limits)
    tiff = _TIFF(raw, limits)
    result = _source_audit(path, raw, tiff)
    result['directories'] = []
    for directory in tiff.image_ifds:
        t = directory['tags']
        item = {'ifd_offset': directory['ifd_offset'], 'next_ifd': directory['next_ifd'],
                **{name: t.get(tag) for tag, name in NAMES.items()},
                'eligible': False}
        try:
            start, size, width, height = tiff.jpeg_range(directory)
            jpeg = raw[start:start + size]
            frame = _jpeg_frame(jpeg, limits)
            if (frame['width'], frame['height']) != (width, height):
                raise PreviewError('IFD/JPEG dimensions disagree')
            item.update(orientation_audit=_orientation(tiff, directory), eligible=True,
                        jpeg_range={'offset': start, 'length': size, 'end_exclusive': start + size},
                        jpeg_sha256=hashlib.sha256(jpeg).hexdigest(), jpeg_frame=frame)
        except PreviewError as error:
            item['rejection_reason'] = str(error)
        result['directories'].append(item)
    result['inspection_scope'] = 'IFD and selected-strip structure only; no full JPEG entropy/color decode'
    return result


def _color_tags(tiff, selected, image, frame, jpeg):
    # Every visible ICC source is reported independently, never silently chosen
    # over another ICC. DNG RAW calibration/AsShotICC tags are not preview ICC.
    # Pillow stops its metadata walk at SOS; our full marker walk is authoritative
    # for JPEG ICC even if complete APP2 chunks occur between scans or before EOI.
    records, conflicts = [], []
    marker_icc = frame['icc_app2']
    info = dict(image.info)
    pillow_icc = info.get('icc_profile')
    pillow_hash = hashlib.sha256(pillow_icc).hexdigest() if pillow_icc else None
    if marker_icc['present']:
        info['icc_profile'] = b''.join(
            jpeg[chunk['payload_offset']:chunk['payload_offset'] + chunk['payload_length']]
            for chunk in marker_icc['chunks'])
    try:
        exif = image.getexif()
        jpeg_exif = exif.get_ifd(34665) if 34665 in exif else {}
        tags, issues = renderer.inspect_input_color_tags(info, jpeg_exif.get(40961))
        jpeg_orientation = exif.get(274)
    except (TypeError, ValueError, KeyError, SyntaxError, struct.error, UserWarning) as error:
        raise PreviewError(f'Malformed JPEG color/EXIF metadata: {error}') from error
    records.append({'source': 'selected_jpeg', **tags,
                    'icc_detection': 'full JPEG marker walk; Pillow info also recorded',
                    'pillow_icc_present': bool(pillow_icc), 'pillow_icc_sha256': pillow_hash,
                    'pillow_icc_matches_marker_walk': (
                        pillow_hash == marker_icc['icc_sha256'] if pillow_icc else None)})
    conflicts.extend('selected_jpeg: ' + issue for issue in issues)
    offsets = list(dict.fromkeys([selected['ifd_offset'], tiff.first]))
    for offset in offsets:
        directory = tiff.directories[offset]
        source = 'selected_ifd' if offset == selected['ifd_offset'] else 'ifd0'
        fields, values = directory['fields'], directory['tags']
        icc = None
        if 34675 in fields:
            field = fields[34675]
            icc = tiff.raw[field['offset']:field['offset'] + field['length']]
        color_space = values.get(40961, [None])[0]
        if 34665 in values:
            pointer = values[34665][0]
            if pointer in {d['ifd_offset'] for d in tiff.image_ifds}:
                raise PreviewError('EXIF pointer aliases an image IFD')
            exif_ifd = tiff.directory(pointer)
            color_space = exif_ifd['tags'].get(40961, [color_space])[0]
        tags, issues = renderer.inspect_input_color_tags({'icc_profile': icc}, color_space)
        preview_space = values.get(50970, [None])[0]
        records.append({'source': source, 'ifd_offset': offset, **tags,
                        'dng_preview_color_space': preview_space})
        conflicts.extend(source + ': ' + issue for issue in issues)
        if preview_space not in (None, 2):
            conflicts.append(f'{source}: DNG PreviewColorSpace={preview_space} is not explicitly sRGB(2)')
    return records, conflicts, jpeg_orientation


def extract_preview_rgb8(path, ifd_offset=None, *, preview_size=None, input_domain,
                         acknowledge_unconverted_color_tags=False, limits=None):
    """Return (upright HxWx3 uint8 RGB, source audit), never render/develop RAW.

    Supply exactly one reachable image-IFD offset or exact stored (width,height).
    Duplicate size matches are rejected. input_domain='encoded-srgb' is an
    explicit numeric assertion, not a color conversion or inferred file tag.
    Selected IFD orientation wins; absent inherits IFD0, then defaults to1.
    """
    if input_domain != 'encoded-srgb':
        raise PreviewError('Explicit input_domain="encoded-srgb" required; no automatic color conversion')
    if type(acknowledge_unconverted_color_tags) is not bool:
        raise PreviewError('Color-tag acknowledgement must be bool')
    if (ifd_offset is None) == (preview_size is None):
        raise PreviewError('Supply exactly one explicit ifd_offset or preview_size; no automatic selection')
    if ifd_offset is not None:
        _integer(ifd_offset, 'ifd_offset', 8)
    else:
        if not isinstance(preview_size, (tuple, list)) or len(preview_size) != 2:
            raise PreviewError('preview_size must be exact (width,height) in stored pixels')
        preview_size = tuple(_integer(v, 'preview_size dimension', 1) for v in preview_size)
    limits = limits or Limits()
    raw = _read_source(path, limits)
    tiff = _TIFF(raw, limits)
    if ifd_offset is not None:
        matches = [d for d in tiff.image_ifds if d['ifd_offset'] == ifd_offset]
    else:
        matches = [d for d in tiff.image_ifds if
                   (d['tags'].get(256, [None])[0], d['tags'].get(257, [None])[0]) == preview_size
                   and d['tags'].get(254) == [1]]
    if not matches:
        raise PreviewError('Explicit image IFD/preview size not found in reachable image graph')
    if len(matches) != 1:
        raise PreviewError('Ambiguous preview_size; specify exact ifd_offset instead')
    selected = matches[0]
    start, size, width, height = tiff.jpeg_range(selected)
    jpeg = raw[start:start + size]
    frame = _jpeg_frame(jpeg, limits)
    if (frame['width'], frame['height']) != (width, height):
        raise PreviewError('IFD/JPEG dimensions disagree')
    orientation = _orientation(tiff, selected)
    if ImageFile.LOAD_TRUNCATED_IMAGES:
        raise PreviewError('Pillow truncated-image decoding must remain disabled')
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', Image.DecompressionBombWarning)
            warnings.simplefilter('error', UserWarning)
            with Image.open(io.BytesIO(jpeg)) as image:
                if image.format != 'JPEG' or image.mode != 'RGB' or image.size != (width, height):
                    raise PreviewError('Require decoded JPEG RGB8 with exactly matching IFD dimensions')
                tags, conflicts, jpeg_orientation = _color_tags(tiff, selected, image, frame, jpeg)
                if conflicts and not acknowledge_unconverted_color_tags:
                    raise PreviewError('; '.join(conflicts) + '; explicit acknowledge_unconverted_color_tags required')
                image.load()
                transpose = ORIENTATION[orientation['value']][0]
                upright = image.transpose(transpose) if transpose is not None else image
                rgb = np.array(upright, dtype=np.uint8, copy=True)
    except (OSError, SyntaxError, UserWarning, Image.DecompressionBombError,
            Image.DecompressionBombWarning) as error:
        raise PreviewError(f'JPEG decode rejected: {error}') from error
    audit = _source_audit(path, raw, tiff)
    audit.update({'ifd_offset': selected['ifd_offset'],
                  'selection': {'ifd_offset': ifd_offset, 'preview_size': list(preview_size) if preview_size else None},
                  'jpeg_range': {'offset': start, 'length': size, 'end_exclusive': start + size},
                  'jpeg_sha256': hashlib.sha256(jpeg).hexdigest(), 'jpeg_frame': frame,
                  'stored_size': [width, height], 'upright_size': [rgb.shape[1], rgb.shape[0]],
                  'source_ifd': {name: selected['tags'].get(tag) for tag, name in NAMES.items()},
                  'orientation': {**orientation, 'jpeg_exif_orientation_ignored': jpeg_orientation},
                  'input_domain': input_domain, 'input_color_tags': tags,
                  'unconverted_tag_warnings': conflicts,
                  'tag_override_acknowledged': acknowledge_unconverted_color_tags,
                  'color_conversion': 'none; JPEG YCbCr-to-RGB decode only; no ICC/gamma/primary transform',
                  'decoder': 'Pillow JPEG RGB8', 'rgb8_sha256': hashlib.sha256(rgb.tobytes()).hexdigest(),
                  'not_included': ['CFA/RAW development', 'automatic official preview selection',
                                   'ICC conversion', 'resize/resample', 'FOTOS save/Apple JPEG encoding']})
    return rgb, audit


def _validate_outputs(source, output):
    source, output = Path(source), Path(output)
    sidecar = Path(str(output) + '.json')
    if output.suffix.lower() != '.png':
        raise PreviewError('Output must be a new RGB8 .png plus .png.json')
    for path in (output, sidecar):
        if path.resolve() == source.resolve():
            raise PreviewError('Output must not equal/alias input')
        # lexists rejects dangling symlinks too; existing hardlinks are never opened.
        if os.path.lexists(path):
            raise PreviewError(f'Never overwrite existing output or symlink: {path}')
    return output, sidecar


def _write_outputs(source, output, rgb, manifest):
    output, sidecar = _validate_outputs(source, output)
    output.parent.mkdir(parents=True, exist_ok=True)
    created = []
    try:
        # Exclusively reserve BOTH names before writing; O_EXCL also rejects symlinks.
        for path in (output, sidecar):
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, 'O_NOFOLLOW', 0), 0o600)
            created.append((path, fd, os.fstat(fd)))
        metadata = PngImagePlugin.PngInfo()
        metadata.add(b'sRGB', bytes([0]))
        metadata.add_text('Description', 'Explicit DNG JPEG-preview adapter + local RGB Look reference; not RAW development')
        with os.fdopen(os.dup(created[0][1]), 'wb') as handle:
            Image.fromarray(rgb).save(handle, format='PNG', pnginfo=metadata)
        digest = hashlib.sha256()
        with output.open('rb') as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b''):
                digest.update(block)
        manifest.update(output=str(output.resolve()), output_sha256=digest.hexdigest())
        with os.fdopen(os.dup(created[1][1]), 'w', encoding='utf-8') as handle:
            json.dump(manifest, handle, indent=2, allow_nan=False)
            handle.write('\n')
    except BaseException:
        # Remove only newly reserved files that still have our original inode.
        for path, _, original in created:
            try:
                current = path.lstat()
                if (current.st_dev, current.st_ino) == (original.st_dev, original.st_ino):
                    path.unlink()
            except FileNotFoundError:
                pass
        raise
    finally:
        for _, fd, _ in created:
            os.close(fd)
    return output, sidecar


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == 'inspect':
        parser = argparse.ArgumentParser(description='Read-only DNG image-IFD inventory; no automatic selection')
        parser.add_argument('input', type=Path)
        args = parser.parse_args(argv[1:])
        try:
            print(json.dumps(inspect_dng(args.input), indent=2, allow_nan=False))
        except (PreviewError, OSError) as error:
            parser.error(str(error))
        return
    if argv and argv[0] == 'render':
        argv = argv[1:]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--ifd-offset', type=lambda text: int(text, 0), required=True,
                        help='Explicit reachable image IFD offset; decimal or0x; never auto-largest')
    parser.add_argument('--look', choices=['steve', 'eternal', 'vivid'], required=True)
    parser.add_argument('--strength', type=float, default=100)
    parser.add_argument('--input-domain', choices=['encoded-srgb'], required=True)
    parser.add_argument('--resource-dir', type=Path, default=renderer.RESOURCE_DIR)
    parser.add_argument('--acknowledge-unconverted-color-tags', action='store_true')
    args = parser.parse_args(argv)
    try:
        _validate_outputs(args.input, args.output)
        if not np.isfinite(args.strength) or not 0 <= args.strength <= 100:
            raise PreviewError('Strength must be finite within0..100')
        if args.look in ('eternal', 'vivid') and args.strength != 100:
            raise PreviewError(f'Only {args.look.title()}100 is supported')
        rgb, audit = extract_preview_rgb8(args.input, args.ifd_offset,
                                         input_domain=args.input_domain,
                                         acknowledge_unconverted_color_tags=args.acknowledge_unconverted_color_tags)
        rendered = renderer.render_rgb8(rgb, args.look, args.strength, args.resource_dir)
        names = {'steve': [renderer.STEVE1, renderer.STEVE3],
                 'eternal': [renderer.ETERNAL], 'vivid': [renderer.VIVID]}[args.look]
        manifest = {'source': audit, 'look': args.look, 'strength': args.strength,
                    'input_domain': args.input_domain,
                    'output_domain': 'encoded-sRGB RGB8 PNG with sRGB chunk',
                    'renderer': 'renderer.render_rgb8',
                    'table_values': ('CUBE decimal samples restored via float16 then float64; frozen Vivid full-weight1 contract'
                                     if args.look == 'vivid' else 'CUBE decimal samples retained as float64'),
                    'validation_scope': ('Vivid100 frozen full-weight binary16 model corroborated on new0333 user-labeled scene without fitting; other Vivid strengths unsupported; UI binding not CPU-traced'
                                         if args.look == 'vivid' else 'Existing Steve/Eternal RGB reference validation; no image-source selection binding inferred'),
                    'resource_hashes': {name: hashlib.sha256((args.resource_dir / name).read_bytes()).hexdigest()
                                        for name in names},
                    'scope': 'container adapter + existing RGB stage; no IR-derived image-source selection claim'}
        output, sidecar = _write_outputs(args.input, args.output, rendered, manifest)
        print(json.dumps({'output': str(output), 'manifest': str(sidecar),
                          'width': rendered.shape[1], 'height': rendered.shape[0]}, indent=2))
    except (ValueError, OSError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()
