#!/usr/bin/env python3
"""Synthetic TIFF/JPEG safety and adapter tests; no real DNG/vendor execution."""
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import struct
import tempfile
import unittest
from unittest import mock

import numpy as np
from PIL import Image, ImageCms

if __package__:
    from . import dng_preview as adapter
else:
    import dng_preview as adapter


def jpeg_fixture(*, width=5, height=3, icc=None, orientation=None, mode='RGB'):
    yy, xx = np.mgrid[:height, :width]
    rgb = np.stack([(xx * 43 + yy * 7) % 256, (yy * 71 + xx * 9) % 256,
                    (xx * 31 + yy * 47) % 256], axis=-1).astype(np.uint8)
    image = Image.fromarray(rgb).convert(mode)
    options = {'quality': 97, 'subsampling': 0}
    if icc is not None:
        options['icc_profile'] = icc
    if orientation is not None:
        exif = Image.Exif(); exif[274] = orientation
        options['exif'] = exif
    stream = io.BytesIO(); image.save(stream, 'JPEG', **options)
    return stream.getvalue()


def icc_segment(payload, sequence=1, total=1):
    chunk = b'ICC_PROFILE\x00' + bytes([sequence, total]) + payload
    return b'\xff\xe2' + struct.pack('>H', len(chunk) + 2) + chunk


def preview_tags(*, width=5, height=3, orientation=1, **unused):
    tags = {254: (4, [1]), 256: (4, [width]), 257: (4, [height]),
            258: (3, [8, 8, 8]), 259: (3, [7]), 262: (3, [6]),
            273: (4, [0]), 277: (3, [3]), 278: (4, [height]), 279: (4, [0])}
    if orientation is not None:
        tags[274] = (3, [orientation])
    return tags


def tiff_fixture(*, endian='<', directories=None, jpeg=None, first=8):
    """Assemble bounded synthetic classic TIFF; offsets remain explicit/testable.

    directories=[(offset, tag dict, next offset)]; every strip initially references
    the same complete synthetic JPEG. Tests can change fields afterwards.
    """
    if jpeg is None:
        jpeg = jpeg_fixture()
    if directories is None:
        directories = [(8, preview_tags(), 0)]
    end = max(offset + 2 + 12 * len(tags) + 4 for offset, tags, _ in directories)
    data = bytearray(end)
    data[:8] = (b'II' if endian == '<' else b'MM') + struct.pack(endian + 'HI', 42, first)
    positions = {}
    for offset, tags, next_ifd in directories:
        struct.pack_into(endian + 'H', data, offset, len(tags))
        for index, (tag, (dtype, values)) in enumerate(sorted(tags.items())):
            start = offset + 2 + 12 * index
            positions[(offset, tag)] = start
            if dtype in (1, 2, 7):
                payload = bytes(values)
                count = len(payload)
            else:
                fmt = {3: 'H', 4: 'I', 13: 'I', 8: 'h', 9: 'i'}[dtype]
                payload = struct.pack(endian + str(len(values)) + fmt, *values)
                count = len(values)
            struct.pack_into(endian + 'HHI', data, start, tag, dtype, count)
            if len(payload) <= 4:
                data[start + 8:start + 12] = payload.ljust(4, b'\x00')
            else:
                if len(data) % 2:
                    data.append(0)
                struct.pack_into(endian + 'I', data, start + 8, len(data))
                data.extend(payload)
        struct.pack_into(endian + 'I', data, offset + 2 + len(tags) * 12, next_ifd)
    if len(data) % 2:
        data.append(0)
    jpeg_start = len(data)
    data.extend(jpeg)
    for offset, tags, _ in directories:
        for tag, value in ((273, jpeg_start), (279, len(jpeg))):
            if tag in tags and tags[tag] == (4, [0]):
                struct.pack_into(endian + 'I', data, positions[(offset, tag)] + 8, value)
    return data, positions, jpeg_start


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.input = self.root / 'synthetic.dng'

    def tearDown(self):
        self.temp.cleanup()

    def put(self, data):
        self.input.write_bytes(data)
        return self.input

    def extract(self, data=None, offset=8, **kwargs):
        if data is not None:
            self.put(data)
        return adapter.extract_preview_rgb8(self.input, offset, input_domain='encoded-srgb', **kwargs)

    def rejected(self, data, pattern, *, offset=8, **kwargs):
        with self.assertRaisesRegex(adapter.PreviewError, pattern):
            self.extract(data, offset, **kwargs)

    def test_explicit_ifd1914_both_endian_subifd_and_next_inventory(self):
        for endian in ('<', '>'):
            with self.subTest(endian=endian):
                root = preview_tags(); root[262] = (3, [32803]); root[254] = (4, [0])
                root[330] = (13, [1914])
                data, _, _ = tiff_fixture(endian=endian, directories=[(8, root, 400),
                                            (400, preview_tags(width=4), 0), (1914, preview_tags(), 0)])
                rgb, audit = self.extract(data, 1914)
                self.assertEqual(rgb.shape, (3, 5, 3)); self.assertEqual(rgb.dtype, np.uint8)
                self.assertEqual(audit['ifd_offset'], 1914)
                self.assertEqual(audit['input_sha256'], hashlib.sha256(data).hexdigest())
                self.assertTrue(audit['noRAWdecode']); self.assertFalse(audit['resized'])
                self.assertEqual(self.input.read_bytes(), data)
                inventory = adapter.inspect_dng(self.input)
                self.assertEqual({x['ifd_offset'] for x in inventory['directories']}, {8, 400, 1914})
                self.assertFalse(inventory['directories'][0]['eligible'])

    def test_explicit_preview_size_and_ambiguity(self):
        data, _, _ = tiff_fixture(); self.put(data)
        rgb, audit = adapter.extract_preview_rgb8(self.input, preview_size=(5, 3), input_domain='encoded-srgb')
        self.assertEqual(rgb.shape, (3, 5, 3)); self.assertEqual(audit['selection']['preview_size'], [5, 3])
        data, _, _ = tiff_fixture(directories=[(8, preview_tags(), 400), (400, preview_tags(), 0)])
        self.put(data)
        with self.assertRaisesRegex(adapter.PreviewError, 'Ambiguous'):
            adapter.extract_preview_rgb8(self.input, preview_size=(5, 3), input_domain='encoded-srgb')

    def test_no_automatic_selection_and_not_found(self):
        data, _, _ = tiff_fixture(); self.put(data)
        for kwargs in ({}, {'ifd_offset': 8, 'preview_size': (5, 3)}):
            with self.assertRaisesRegex(adapter.PreviewError, 'exactly one'):
                adapter.extract_preview_rgb8(self.input, input_domain='encoded-srgb', **kwargs)
        self.rejected(data, 'not found', offset=1914)
        with self.assertRaisesRegex(adapter.PreviewError, 'not found'):
            adapter.extract_preview_rgb8(self.input, preview_size=(9, 9), input_domain='encoded-srgb')

    def test_exact_integer_and_domain_contract(self):
        data, _, _ = tiff_fixture(); self.put(data)
        for offset in (True, 8.0, -1, 0, 2**53, '8', float('nan')):
            with self.subTest(offset=offset):
                self.rejected(data, 'exact integer', offset=offset)
        with self.assertRaises(TypeError):
            adapter.extract_preview_rgb8(self.input, 8)
        with self.assertRaisesRegex(adapter.PreviewError, 'encoded-srgb'):
            adapter.extract_preview_rgb8(self.input, 8, input_domain='linear-srgb')
        with self.assertRaisesRegex(adapter.PreviewError, 'exact integer'):
            adapter.Limits(max_file_bytes=True)

    def test_forbid_cfa_even_when_jpeg_stream_exists(self):
        tags = preview_tags(); tags[262] = (3, [32803])
        data, _, _ = tiff_fixture(directories=[(8, tags, 0)])
        self.rejected(data, 'CFA IFD explicitly forbidden')
        inventory = adapter.inspect_dng(self.input)
        self.assertFalse(inventory['directories'][0]['eligible'])
        self.assertIn('CFA', inventory['directories'][0]['rejection_reason'])

    def test_require_reduced_ycbcr_jpeg7(self):
        for change in ({254: (4, [0])}, {254: None}, {254: (4, [5])},
                       {262: (3, [2])}, {259: (3, [1])}):
            tags = preview_tags()
            for key, value in change.items():
                if value is None:
                    del tags[key]
                else:
                    tags[key] = value
            data, _, _ = tiff_fixture(directories=[(8, tags, 0)])
            self.rejected(data, 'Require')

    def test_invalid_dtype_relevant_unknown_and_big_tiff(self):
        for tag, dtype in ((256, 0), (256, 16), (256, 9), (65000, 14)):
            tags = preview_tags(); tags[65000] = (3, [1])
            data, positions, _ = tiff_fixture(directories=[(8, tags, 0)])
            struct.pack_into('<H', data, positions[(8, tag)] + 2, dtype)
            self.rejected(data, 'dtype')
        data, _, _ = tiff_fixture(); struct.pack_into('<H', data, 2, 43)
        self.rejected(data, 'BigTIFF')

    def test_tag_count_payload_and_directory_bounds(self):
        data, positions, _ = tiff_fixture()
        samples = []
        modified = data.copy(); struct.pack_into('<I', modified, positions[(8, 258)] + 8, len(data) - 1)
        samples.append((modified, 'outside file bounds'))
        modified = data.copy(); struct.pack_into('<I', modified, positions[(8, 258)] + 4, 0xffffffff)
        samples.append((modified, 'count outside'))
        modified = data.copy(); struct.pack_into('<I', modified, positions[(8, 256)] + 4, 2)
        struct.pack_into('<I', modified, positions[(8, 256)] + 8, 100)
        samples.append((modified, 'scalar count1'))
        modified = data.copy(); struct.pack_into('<I', modified, 4, 0xfffffff0)
        samples.append((modified, 'outside file bounds'))
        modified = data.copy(); struct.pack_into('<H', modified, 8, 4097)
        samples.append((modified, 'max_tags_per_ifd'))
        samples.append((data[:20], 'IFD table outside'))
        for malformed, pattern in samples:
            with self.subTest(pattern=pattern):
                self.rejected(malformed, pattern)

    def test_duplicate_tag_null_ifd_and_header_overlap(self):
        data, positions, _ = tiff_fixture()
        duplicate = data.copy(); struct.pack_into('<H', duplicate, positions[(8, 256)], 254)
        self.rejected(duplicate, 'Duplicate tag')
        for offset, pattern in ((0, 'Missing IFD0'), (4, 'overlaps TIFF header')):
            modified = data.copy(); struct.pack_into('<I', modified, 4, offset)
            self.rejected(modified, pattern)

    def test_cycles_next_and_subifd_rejected(self):
        for mode in ('next', 'subifd'):
            tags = preview_tags()
            if mode == 'subifd':
                tags[330] = (4, [8])
            data, _, _ = tiff_fixture(directories=[(8, tags, 8 if mode == 'next' else 0)])
            self.rejected(data, 'Cycle')
        tags = preview_tags(); tags[330] = (4, [400])
        data, _, _ = tiff_fixture(directories=[(8, tags, 0), (400, preview_tags(), 8)])
        self.rejected(data, 'Cycle')

    def test_ifd_file_field_and_pixel_limits(self):
        data, _, _ = tiff_fixture(directories=[(8, preview_tags(), 400), (400, preview_tags(), 0)])
        self.rejected(data, 'max_ifds', limits=adapter.Limits(max_ifds=1))
        self.rejected(data, 'Input size', limits=adapter.Limits(max_file_bytes=32))
        self.rejected(data, 'max_pixels', limits=adapter.Limits(max_pixels=10))
        self.rejected(data, 'max_tag_bytes', limits=adapter.Limits(max_tag_bytes=4))
        self.rejected(data, 'max_tags_per_ifd', limits=adapter.Limits(max_tags_per_ifd=2))

    def test_no_tile_multistrip_or_partial_rows(self):
        for changes, pattern in (({324: (4, [8])}, 'Tiled'),
                                  ({273: (4, [8, 9]), 279: (4, [8, 9])}, 'Multistrip'),
                                  ({278: (4, [1])}, 'RowsPerStrip'),
                                  ({284: (3, [2])}, 'planar')):
            tags = preview_tags(); tags.update(changes)
            data, _, _ = tiff_fixture(directories=[(8, tags, 0)])
            self.rejected(data, pattern)

    def test_ifd_bits_and_jpeg_actual_precision(self):
        for bits in ([16, 16, 16], [8], [8, 8, 16]):
            tags = preview_tags(); tags[258] = (3, bits)
            data, _, _ = tiff_fixture(directories=[(8, tags, 0)])
            self.rejected(data, 'BitsPerSample')
        jpeg = bytearray(jpeg_fixture())
        sof = jpeg.index(b'\xff\xc0'); jpeg[sof + 4] = 12
        data, _, _ = tiff_fixture(jpeg=bytes(jpeg)); self.rejected(data, 'actualJPEG8')
        for mode in ('L', 'CMYK'):
            data, _, _ = tiff_fixture(jpeg=jpeg_fixture(mode=mode))
            self.rejected(data, 'actualJPEG8')

    def test_jpeg_bounds_soi_eoi_dimension_mismatch(self):
        data, positions, _ = tiff_fixture()
        modified = data.copy(); struct.pack_into('<I', modified, positions[(8, 273)] + 8, 0xfffffff0)
        self.rejected(modified, 'outside file bounds')
        jpeg = jpeg_fixture()
        for malformed in (jpeg[:-2], b'xx' + jpeg[2:], jpeg + b'junk', jpeg[:-2] + b'\xff\xd9junk\xff\xd9'):
            data, _, _ = tiff_fixture(jpeg=malformed)
            self.rejected(data, 'SOI/EOI|trailing JPEG EOI')
        data, _, _ = tiff_fixture(directories=[(8, preview_tags(width=6), 0)])
        self.rejected(data, 'dimensions disagree')
        malformed = bytearray(jpeg); malformed[4:6] = b'\xff\xff'
        data, _, _ = tiff_fixture(jpeg=bytes(malformed))
        self.rejected(data, 'segment outside')

    def test_all_eight_exact_orientation_transposes_both_endian(self):
        jpeg = jpeg_fixture()
        with Image.open(io.BytesIO(jpeg)) as image:
            reference = image.copy()
        for endian in ('<', '>'):
            for value in range(1, 9):
                with self.subTest(endian=endian, orientation=value):
                    data, _, _ = tiff_fixture(endian=endian, jpeg=jpeg,
                                              directories=[(8, preview_tags(orientation=value), 0)])
                    rgb, audit = self.extract(data)
                    method = adapter.ORIENTATION[value][0]
                    expected = reference.transpose(method) if method is not None else reference
                    np.testing.assert_array_equal(rgb, np.asarray(expected))
                    self.assertEqual(audit['orientation']['source'], 'selected_ifd')
                    self.assertFalse(audit['orientation']['resized'])
                    self.assertEqual(audit['orientation']['interpolation'], 'none')
                    self.assertEqual(audit['upright_size'], list(expected.size))

    def test_orientation_inheritance_override_absent_invalid_and_jpeg_ignored(self):
        root = preview_tags(orientation=6); root[330] = (4, [400])
        for selected_orientation, expected, source in ((None, 6, 'ifd0_inherited'), (2, 2, 'selected_ifd')):
            data, _, _ = tiff_fixture(jpeg=jpeg_fixture(orientation=8),
                                      directories=[(8, root, 0), (400, preview_tags(orientation=selected_orientation), 0)])
            _, audit = self.extract(data, 400)
            self.assertEqual(audit['orientation']['value'], expected)
            self.assertEqual(audit['orientation']['source'], source)
            self.assertEqual(audit['orientation']['jpeg_exif_orientation_ignored'], 8)
        data, _, _ = tiff_fixture(directories=[(8, preview_tags(orientation=None), 0)])
        _, audit = self.extract(data); self.assertEqual(audit['orientation']['source'], 'default_absent')
        for value in (0, 9):
            data, _, _ = tiff_fixture(directories=[(8, preview_tags(orientation=value), 0)])
            self.rejected(data, 'Orientation must')

    def test_icc_conflict_reject_and_explicit_ack_without_conversion(self):
        plain, _, _ = tiff_fixture(); reference, _ = self.extract(plain)
        tagged, _, _ = tiff_fixture(jpeg=jpeg_fixture(icc=b'synthetic-not-an-ICC-conversion'))
        self.rejected(tagged, 'explicit acknowledge')
        rgb, audit = self.extract(tagged, acknowledge_unconverted_color_tags=True)
        np.testing.assert_array_equal(rgb, reference)
        self.assertTrue(audit['tag_override_acknowledged'])
        self.assertTrue(audit['input_color_tags'][0]['icc_present'])
        self.assertTrue(audit['unconverted_tag_warnings'])
        tags = preview_tags(); tags[34675] = (7, b'container-ICC')
        tagged, _, _ = tiff_fixture(directories=[(8, tags, 0)])
        self.rejected(tagged, 'selected_ifd: Embedded ICC')
        rgb, audit = self.extract(tagged, acknowledge_unconverted_color_tags=True)
        np.testing.assert_array_equal(rgb, reference)
        self.assertTrue(audit['input_color_tags'][1]['icc_present'])

    def test_complete_post_sos_icc_requires_ack_and_audits_full_payload(self):
        jpeg = jpeg_fixture()
        profile = ImageCms.ImageCmsProfile(ImageCms.createProfile('sRGB')).tobytes()
        late = jpeg[:-2] + icc_segment(profile) + jpeg[-2:]
        with Image.open(io.BytesIO(late)) as image:
            self.assertFalse(image.info.get('icc_profile'))  # Pillow10 metadata blind spot.
        plain, _, _ = tiff_fixture(jpeg=jpeg)
        reference, _ = self.extract(plain)
        for endian in ('<', '>'):
            data, _, _ = tiff_fixture(endian=endian, jpeg=late)
            self.rejected(data, 'selected_jpeg: Embedded ICC')
            rgb, audit = self.extract(data, acknowledge_unconverted_color_tags=True)
            np.testing.assert_array_equal(rgb, reference)
            self.assertEqual(self.input.read_bytes(), data)
            self.assertTrue(audit['tag_override_acknowledged'])
            self.assertTrue(audit['unconverted_tag_warnings'])
            tags = audit['input_color_tags'][0]
            self.assertTrue(tags['icc_present']); self.assertFalse(tags['pillow_icc_present'])
            self.assertEqual(tags['icc_sha256'], hashlib.sha256(profile).hexdigest())
            marker = audit['jpeg_frame']['icc_app2']
            self.assertTrue(marker['present']); self.assertEqual(marker['chunk_count'], 1)
            self.assertEqual(marker['icc_sha256'], tags['icc_sha256'])
            chunk = marker['chunks'][0]
            self.assertTrue(chunk['after_sos'])
            self.assertEqual(chunk['payload_length'], len(profile))
            self.assertEqual(chunk['payload_sha256'], hashlib.sha256(profile).hexdigest())
            self.assertEqual(late[chunk['payload_offset']:chunk['payload_offset'] + chunk['payload_length']], profile)
            inventory = adapter.inspect_dng(self.input)
            self.assertEqual(inventory['directories'][0]['jpeg_frame']['icc_app2'], marker)
            json.dumps(audit)  # No raw ICC bytes leak into the audit schema.

    def test_icc_chunks_across_sos_and_out_of_order_assemble_by_sequence(self):
        jpeg = jpeg_fixture()
        profile = b'opaque-profile-two-halves'
        split = 9
        # Physical chunk2 precedes SOS, chunk1 follows entropy data.
        tagged = (jpeg[:2] + icc_segment(profile[split:], 2, 2) + jpeg[2:-2]
                  + icc_segment(profile[:split], 1, 2) + jpeg[-2:])
        data, _, _ = tiff_fixture(jpeg=tagged)
        self.rejected(data, 'explicit acknowledge')
        _, audit = self.extract(data, acknowledge_unconverted_color_tags=True)
        marker = audit['jpeg_frame']['icc_app2']
        self.assertEqual(marker['icc_sha256'], hashlib.sha256(profile).hexdigest())
        self.assertEqual(marker['payload_bytes'], len(profile))
        self.assertEqual([part['sequence'] for part in marker['chunks']], [1, 2])
        self.assertEqual([part['after_sos'] for part in marker['chunks']], [True, False])
        self.assertFalse(audit['input_color_tags'][0]['pillow_icc_present'])
        self.assertEqual(audit['input_color_tags'][0]['icc_sha256'], marker['icc_sha256'])

    def test_opaque_icc_content_ack_allowed_before_or_after_sos(self):
        jpeg = jpeg_fixture()
        plain, _, _ = tiff_fixture(jpeg=jpeg); reference, _ = self.extract(plain)
        for placement in ('before', 'after'):
            tagged = (jpeg[:2] + icc_segment(b'x') + jpeg[2:] if placement == 'before'
                      else jpeg[:-2] + icc_segment(b'x') + jpeg[-2:])
            data, _, _ = tiff_fixture(jpeg=tagged)
            self.rejected(data, 'Embedded ICC')
            rgb, audit = self.extract(data, acknowledge_unconverted_color_tags=True)
            np.testing.assert_array_equal(rgb, reference)
            self.assertEqual(audit['input_color_tags'][0]['icc_sha256'], hashlib.sha256(b'x').hexdigest())
            self.assertEqual(audit['jpeg_frame']['icc_app2']['payload_interpretation'],
                             'opaque; no ICC profile validation/conversion')

    def test_post_sos_malformed_icc_chunk_sequences_reject_even_with_ack(self):
        jpeg = jpeg_fixture()
        for segments in (icc_segment(b'x', 1, 2),
                         icc_segment(b'x') + icc_segment(b'y'),
                         icc_segment(b'x', 0, 1),
                         icc_segment(b'x', 1, 2) + icc_segment(b'y', 2, 3),
                         icc_segment(b'')):
            data, _, _ = tiff_fixture(jpeg=jpeg[:-2] + segments + jpeg[-2:])
            for ack in (False, True):
                self.rejected(data, 'JPEG ICC APP2', acknowledge_unconverted_color_tags=ack)

    def test_container_exif_and_dng_preview_color_space(self):
        for color in (0, 3, 4):
            tags = preview_tags(); tags[50970] = (4, [color])
            data, _, _ = tiff_fixture(directories=[(8, tags, 0)])
            self.rejected(data, 'PreviewColorSpace')
        tags = preview_tags(); tags[34665] = (4, [400])
        data, _, _ = tiff_fixture(directories=[(8, tags, 0), (400, {40961: (3, [65535])}, 0)])
        self.rejected(data, 'EXIF color space')
        _, audit = self.extract(data, acknowledge_unconverted_color_tags=True)
        self.assertEqual(audit['input_color_tags'][1]['exif_color_space'], 65535)

    def test_nonregular_fifo_rejected_without_blocking(self):
        os.mkfifo(self.input)
        with self.assertRaisesRegex(adapter.PreviewError, 'regular file'):
            adapter.extract_preview_rgb8(self.input, 8, input_domain='encoded-srgb')

    def test_unreferenced_container_tail_is_preserved_and_hashed(self):
        data, _, _ = tiff_fixture()
        tailed = data + b'UNREFERENCED-SYNTHETIC-TAIL-READONLY'
        plain, _ = self.extract(data)
        rgb, audit = self.extract(tailed)
        np.testing.assert_array_equal(rgb, plain)
        self.assertEqual(audit['input_sha256'], hashlib.sha256(tailed).hexdigest())
        self.assertEqual(self.input.read_bytes(), tailed)

    def test_aggregate_value_budget_and_unsigned_sample_format(self):
        data, _, _ = tiff_fixture()
        self.rejected(data, 'aggregate max_decoded_tag_values', limits=adapter.Limits(max_decoded_tag_values=4))
        tags = preview_tags(); tags[339] = (3, [2, 2, 2])
        data, _, _ = tiff_fixture(directories=[(8, tags, 0)])
        self.rejected(data, 'unsigned integer SampleFormat')

    def test_incomplete_icc_sequence_and_invalid_metadata_type(self):
        jpeg = bytearray(jpeg_fixture(icc=b'synthetic-ICC-payload'))
        signature = jpeg.index(b'ICC_PROFILE\x00')
        jpeg[signature + 13] = 2  # Say two chunks exist, but supply just one.
        data, _, _ = tiff_fixture(jpeg=bytes(jpeg))
        self.rejected(data, 'Incomplete JPEG ICC APP2')
        self.rejected(data, 'Incomplete JPEG ICC APP2', acknowledge_unconverted_color_tags=True)
        tags = preview_tags(); tags[274] = (4, [6])
        data, _, _ = tiff_fixture(directories=[(8, tags, 0)])
        self.rejected(data, 'Wrong metadata dtype')

    def test_output_no_overwrite_hardlink_symlink_dangling_and_input_alias(self):
        data, _, _ = tiff_fixture(); self.put(data)
        rgb = np.zeros((3, 5, 3), np.uint8)
        output = self.root / 'new.png'
        for kind in ('regular', 'hardlink', 'symlink', 'dangling'):
            with self.subTest(kind=kind):
                if kind == 'regular':
                    output.write_bytes(b'KEEP')
                elif kind == 'hardlink':
                    os.link(self.input, output)
                else:
                    output.symlink_to(self.input if kind == 'symlink' else self.root / 'absent')
                with self.assertRaises(adapter.PreviewError):
                    adapter._write_outputs(self.input, output, rgb, {})
                output.unlink()
                self.assertEqual(self.input.read_bytes(), data)
        with self.assertRaises(adapter.PreviewError):
            adapter._write_outputs(self.input, self.input, rgb, {})
        sidecar = Path(str(output) + '.json'); sidecar.symlink_to(self.input)
        with self.assertRaises(adapter.PreviewError):
            adapter._write_outputs(self.input, output, rgb, {})
        self.assertFalse(output.exists()); self.assertEqual(self.input.read_bytes(), data)

    def test_exclusive_output_pair_and_failure_cleanup(self):
        data, _, _ = tiff_fixture(); self.put(data)
        output = self.root / 'new.png'; sidecar = Path(str(output) + '.json')
        with mock.patch.object(Image.Image, 'save', side_effect=OSError('synthetic write failure')):
            with self.assertRaisesRegex(OSError, 'synthetic write failure'):
                adapter._write_outputs(self.input, output, np.zeros((3, 5, 3), np.uint8), {})
        self.assertFalse(output.exists()); self.assertFalse(sidecar.exists())
        adapter._write_outputs(self.input, output, np.zeros((3, 5, 3), np.uint8), {'test': True})
        self.assertTrue(output.exists()); self.assertTrue(json.loads(sidecar.read_text())['test'])
        with self.assertRaises(adapter.PreviewError):
            adapter._write_outputs(self.input, output, np.zeros((3, 5, 3), np.uint8), {})

    def test_cli_synthetic_render_and_inspect(self):
        data, _, _ = tiff_fixture(directories=[(8, preview_tags(orientation=6), 0)]); self.put(data)
        resources = self.root / 'cubes'; resources.mkdir()
        cube = 'LUT_3D_SIZE 2\n' + ''.join(f'{r} {g} {b}\n' for b in range(2) for g in range(2) for r in range(2))
        for name in (adapter.renderer.STEVE1, adapter.renderer.STEVE3, adapter.renderer.ETERNAL, adapter.renderer.VIVID):
            (resources / name).write_text(cube)
        with contextlib.redirect_stdout(io.StringIO()) as stdout:
            adapter.main(['inspect', str(self.input)])
        self.assertTrue(json.loads(stdout.getvalue())['directories'][0]['eligible'])
        for look in ('steve', 'eternal', 'vivid'):
            output = self.root / (look + '.png')
            args = [str(self.input), str(output), '--ifd-offset', '8', '--look', look,
                    '--strength', '100', '--input-domain', 'encoded-srgb', '--resource-dir', str(resources)]
            with contextlib.redirect_stdout(io.StringIO()):
                adapter.main(args)
            with Image.open(output) as image:
                self.assertEqual(image.size, (3, 5)); self.assertEqual(image.mode, 'RGB')
                self.assertEqual(image.info['srgb'], 0)
            manifest = json.loads(Path(str(output) + '.json').read_text())
            self.assertTrue(manifest['source']['noRAWdecode'])
            self.assertEqual(manifest['renderer'], 'renderer.render_rgb8')
            if look == 'vivid':
                self.assertEqual(set(manifest['resource_hashes']), {adapter.renderer.VIVID})
                self.assertIn('float16 then float64', manifest['table_values'])
                self.assertIn('Vivid100', manifest['validation_scope'])
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
                adapter.main(args)
            self.assertEqual(caught.exception.code, 2)
        self.assertEqual(self.input.read_bytes(), data)

    def test_cli_post_sos_icc_requires_ack_and_preserves_warnings(self):
        jpeg = jpeg_fixture()
        late = jpeg[:-2] + icc_segment(b'x') + jpeg[-2:]
        data, _, _ = tiff_fixture(jpeg=late); self.put(data)
        resources = self.root / 'cubes'; resources.mkdir()
        cube = 'LUT_3D_SIZE 2\n' + ''.join(f'{r} {g} {b}\n' for b in range(2) for g in range(2) for r in range(2))
        for name in (adapter.renderer.STEVE1, adapter.renderer.STEVE3):
            (resources / name).write_text(cube)
        output = self.root / 'late.png'
        args = [str(self.input), str(output), '--ifd-offset', '8', '--look', 'steve',
                '--input-domain', 'encoded-srgb', '--resource-dir', str(resources)]
        with contextlib.redirect_stderr(io.StringIO()) as stderr, self.assertRaises(SystemExit):
            adapter.main(args)
        self.assertIn('Embedded ICC', stderr.getvalue())
        self.assertFalse(output.exists()); self.assertFalse(Path(str(output) + '.json').exists())
        with contextlib.redirect_stdout(io.StringIO()):
            adapter.main(args + ['--acknowledge-unconverted-color-tags'])
        manifest = json.loads(Path(str(output) + '.json').read_text())
        self.assertTrue(manifest['source']['tag_override_acknowledged'])
        self.assertTrue(manifest['source']['unconverted_tag_warnings'])
        self.assertTrue(manifest['source']['input_color_tags'][0]['icc_present'])
        self.assertTrue(manifest['source']['jpeg_frame']['icc_app2']['chunks'][0]['after_sos'])
        self.assertEqual(self.input.read_bytes(), data)

    def test_cli_requires_explicit_ifd_domain_and_validated_full_strength(self):
        data, _, _ = tiff_fixture(); self.put(data)
        base = [str(self.input), str(self.root / 'no.png'), '--look', 'steve']
        cases = [base, base + ['--ifd-offset', '8'],
                 base + ['--ifd-offset', '8', '--input-domain', 'encoded-srgb', '--strength', 'nan'],
                 [str(self.input), str(self.root / 'no.png'), '--look', 'eternal', '--ifd-offset', '8',
                  '--input-domain', 'encoded-srgb', '--strength', '50'],
                 [str(self.input), str(self.root / 'no.png'), '--look', 'vivid', '--ifd-offset', '8',
                  '--input-domain', 'encoded-srgb', '--strength', '50']]
        for args in cases:
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
                adapter.main(args)
            self.assertEqual(caught.exception.code, 2)
            self.assertFalse((self.root / 'no.png').exists())


if __name__ == '__main__':
    unittest.main(verbosity=2)
