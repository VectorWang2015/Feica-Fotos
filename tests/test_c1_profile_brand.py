"""Metadata-only profile branding preserves the complete stored color transform."""
import struct
import unittest

from experiments.c1 import c1_single_profile as c1
from experiments.c1 import rename_profile_brand as brand


def fixture():
    tags = [(b'desc', c1.description('LocalLooks-Q1-Synthetic-S100-v1')),
            (b'cprt', b'text'+bytes(4)+b'Original vendor attribution\0'),
            (b'A2B0', b'mft2'+bytes(range(4, 252))),
            (b'wtpt', b'XYZ '+bytes(range(4, 20)))]
    header = bytearray(128)
    header[8:12] = bytes.fromhex('02100000')
    header[12:24] = b'scnrRGB Lab '
    header[36:40] = b'acsp'
    header[80:84] = b'TEST'
    result = header+struct.pack('>I', len(tags))+bytes(12*len(tags))
    for i, (sig, payload) in enumerate(tags):
        result.extend(bytes((-len(result)) % 4))
        struct.pack_into('>4sII', result, 132+12*i, sig, len(result), len(payload))
        result.extend(payload)
    struct.pack_into('>I', result, 0, len(result))
    return bytes(result)


class BrandTests(unittest.TestCase):
    def test_rewrites_only_description_payload_and_directory_layout(self):
        before = fixture()
        after = brand.replace_description(before, 'Feica Fotos-Q1-Synthetic-S100-v1')
        self.assertTrue(all(brand.unchanged_payloads(before, after).values()))
        self.assertEqual(before[4:128], after[4:128])
        self.assertEqual(brand.display_name(dict(c1.read_tags(after))[b'desc']),
                         'Feica Fotos-Q1-Synthetic-S100-v1')
        self.assertNotEqual(dict(c1.read_tags(before))[b'desc'], dict(c1.read_tags(after))[b'desc'])
        for sig, payload in c1.read_tags(before):
            if sig != b'desc':
                self.assertEqual(payload, dict(c1.read_tags(after))[sig])

    def test_idempotent_rewrite(self):
        once = brand.replace_description(fixture(), 'Feica Fotos-Q1-Synthetic-S100-v1')
        self.assertEqual(brand.replace_description(once, 'Feica Fotos-Q1-Synthetic-S100-v1'), once)

    def test_product_prefix_conversion_keeps_remaining_label(self):
        self.assertEqual(brand.renamed_description('LocalLooks-Q1-Sepia-S075-FilterRed-v1'),
                         'Feica Fotos-Q1-Sepia-S075-FilterRed-v1')
        self.assertEqual(brand.renamed_description('Feica Fotos-Bypass'), 'Feica Fotos-Bypass')
        with self.assertRaises(ValueError):
            brand.renamed_description('Leica Q Generic')

    def test_machine_filename_uses_compact_brand(self):
        self.assertEqual(brand.renamed_path('colors/LeicaQTyp116-LocalLooks-Vivid-S100-v1.icm'),
                         'colors/LeicaQTyp116-FeicaFotos-Vivid-S100-v1.icm')

    def test_paths_and_unrelated_profiles_rejected(self):
        for p in ('../LeicaQTyp116-LocalLooks-Vivid.icm', '/LeicaQTyp116-LocalLooks-Vivid.icm',
                  'colors\\LeicaQTyp116-LocalLooks-Vivid.icm', 'LeicaQTyp116-Generic.icm'):
            with self.assertRaises(ValueError): brand.renamed_path(p)

    def test_invalid_or_truncated_description_rejected(self):
        for p in (b'', b'mluc'+bytes(100), b'desc'+bytes(4)+struct.pack('>I', 30)+b'bad'):
            with self.assertRaises(ValueError): brand.display_name(p)

    def test_payload_comparison_detects_color_changes(self):
        before = fixture()
        after = bytearray(before)
        offset = struct.unpack_from('>I', after, 132+12*2+4)[0]
        after[offset+9] ^= 1
        result = brand.unchanged_payloads(before, bytes(after))
        self.assertFalse(result['A2B0_exact'])
        self.assertFalse(result['all_non_description_payloads'])

    def test_generator_descriptions_and_filename_brand(self):
        from experiments.c1 import c1_single_look_preview as preview
        from experiments.c1 import look_composition as composition
        self.assertEqual(preview.NAME, 'Feica Fotos-Q1-VividPreview-Native33-v1')
        self.assertEqual(c1.PROBE_NAME, 'Feica Fotos-C1Single-OrderProbe-v1')
        for v in composition.build_variant_plan():
            self.assertIn('LeicaQTyp116-FeicaFotos-', v.relative_path)
            self.assertTrue(v.description.startswith('Feica Fotos-Q1-'))
            self.assertLessEqual(len(v.description.encode('ascii'))+1, 67)
            self.assertEqual(brand.display_name(c1.description(v.description)), v.description)


if __name__ == '__main__':
    unittest.main()
