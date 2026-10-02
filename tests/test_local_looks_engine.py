"""Synthetic-only unittest coverage for the offline Local Looks backend.

Run from the repository root:
    /usr/bin/python3 -m unittest discover -s tests -p 'test_local_looks_engine.py' -v
No user photos, vendor resources, network, Qt or pytest are used.
"""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError
import io
import os
from pathlib import Path
import struct
import tempfile
import threading
import unittest
from unittest import mock
import zlib

import numpy as np
from PIL import Image, ImageCms, ImageFile

from apps.local_looks import catalog, engine
from reproduction.ios_looks import color_spaces, renderer


def pattern(width=7, height=5):
    y, x = np.mgrid[:height, :width]
    return np.stack(((x * 29 + y * 3) % 256, (x * 7 + y * 43) % 256,
                     (x * 51 + y * 17) % 256), axis=-1).astype(np.uint8)


def jpeg_bytes(width=7, height=5, *, orientation=None, icc=None, mode="RGB", color_space=None):
    options = {"quality": 97, "subsampling": 0}
    if orientation is not None or color_space is not None:
        exif = Image.Exif()
        if orientation is not None:
            exif[274] = orientation
            exif[34853] = {1: "N", 2: (1.0, 2.0, 3.0)}
        if color_space is not None:
            exif[34665] = {40961: color_space}
        options["exif"] = exif
    if icc is not None:
        options["icc_profile"] = icc
    stream = io.BytesIO()
    Image.fromarray(pattern(width, height)).convert(mode).save(stream, "JPEG", **options)
    return stream.getvalue()


def icc_segment(payload, sequence=1, total=1):
    chunk = b"ICC_PROFILE\x00" + bytes((sequence, total)) + payload
    return b"\xff\xe2" + struct.pack(">H", len(chunk) + 2) + chunk


def dng_bytes(previews, *, root_icc=None):
    """Tiny classic TIFF with independent JPEG strips; each dict is one IFD."""
    offsets = [8 + i * 400 for i in range(len(previews))]
    data = bytearray(offsets[-1] + 400)
    data[:8] = b"II" + struct.pack("<HI", 42, offsets[0])
    for i, (offset, preview) in enumerate(zip(offsets, previews)):
        width, height = preview.get("width", 7), preview.get("height", 5)
        jpeg = jpeg_bytes(width, height, icc=preview.get("icc"))
        tags = {254: (4, [1]), 256: (4, [width]), 257: (4, [height]),
                258: (3, [8, 8, 8]), 259: (3, [7]), 262: (3, [6]),
                273: (4, [0]), 274: (3, [preview.get("orientation", 1)]),
                277: (3, [3]), 278: (4, [height]), 279: (4, [len(jpeg)]),
                50706: (1, [1, 4, 0, 0])}
        if preview.get("cfa"):
            tags[262] = (3, [32803])
            tags[254] = (4, [0])
        if i == 0 and root_icc is not None:
            tags[34675] = (7, root_icc)
        jpeg_start = len(data)
        data.extend(jpeg)
        tags[273] = (4, [jpeg_start])
        struct.pack_into("<H", data, offset, len(tags))
        for n, (tag, (dtype, values)) in enumerate(sorted(tags.items())):
            start = offset + 2 + n * 12
            payload = (bytes(values) if dtype in (1, 7)
                       else struct.pack("<" + str(len(values)) + {3: "H", 4: "I"}[dtype], *values))
            struct.pack_into("<HHI", data, start, tag, dtype, len(values))
            if len(payload) <= 4:
                data[start + 8:start + 12] = payload.ljust(4, b"\x00")
            else:
                if len(data) % 2:
                    data.append(0)
                struct.pack_into("<I", data, start + 8, len(data))
                data.extend(payload)
        struct.pack_into("<I", data, offset + 2 + len(tags) * 12,
                         offsets[i + 1] if i + 1 < len(offsets) else 0)
    return bytes(data), offsets


def png_rgb16_bytes():
    def chunk(name, payload):
        return (struct.pack(">I", len(payload)) + name + payload
                + struct.pack(">I", zlib.crc32(name + payload)))
    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 16, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(b"\x00\x80\x00\x40\x00\xff\xff"))
            + chunk(b"IEND", b""))


class EngineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.resources = self.root / "synthetic-cubes"
        self.resources.mkdir()
        self.engine = engine.ImageEngine(self.resources)

    def tearDown(self):
        self.temp.cleanup()

    def cube(self, name, values):
        values = np.asarray(values, dtype=np.float64)
        if values.shape == (3,):
            values = np.broadcast_to(values, (2, 2, 2, 3))
        path = self.resources / name
        rows = ["LUT_3D_SIZE " + str(values.shape[0])]
        rows += [" ".join(format(float(v), ".17g") for v in row) for row in values.reshape(-1, 3)]
        path.write_text("\n".join(rows) + "\n", encoding="utf8")
        return path

    def fixture_tables(self):
        rng = np.random.default_rng(739)
        for name in (renderer.STEVE1, renderer.STEVE3, renderer.ETERNAL, renderer.VIVID):
            self.cube(name, rng.uniform(-0.05, 1.05, (3, 3, 3, 3)))

    def photo(self, name="source.png", rgb=None):
        path = self.root / name
        Image.fromarray(pattern() if rgb is None else rgb).save(path)
        return path

    def test_look_specs_defaults_continuous_strengths_and_rules(self):
        self.assertEqual([x.id for x in engine.LOOKS], [
            "original", "standard", "vivid", "natural", "monochrome_natural",
            "monochrome_high_contrast", "steve", "eternal", "greg", "hundred_years",
            "classic", "contemporary", "bleach", "blue", "brass", "chrome", "cine", "pure",
            "selenium", "sepia", "silver", "teal"])
        for spec in engine.LOOKS:
            self.assertEqual(spec.default_strength, 50 if spec.id == "steve" else 100)
        self.assertEqual(list(engine.LookSpec.__dataclass_fields__), ["id", "title", "default_strength", "adjustable"])
        self.assertFalse(engine.LOOK_BY_ID["original"].adjustable)
        for look in (spec.id for spec in engine.LOOKS if spec.id != "original"):
            self.assertTrue(engine.LOOK_BY_ID[look].adjustable)
            for value in (-1, 101, float("nan"), float("inf"), True, "50"):
                with self.subTest(look=look, value=value), self.assertRaises(ValueError):
                    self.engine.render(pattern(), look, value)
        for look, value in (("unknown", 100), ("original", 0), ("original", 50)):
            with self.subTest(look=look, value=value), self.assertRaises(ValueError):
                self.engine.render(pattern(), look, value)
        self.assertEqual({k: engine.LOOK_STRENGTH_RULES[k] for k in ("original", "steve", "eternal", "vivid", "greg")}, {
            "original": "identity", "steve": "observed_endpoint_mix", "greg": "candidate_endpoint_mix",
            "eternal": "product_source_blend", "vivid": "product_source_blend"})
        with self.assertRaises(TypeError):
            engine.LOOK_STRENGTH_RULES["eternal"] = "observed_endpoint_mix"

    def bound_cube(self, look, values, role="primary_cube"):
        binding = engine.LOOK_BINDINGS[look][role]
        path = self.cube(binding["filename"], values)
        table = renderer.TextureLUT.from_cube(path)
        if binding["restore_half"]:
            table = renderer.TextureLUT(table.values.astype(np.float16).astype(np.float64))
        return table

    def test_catalog_bindings_groups_resources_and_immutability(self):
        self.assertEqual(set(engine.GROUP_TITLES), {"original", "color", "monochrome", "artist"})
        main_files = set()
        for spec in engine.LOOKS:
            binding = engine.LOOK_BINDINGS[spec.id]
            self.assertEqual(engine.LOOK_GROUPS[spec.id], binding["group"])
            self.assertIn(binding["group"], engine.GROUP_TITLES)
            self.assertIn(binding["input_space"], ("srgb", "display-p3"))
            self.assertIn(binding["output_space"], ("srgb", "display-p3"))
            if spec.id == "original":
                self.assertIsNone(binding["primary_cube"])
                continue
            for field in ("primary_cube", "secondary_cube"):
                cube = binding[field]
                if cube is None:
                    continue
                main_files.add(cube["filename"])
                self.assertEqual(Path(cube["filename"]).name, cube["filename"])
                self.assertTrue(cube["filename"].endswith(".cube"))
                self.assertRegex(cube["sha256"], r"^[0-9a-f]{64}$")
                self.assertGreaterEqual(cube["grid_size"], 2)
            with self.assertRaises(TypeError):
                binding["primary_cube"]["filename"] = "replaced.cube"
        self.assertEqual(len(main_files), 23)
        self.assertEqual(set(engine.COLOR_FILTERS), {"red", "orange", "yellow", "green", "blue"})
        filter_files = {f["cube"]["filename"] for f in engine.COLOR_FILTERS.values()}
        self.assertEqual(len(filter_files | main_files), 28)
        self.assertNotIn("red", engine.LOOK_BY_ID)
        self.assertFalse(catalog.CATALOG["availability_exhaustive"])
        self.assertFalse(catalog.CATALOG["conversion_clip"])
        self.assertEqual(catalog.CATALOG["source_blend_domain"], "encoded-srgb-after-output-conversion")
        for key in ("steve", "eternal"):
            self.assertFalse(engine.LOOK_BINDINGS[key]["primary_cube"]["restore_half"])
        self.assertTrue(engine.LOOK_BINDINGS["vivid"]["primary_cube"]["restore_half"])

    def test_catalog_dynamic_preview_policy(self):
        self.assertFalse(engine.LOOK_PREVIEW["original"])
        self.assertFalse(engine.LOOK_PREVIEW["steve"])
        self.assertTrue(engine.LOOK_PREVIEW["eternal"])
        self.assertTrue(engine.LOOK_PREVIEW["vivid"])
        self.assertFalse(engine.look_preview("steve", 37.25))
        for look in ("eternal", "vivid"):
            self.assertFalse(engine.look_preview(look))
            self.assertFalse(engine.look_preview(look, 100))
            self.assertTrue(engine.look_preview(look, 37.25))
        for look in engine.LOOKS:
            if look.id not in ("original", "steve", "eternal", "vivid"):
                self.assertTrue(engine.look_preview(look.id, 100))
        self.assertTrue(engine.look_preview("greg", 100, "red"))

    def test_photo_load_never_requires_catalog_tables_and_candidate_load_is_lazy(self):
        photo = self.photo()
        with mock.patch.object(renderer.TextureLUT, "from_cube", side_effect=AssertionError("must not read cube")):
            missing_engine = engine.ImageEngine(self.root / "no-cubes")
            doc = missing_engine.load_document(photo)
            np.testing.assert_array_equal(missing_engine.render(doc.rgb, "original"), doc.rgb)
        with self.assertRaisesRegex(engine.ResourceError, "Required classic.*Classic_sRGB"):
            self.engine.render(pattern(), "classic")
        self.bound_cube("classic", [0.00196079] * 3)
        self.assertTrue(np.all(self.engine.render(pattern(), "classic") == 0))
        self.assertEqual(len(self.engine._cache), 1)

    def test_p3_source_blend_after_conversion_preserves_extended_values(self):
        table = self.bound_cube("standard", [1.0, 0.05, 0.15])
        rgb = pattern()
        source = rgb.astype(np.float64) / 255
        p3 = color_spaces.convert_rgb(source, "srgb", "display-p3", clip=False)
        full_p3 = renderer.sample_texture(table, p3)
        full = color_spaces.convert_rgb(full_p3, "display-p3", "srgb", clip=False)
        self.assertTrue(np.any((full < 0) | (full > 1)))
        for strength in (0, 37.25, 100):
            expected = (rgb if strength == 0 else renderer.quantize_u8(full) if strength == 100
                        else renderer.quantize_u8(source + (full - source) * (strength / 100)))
            with mock.patch.object(color_spaces, "convert_rgb", wraps=color_spaces.convert_rgb) as spy:
                actual = self.engine.render(rgb, "standard", strength)
                self.assertTrue(all(call.kwargs["clip"] is False for call in spy.call_args_list))
            np.testing.assert_array_equal(actual, expected)
        proper = self.engine.render(rgb, "standard", 37.25)
        wrong_clip = renderer.quantize_u8(source + (np.clip(full, 0, 1) - source) * 0.3725)
        wrong_p3 = renderer.quantize_u8(color_spaces.convert_rgb(
            p3 + (full_p3 - p3) * 0.3725, "display-p3", "srgb", clip=False))
        self.assertFalse(np.array_equal(proper, wrong_clip))
        self.assertFalse(np.array_equal(proper, wrong_p3))
        self.assertFalse(np.array_equal(self.engine.render(rgb, "standard"), rgb))

    def test_greg_candidate_mix_uses_two_half_tables_in_p3(self):
        primary = self.bound_cube("greg", [1.0, 0.1, 0.0])
        secondary = self.bound_cube("greg", [0.0, 1.0, 0.1], role="secondary_cube")
        rgb = pattern()
        p3 = color_spaces.convert_rgb(rgb.astype(np.float64) / 255, "srgb", "display-p3", clip=False)
        for strength in (0, 37.25, 100):
            mixed = renderer.mix_tables(secondary, primary, p3, strength / 100)
            expected = renderer.quantize_u8(color_spaces.convert_rgb(mixed, "display-p3", "srgb", clip=False))
            np.testing.assert_array_equal(self.engine.render(rgb, "greg", strength), expected)
        np.testing.assert_array_equal(self.engine.render(rgb, "greg"), self.engine.render(rgb, "greg", 100))
        self.assertFalse(np.array_equal(self.engine.render(rgb, "greg", 0), rgb))
        a = color_spaces.convert_rgb(renderer.sample_texture(secondary, p3), "display-p3", "srgb", clip=False)
        b = color_spaces.convert_rgb(renderer.sample_texture(primary, p3), "display-p3", "srgb", clip=False)
        wrong_domain_mix = renderer.quantize_u8(a + (b - a) * 0.3725)
        self.assertFalse(np.array_equal(self.engine.render(rgb, "greg", 37.25), wrong_domain_mix))

    def test_attachment_samples_before_both_tones_and_source_blends_unfiltered_original(self):
        rng = np.random.default_rng(7119)
        attachment = engine.COLOR_FILTERS["red"]["cube"]
        self.cube(attachment["filename"], rng.uniform(0, 1, (3, 3, 3, 3)))
        filter_table = renderer.TextureLUT.from_cube(self.resources / attachment["filename"])
        filter_table = renderer.TextureLUT(filter_table.values.astype(np.float16).astype(np.float64))
        source = pattern().astype(np.float64) / 255
        for look in ("greg", "monochrome_natural"):
            primary = self.bound_cube(look, rng.uniform(0, 1, (3, 3, 3, 3)))
            p3 = color_spaces.convert_rgb(source, "srgb", "display-p3", clip=False)
            filtered = renderer.sample_texture(filter_table, p3)
            if look == "greg":
                secondary = self.bound_cube(look, rng.uniform(0, 1, (3, 3, 3, 3)), "secondary_cube")
                working = renderer.mix_tables(secondary, primary, filtered, 0.3725)
                expected = renderer.quantize_u8(color_spaces.convert_rgb(working, "display-p3", "srgb", clip=False))
            else:
                working = renderer.sample_texture(primary, filtered)
                full = color_spaces.convert_rgb(working, "display-p3", "srgb", clip=False)
                expected = renderer.quantize_u8(source + (full - source) * 0.3725)
                np.testing.assert_array_equal(self.engine.render(pattern(), look, 0, color_filter="red"), pattern())
            np.testing.assert_array_equal(self.engine.render(pattern(), look, 37.25, color_filter="red"), expected)
        self.assertEqual(len(self.engine._cache), 4)  # The shared attachment was cached once.

    def test_disallowed_attachments_rejected_without_reading_resources(self):
        for look, attachment in (("original", "red"), ("steve", "red"), ("standard", "red"),
                                 ("monochrome_natural", "unknown")):
            with self.subTest(look=look), mock.patch.object(renderer.TextureLUT, "from_cube") as spy:
                with self.assertRaisesRegex(ValueError, "not allowed"):
                    self.engine.render(pattern(), look, color_filter=attachment)
                spy.assert_not_called()

    def test_candidate_export_records_recipe_domain_preview_and_attachment(self):
        self.bound_cube("monochrome_natural", [0.4, 0.4, 0.4])
        self.cube(engine.COLOR_FILTERS["green"]["cube"]["filename"], [0.2, 0.5, 0.7])
        doc = self.engine.load_document(self.photo())
        meta = self.engine.export(doc, self.root / "candidate.png", "monochrome_natural", 37.25, color_filter="green")
        self.assertEqual(meta["recipe"], "source_to_primary")
        self.assertEqual(meta["working_domain"], "display-p3")
        self.assertEqual(meta["output_domain"], "srgb")
        self.assertTrue(meta["preview"])
        self.assertEqual(meta["color_filter"], "green")
        with Image.open(meta["path"]) as output:
            np.testing.assert_array_equal(np.asarray(output), self.engine.render(doc.rgb, "monochrome_natural", 37.25, color_filter="green"))

    def test_original_without_any_resources_is_independent_identity(self):
        source = pattern()
        progress = []
        output = engine.ImageEngine(self.root / "missing").render(source, "original", progress=progress.append)
        np.testing.assert_array_equal(output, source)
        self.assertFalse(np.shares_memory(output, source))
        output[:] = 0
        self.assertTrue(np.any(source))
        self.assertEqual(progress, [0.0, 1.0])

    def test_steve_two_endpoints_default50_not_identity_mix(self):
        self.cube(renderer.STEVE1, [0.1, 0.2, 0.3])
        self.cube(renderer.STEVE3, [0.7, 0.8, 0.9])
        for strength in (0, 25, 37.25, 50, 75, 100):
            output = self.engine.render(pattern(), "steve", strength)
            endpoint = renderer.quantize_u8(np.array([0.1, 0.2, 0.3])
                                            + np.array([0.6, 0.6, 0.6]) * (strength / 100))
            # Independent arithmetic can differ at an exact .5 rounding tie;
            # compare renderer's exact floating-point operand/mix order too.
            expected = renderer.render_rgb8(pattern(), "steve", strength, self.resources)
            np.testing.assert_array_equal(output, expected)
            self.assertLessEqual(np.max(np.abs(output[0, 0].astype(int) - endpoint.astype(int))), 1)
        np.testing.assert_array_equal(self.engine.render(pattern(), "steve"),
                                      self.engine.render(pattern(), "steve", 50))
        self.assertFalse(np.array_equal(self.engine.render(pattern(), "steve", 0), pattern()))

    def test_chunk_sampling_matches_reference_for_all_looks(self):
        self.fixture_tables()
        rgb = np.random.default_rng(711).integers(0, 256, (401, 501, 3), dtype=np.uint8)
        for look, strength in (("steve", 37.5), ("eternal", 100), ("vivid", 100)):
            with self.subTest(look=look):
                progress = []
                actual = self.engine.render(rgb, look, strength, progress=progress.append)
                expected = renderer.render_rgb8(rgb, look, strength, self.resources)
                np.testing.assert_array_equal(actual, expected)
                self.assertEqual(progress[0], 0.0)
                self.assertEqual(progress[-1], 1.0)
                self.assertEqual(progress, sorted(progress))
                self.assertEqual(len(progress), 3)

    def test_product_source_blend_continuous_strength_and_exact_endpoints(self):
        self.fixture_tables()
        rgb = np.random.default_rng(151).integers(0, 256, (17, 31, 3), dtype=np.uint8)
        coordinates = rgb.astype(np.float64) / 255
        for look, filename in (("eternal", renderer.ETERNAL), ("vivid", renderer.VIVID)):
            table = renderer.TextureLUT.from_cube(self.resources / filename)
            if look == "vivid":
                table = renderer.TextureLUT(table.values.astype(np.float16).astype(np.float64))
            full = renderer.sample_texture(table, coordinates)
            for strength in (0, 0.01, 37.25, 50.13, 99.99, 100):
                with self.subTest(look=look, strength=strength):
                    expected = (rgb if strength == 0 else renderer.quantize_u8(full) if strength == 100
                                else renderer.quantize_u8(coordinates + (full - coordinates) * (strength / 100)))
                    np.testing.assert_array_equal(self.engine.render(rgb, look, strength), expected)
            np.testing.assert_array_equal(self.engine.render(rgb, look),
                                          renderer.render_rgb8(rgb, look, 100, self.resources))
            # Revisit zero after rendering arbitrary strengths: no accumulated
            # edits or round-tripping through already-rendered RGB8 pixels.
            np.testing.assert_array_equal(self.engine.render(rgb, look, 0), rgb)

    def test_product_source_blend_has_no_early_quantization_or_clamp(self):
        rgb = np.zeros((2, 3, 3), dtype=np.uint8)
        coordinates = rgb.astype(np.float64) / 255
        for look, filename in (("eternal", renderer.ETERNAL), ("vivid", renderer.VIVID)):
            with self.subTest(look=look):
                self.cube(filename, [1.49 / 255, 1.2, -0.2])
                table = renderer.TextureLUT.from_cube(self.resources / filename)
                if look == "vivid":
                    table = renderer.TextureLUT(table.values.astype(np.float16).astype(np.float64))
                full = renderer.sample_texture(table, coordinates)
                expected = renderer.quantize_u8(coordinates + (full - coordinates) * 0.3725)
                early_u8 = renderer.quantize_u8(full).astype(np.float64) / 255
                wrong = renderer.quantize_u8(coordinates + (early_u8 - coordinates) * 0.3725)
                self.assertTrue(np.all(expected[..., 0] == 1))
                self.assertTrue(np.all(wrong[..., 0] == 0))
                self.assertTrue(np.all(expected[..., 1] > wrong[..., 1]))
                np.testing.assert_array_equal(self.engine.render(rgb, look, 37.25), expected)

    def test_export_records_strength_rule_and_fractional_strength(self):
        self.fixture_tables()
        doc = self.engine.load_document(self.photo())
        for look in (engine.LOOK_BY_ID[k] for k in ("original", "steve", "eternal", "vivid")):
            strength = 100 if look.id == "original" else 37.25
            metadata = self.engine.export(doc, self.root / f"strength-{look.id}.png", look.id, strength)
            self.assertEqual(metadata["strength"], strength)
            self.assertEqual(metadata["strength_rule"], engine.LOOK_STRENGTH_RULES[look.id])

    def test_vivid_only_restores_half_samples(self):
        # Above one half-byte quantization threshold in decimal, below it after
        # float16 recovery: this distinguishes a missing/overbroad conversion.
        for name in (renderer.VIVID, renderer.ETERNAL, renderer.STEVE1, renderer.STEVE3):
            self.cube(name, [0.00196079] * 3)
        self.assertTrue(np.all(self.engine.render(pattern(), "vivid") == 0))
        self.assertTrue(np.all(self.engine.render(pattern(), "eternal") == 1))
        self.assertTrue(np.all(self.engine.render(pattern(), "steve") == 1))

    def test_resource_missing_invalid_and_only_required_tables_read(self):
        with self.assertRaisesRegex(engine.ResourceError, "steve.*LUT.*missing|Required steve"):
            self.engine.render(pattern(), "steve")
        self.cube(renderer.ETERNAL, [0.2, 0.3, 0.4])
        self.engine.render(pattern(), "eternal")  # No other 27 resources needed.
        (self.resources / renderer.VIVID).write_text("LUT_3D_SIZE\n", encoding="utf8")
        with self.assertRaises(engine.ResourceError):
            self.engine.render(pattern(), "vivid")
        with self.assertRaises(ValueError):
            engine.ImageEngine(None)

    def test_cache_thread_safe_and_no_repeated_resource_reads(self):
        self.fixture_tables()
        loader = renderer.TextureLUT.from_cube
        with mock.patch.object(renderer.TextureLUT, "from_cube", wraps=loader) as spy:
            with ThreadPoolExecutor(max_workers=4) as pool:
                outputs = list(pool.map(lambda _: self.engine.render(pattern(), "steve"), range(12)))
            self.assertEqual(spy.call_count, 2)
        for output in outputs:
            np.testing.assert_array_equal(output, outputs[0])
        for path in self.resources.iterdir():
            path.unlink()
        np.testing.assert_array_equal(self.engine.render(pattern(), "steve"), outputs[0])

    def test_documents_have_immutable_pixels_and_max1600_preview(self):
        source = pattern(1701, 7)
        path = self.photo(rgb=source)
        original_bytes = path.read_bytes()
        doc = self.engine.load_document(path)
        self.assertEqual((doc.width, doc.height), (1701, 7))
        self.assertEqual(max(doc.preview_rgb.shape[:2]), 1600)
        self.assertEqual(doc.rgb.dtype, np.uint8)
        self.assertIn("assumption", doc.source_info)
        for rgb in (doc.rgb, doc.preview_rgb):
            self.assertFalse(rgb.flags.writeable)
            with self.assertRaises(ValueError):
                rgb.setflags(write=True)
            with self.assertRaises(ValueError):
                rgb[0, 0] = 0
        with self.assertRaises(FrozenInstanceError):
            doc.width = 0
        np.testing.assert_array_equal(doc.rgb, source)
        self.assertEqual(original_bytes, path.read_bytes())

    def test_all_eight_jpeg_orientations_exact_pixel_permutation(self):
        actions = {1: None, 2: Image.Transpose.FLIP_LEFT_RIGHT, 3: Image.Transpose.ROTATE_180,
                   4: Image.Transpose.FLIP_TOP_BOTTOM, 5: Image.Transpose.TRANSPOSE,
                   6: Image.Transpose.ROTATE_270, 7: Image.Transpose.TRANSVERSE, 8: Image.Transpose.ROTATE_90}
        for orientation, transpose in actions.items():
            with self.subTest(orientation=orientation):
                path = self.root / f"orientation{orientation}.JPG"
                path.write_bytes(jpeg_bytes(orientation=orientation))
                with Image.open(path) as image:
                    expected = np.asarray(image.transpose(transpose) if transpose is not None else image)
                doc = self.engine.load_document(path)
                np.testing.assert_array_equal(doc.rgb, expected)
                self.assertEqual((doc.width, doc.height), (expected.shape[1], expected.shape[0]))
                self.assertEqual(doc.source_info["output_orientation"], 1)
        path.write_bytes(jpeg_bytes(orientation=9))
        with self.assertRaisesRegex(engine.ImageLoadError, "Orientation"):
            self.engine.load_document(path)

    def test_jpeg_icc_runs_real_imagecms_conversion(self):
        profile = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
        path = self.root / "profile.jpg"
        path.write_bytes(jpeg_bytes(icc=profile, orientation=6))
        with Image.open(path) as image:
            expected = ImageCms.profileToProfile(image, ImageCms.ImageCmsProfile(io.BytesIO(profile)),
                                                 ImageCms.createProfile("sRGB"), outputMode="RGB")
            expected = np.asarray(expected.transpose(Image.Transpose.ROTATE_270))
        with mock.patch.object(ImageCms, "profileToProfile", wraps=ImageCms.profileToProfile) as spy:
            doc = self.engine.load_document(path)
            self.assertEqual(spy.call_count, 1)
        np.testing.assert_array_equal(doc.rgb, expected)
        self.assertIn("ICC to standard sRGB", doc.source_info["color_conversion"])
        self.assertNotIn("assumption", doc.source_info)
        with mock.patch.object(ImageCms, "profileToProfile", return_value=Image.new("RGB", (7, 5), (23, 42, 67))):
            converted = self.engine.load_document(path)
            self.assertTrue(np.all(converted.rgb == [23, 42, 67]))

    def test_jpeg_without_icc_rejects_explicit_non_srgb_exif(self):
        path = self.root / "exif-conflict.jpg"
        for color_space in (65535, 2):
            with self.subTest(color_space=color_space):
                path.write_bytes(jpeg_bytes(color_space=color_space))
                with self.assertRaisesRegex(engine.ImageLoadError, "EXIF color space.*valid embedded ICC"):
                    self.engine.load_document(path)

    def test_jpeg_without_icc_accepts_srgb_or_absent_exif_and_records_it(self):
        path = self.root / "exif-srgb.jpg"
        for color_space in (1, None):
            with self.subTest(color_space=color_space):
                path.write_bytes(jpeg_bytes(color_space=color_space))
                doc = self.engine.load_document(path)
                self.assertEqual(doc.source_info["exif_color_space"], color_space)
                self.assertIn("assumption", doc.source_info)

    def test_jpeg_valid_icc_overrides_uncalibrated_exif(self):
        profile = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
        path = self.root / "icc-with-uncalibrated-exif.jpg"
        path.write_bytes(jpeg_bytes(icc=profile, color_space=65535))
        with mock.patch.object(ImageCms, "profileToProfile", wraps=ImageCms.profileToProfile) as spy:
            doc = self.engine.load_document(path)
            self.assertEqual(spy.call_count, 1)
        self.assertEqual(doc.source_info["exif_color_space"], 65535)
        self.assertIn("ICC to standard sRGB", doc.source_info["color_conversion"])
        self.assertNotIn("assumption", doc.source_info)

    def test_png_exif_color_space_conflicts_and_profile_override(self):
        path = self.root / "exif-colors.png"
        profile = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
        for color_space in (65535, 2, 1):
            exif = Image.Exif()
            exif[34665] = {40961: color_space}
            Image.fromarray(pattern()).save(path, exif=exif)
            if color_space != 1:
                with self.assertRaisesRegex(engine.ImageLoadError, "EXIF color space.*valid embedded ICC"):
                    self.engine.load_document(path)
            else:
                self.assertEqual(self.engine.load_document(path).source_info["exif_color_space"], 1)
            Image.fromarray(pattern()).save(path, exif=exif, icc_profile=profile)
            with mock.patch.object(ImageCms, "profileToProfile", wraps=ImageCms.profileToProfile) as spy:
                doc = self.engine.load_document(path)
                self.assertEqual(spy.call_count, 1)
            self.assertEqual(doc.source_info["exif_color_space"], color_space)
            self.assertNotIn("assumption", doc.source_info)

    def test_unreadable_exif_ifd_cannot_silently_assume_srgb(self):
        path = self.root / "malformed-exif.jpg"
        path.write_bytes(jpeg_bytes(color_space=65535))
        with mock.patch.object(Image.Exif, "get_ifd", side_effect=ValueError("bad IFD offset")):
            with self.assertRaisesRegex(engine.ImageLoadError, "Cannot parse EXIF.*bad IFD offset"):
                self.engine.load_document(path)

    def test_corrupt_mismatched_and_incomplete_jpeg_icc_rejected(self):
        path = self.root / "bad-icc.jpg"
        for icc in (b"not a profile", ImageCms.ImageCmsProfile(ImageCms.createProfile("LAB")).tobytes()):
            path.write_bytes(jpeg_bytes(icc=icc))
            with self.assertRaisesRegex(engine.ImageLoadError, "ICC conversion"):
                self.engine.load_document(path)
        raw = jpeg_bytes()
        path.write_bytes(raw[:2] + icc_segment(b"chunk", total=2) + raw[2:])
        with self.assertRaisesRegex(engine.ImageLoadError, "Incomplete JPEG ICC"):
            self.engine.load_document(path)

    def test_jpeg_icc_after_scan_is_not_silently_ignored(self):
        raw = jpeg_bytes()
        path = self.root / "late-icc.jpg"
        path.write_bytes(raw[:-2] + icc_segment(b"unconverted profile") + raw[-2:])
        with self.assertRaisesRegex(engine.ImageLoadError, "ICC conversion"):
            self.engine.load_document(path)

    def test_cmyk_without_profile_rejected(self):
        path = self.root / "cmyk.jpg"
        path.write_bytes(jpeg_bytes(mode="CMYK"))
        with self.assertRaisesRegex(engine.ImageLoadError, "CMYK.*without.*ICC"):
            self.engine.load_document(path)

    def test_small_dng_largest_eligible_preview_and_cfa_exclusion(self):
        raw, offsets = dng_bytes([{"width": 12, "height": 10, "cfa": True},
                                  {"width": 4, "height": 3},
                                  {"width": 9, "height": 5, "orientation": 6}])
        path = self.root / "previews.DNG"
        path.write_bytes(raw)
        doc = self.engine.load_document(path)
        self.assertEqual(doc.source_kind, "DNG preview")
        self.assertEqual((doc.width, doc.height), (5, 9))
        self.assertEqual(doc.source_info["ifd_offset"], offsets[2])
        self.assertEqual(doc.source_info["selection_policy"], engine.DNG_SELECTION_POLICY)
        self.assertFalse(doc.source_info["tag_override_acknowledged"])
        self.assertEqual(path.read_bytes(), raw)

    def test_dng_tied_area_no_preview_and_unconverted_profiles_rejected(self):
        path = self.root / "unsupported.dng"
        profile = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
        cases = [([{"width": 6, "height": 4}, {"width": 8, "height": 3}], None, "Ambiguous"),
                 ([{"cfa": True}], None, "No supported.*CFA"),
                 ([{"icc": profile}], None, "unconverted"),
                 ([{}], profile, "unconverted")]
        for previews, root_icc, message in cases:
            with self.subTest(message=message):
                path.write_bytes(dng_bytes(previews, root_icc=root_icc)[0])
                with self.assertRaisesRegex(engine.ImageLoadError, message):
                    self.engine.load_document(path)

    def test_dng_source_change_between_public_adapter_reads_rejected(self):
        path = self.root / "change.dng"
        path.write_bytes(dng_bytes([{}])[0])
        extract = engine.dng_preview.extract_preview_rgb8
        def changed(*args, **kwargs):
            rgb, info = extract(*args, **kwargs)
            info["input_sha256"] = "changed"
            return rgb, info
        with mock.patch.object(engine.dng_preview, "extract_preview_rgb8", side_effect=changed):
            with self.assertRaisesRegex(engine.ImageLoadError, "changed"):
                self.engine.load_document(path)

    def test_pre_cancel_and_chunk_cancel_are_typed(self):
        event = threading.Event()
        event.set()
        with self.assertRaises(engine.CancelledError):
            self.engine.render(pattern(), "steve", cancel=event)  # Before resource access.
        event.clear()
        progress = []
        def cancel_after_first_chunk(value):
            progress.append(value)
            if value > 0:
                event.set()
        with self.assertRaises(engine.CancelledError):
            self.engine.render(pattern(501, 401), "original", progress=cancel_after_first_chunk, cancel=event)
        self.assertEqual(len(progress), 2)
        self.assertLess(progress[-1], 1.0)

    def test_export_png_jpeg_rgb_srgb_clean_exif_no_sidecar(self):
        source = self.root / "gps-source.jpg"
        source.write_bytes(jpeg_bytes(orientation=6))
        before = source.read_bytes()
        doc = self.engine.load_document(source)
        for suffix, expected_format in ((".png", "PNG"), (".jpg", "JPEG"), (".jpeg", "JPEG")):
            target = self.root / ("export" + suffix)
            progress = []
            metadata = self.engine.export(doc, target, "original", progress=progress.append)
            self.assertEqual(metadata["format"], expected_format)
            self.assertFalse(metadata["gps_included"])
            self.assertEqual(progress[-1], 1.0)
            self.assertEqual(progress, sorted(progress))
            with Image.open(target) as image:
                self.assertEqual(image.format, expected_format)
                self.assertEqual(image.mode, "RGB")
                self.assertEqual(image.size, (doc.width, doc.height))
                self.assertTrue(image.info["icc_profile"])
                profile = ImageCms.ImageCmsProfile(io.BytesIO(image.info["icc_profile"]))
                self.assertIn("sRGB", ImageCms.getProfileDescription(profile))
                exif = image.getexif()
                self.assertEqual(exif[274], 1)
                self.assertEqual(exif.get_ifd(34665)[40961], 1)
                self.assertNotIn(34853, exif)
                self.assertNotIn(37500, exif.get_ifd(34665))
                if suffix == ".png":
                    np.testing.assert_array_equal(np.asarray(image), doc.rgb)
            self.assertFalse(Path(str(target) + ".json").exists())
        self.assertEqual(source.read_bytes(), before)

    def test_export_new_only_regular_symlink_hardlink_alias_and_dangling(self):
        source = self.photo()
        doc = self.engine.load_document(source)
        before = source.read_bytes()
        targets = [source, source.parent / "." / source.name]
        existing = self.root / "exists.jpg"
        existing.write_bytes(b"keep me")
        targets.append(existing)
        hard = self.root / "hard.png"
        os.link(source, hard)
        targets.append(hard)
        for name, destination in (("link.png", source), ("dangling.png", self.root / "missing.png")):
            link = self.root / name
            try:
                link.symlink_to(destination)
            except (OSError, NotImplementedError):
                continue  # Windows without symlink privilege still runs the rest.
            targets.append(link)
        for target in targets:
            with self.subTest(target=target.name), self.assertRaises(engine.ExportError):
                self.engine.export(doc, target, "original")
        self.assertEqual(existing.read_bytes(), b"keep me")
        self.assertEqual(source.read_bytes(), before)
        self.assertEqual(hard.read_bytes(), before)
        for target in targets:
            self.assertTrue(os.path.lexists(target))

    def test_export_oexcl_race_preserves_competing_file(self):
        doc = self.engine.load_document(self.photo())
        target = self.root / "raced.png"
        def create_at_render_end(value):
            if value == 0.9:
                target.write_bytes(b"not ours")
        with self.assertRaises(engine.ExportError):
            self.engine.export(doc, target, "original", progress=create_at_render_end)
        self.assertEqual(target.read_bytes(), b"not ours")

    def test_export_save_failure_removes_only_new_file(self):
        doc = self.engine.load_document(self.photo())
        target = self.root / "failed.png"
        def fail(image, handle, **kwargs):
            handle.write(b"partial")
            raise OSError("synthetic encoder failure")
        with mock.patch.object(Image.Image, "save", side_effect=fail, autospec=True):
            with self.assertRaisesRegex(engine.ExportError, "encoder failure"):
                self.engine.export(doc, target, "original")
        self.assertFalse(os.path.lexists(target))
        self.assertTrue(doc.path.exists())

    @unittest.skipIf(os.name == "nt", "Replacing an open inode is POSIX-only")
    def test_export_cleanup_preserves_replacement_inode(self):
        doc = self.engine.load_document(self.photo())
        target = self.root / "replaced.png"
        def replace_and_fail(image, handle, **kwargs):
            target.unlink()
            target.write_bytes(b"replacement, not ours")
            raise OSError("synthetic replacement failure")
        with mock.patch.object(Image.Image, "save", side_effect=replace_and_fail, autospec=True):
            with self.assertRaises(engine.ExportError):
                self.engine.export(doc, target, "original")
        self.assertEqual(target.read_bytes(), b"replacement, not ours")

    def test_export_cancel_during_encoding_and_final_callback_cleanup(self):
        doc = self.engine.load_document(self.photo())
        event = threading.Event()
        target = self.root / "cancelled.png"
        event.set()
        with self.assertRaises(engine.CancelledError):
            self.engine.export(doc, target, "original", cancel=event)
        self.assertFalse(target.exists())
        event.clear()
        save = Image.Image.save
        def save_then_cancel(image, handle, **kwargs):
            save(image, handle, **kwargs)
            event.set()
        with mock.patch.object(Image.Image, "save", side_effect=save_then_cancel, autospec=True):
            with self.assertRaises(engine.CancelledError):
                self.engine.export(doc, target, "original", cancel=event)
        self.assertFalse(target.exists())
        event.clear()
        def final_cancel(value):
            if value == 1.0:
                event.set()
        with self.assertRaises(engine.CancelledError):
            self.engine.export(doc, target, "original", cancel=event, progress=final_cancel)
        self.assertFalse(target.exists())

    def test_export_validation_does_not_create_directories_or_files(self):
        doc = self.engine.load_document(self.photo())
        for name, options in (("bad.tiff", {}), ("bad.jpg", {"format": "PNG"}),
                              ("bad.png", {"quality": 101}), ("bad.png", {"quality": True})):
            with self.subTest(name=name, options=options), self.assertRaises((engine.ExportError, ValueError)):
                self.engine.export(doc, self.root / name, "original", **options)
            self.assertFalse((self.root / name).exists())
        with self.assertRaises(engine.ExportError):
            self.engine.export(doc, self.root / "absent" / "output.png", "original")
        self.assertFalse((self.root / "absent").exists())

    def test_16bit_and_unsupported_mode_inputs_never_silently_truncate(self):
        path = self.root / "16bit.png"
        path.write_bytes(png_rgb16_bytes())
        with self.assertRaisesRegex(engine.ImageLoadError, "16-bit"):
            self.engine.load_document(path)
        Image.new("RGBA", (7, 5)).save(path)
        with self.assertRaisesRegex(engine.ImageLoadError, "alpha"):
            self.engine.load_document(path)
        raw = bytearray(jpeg_bytes())
        start = raw.index(b"\xff\xc0")
        raw[start + 4] = 12
        path = self.root / "12bit.jpg"
        path.write_bytes(raw)
        with self.assertRaisesRegex(engine.ImageLoadError, "8-bit"):
            self.engine.load_document(path)
        with self.assertRaises(ValueError):
            self.engine.render(pattern().astype(np.uint16), "original")

    def test_truncated_and_corrupt_images_rejected(self):
        jpg = jpeg_bytes()
        good_png = self.photo().read_bytes()
        for name, raw in (("broken.jpg", jpg[:-20]), ("broken.jpg", b"garbage image bytes"),
                          ("broken.png", good_png[:-20]), ("broken.dng", b"II*\x00\x08\x00\x00\x00")):
            with self.subTest(name=name):
                path = self.root / name
                path.write_bytes(raw)
                with self.assertRaises(engine.ImageLoadError):
                    self.engine.load_document(path)
        with mock.patch.object(ImageFile, "LOAD_TRUNCATED_IMAGES", True):
            with self.assertRaisesRegex(engine.ImageLoadError, "truncated-image"):
                self.engine.load_document(self.photo("strict.png"))

    def test_size_limits_before_decode(self):
        raw = bytearray(jpeg_bytes())
        sof = raw.index(b"\xff\xc0")
        struct.pack_into(">HH", raw, sof + 5, 9000, 9000)
        path = self.root / "oversize.jpg"
        path.write_bytes(raw)
        with self.assertRaisesRegex(engine.ImageLoadError, "64 MP"):
            self.engine.load_document(path)
        with mock.patch.object(engine, "MAX_FILE_BYTES", 16):
            with self.assertRaisesRegex(engine.ImageLoadError, "512 MiB"):
                self.engine.load_document(self.photo("bounded.png"))

    def test_load_and_render_can_run_concurrently(self):
        self.fixture_tables()
        path = self.photo()
        def task(i):
            if i % 2:
                return self.engine.load_document(path).rgb
            return self.engine.render(pattern(), "eternal")
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(task, range(12)))
        for i, result in enumerate(results):
            np.testing.assert_array_equal(result, pattern() if i % 2 else results[0])


if __name__ == "__main__":
    unittest.main()
