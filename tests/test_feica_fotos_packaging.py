"""Portable source/build contracts; no photos, vendor tables, or native build."""
import importlib.util
import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch

from apps.feica_fotos import catalog, ui
from reproduction.ios_looks import renderer

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "feica_fotos_builder", ROOT / "scripts/build_feica_fotos.py")
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


class PackagingTests(unittest.TestCase):
    def test_public_executable_name_preserves_space_as_one_argument(self):
        self.assertEqual(builder.APP_NAME, 'Feica Fotos')
        stage = Path('build with spaces')
        entry = stage/'feica_fotos_entry.py'
        command = builder.pyinstaller_command(stage, entry, True)
        self.assertEqual(command[command.index('--name')+1], 'Feica Fotos')
        with patch.object(builder.sys, 'platform', 'win32'):
            windows = builder.pyinstaller_command(stage, entry, True)
        self.assertEqual(windows[windows.index('--icon')+1],
                         str(stage/'apps/feica_fotos/assets/feica-fotos.ico'))
        self.assertIn('--windowed', windows)
        self.assertIn('apps/feica_fotos/settings.py', builder.SOURCE_FILES)

    def test_source_defaults_use_prepared_filter_directory(self):
        expected = ROOT / "filters/looks"
        self.assertEqual(ui.WORKSPACE_RESOURCES, expected)
        self.assertEqual(renderer.RESOURCE_DIR, expected)

    def test_all_build_sources_and_assets_are_present_without_resources(self):
        sources, resources = builder.load_inputs(None)
        expected = set(builder.SOURCE_FILES + builder.ICON_FILES + builder.APP_DATA_FILES)
        self.assertEqual(set(sources), expected)
        self.assertFalse(resources)
        self.assertTrue(all(sources.values()))

    def test_resource_allowlist_matches_exact_catalog_filenames(self):
        raw = json.loads(catalog.CATALOG_PATH.read_text(encoding="utf-8"))
        names = set()
        for look in raw["looks"]:
            for field in ("primary_cube", "secondary_cube"):
                if look.get(field):
                    names.add(look[field]["filename"])
        names.update(item["cube"]["filename"] for item in raw["color_filters"])
        self.assertEqual(len(names), 28)
        self.assertEqual(builder.resource_names(json.dumps(raw)), tuple(sorted(names)))

    def test_catalog_resource_names_cannot_escape_resource_directory(self):
        for name in ("../table.cube", "nested/table.cube", "nested\\table.cube",
                     "C:table.cube", ".hidden.cube", "table.icc"):
            raw = {"looks": [{"primary_cube": {"filename": name}}], "color_filters": []}
            with self.subTest(name=name), self.assertRaises(ValueError):
                builder.resource_names(json.dumps(raw))

    def test_bundle_keeps_explicit_look_resources_destination(self):
        stage = Path("synthetic-build-stage")
        entry = stage / "feica_fotos_entry.py"
        resource_data = str(stage / "look-resources") + os.pathsep + "look-resources"
        without = builder.pyinstaller_command(stage, entry, False)
        with_resources = builder.pyinstaller_command(stage, entry, True)
        self.assertNotIn(resource_data, without)
        self.assertIn(resource_data, with_resources)
        self.assertIn("--onedir", with_resources)
        self.assertEqual(with_resources[-1], str(entry))


if __name__ == "__main__":
    unittest.main()
