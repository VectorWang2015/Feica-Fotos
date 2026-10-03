"""Isolated rename, preference migration and branded export regressions."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
from PySide6.QtCore import QByteArray, QSettings
from PySide6.QtWidgets import QApplication

from apps.feica_fotos import __version__, engine, settings, ui


class SettingsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name)
        self.legacy = QSettings(str(self.path/'legacy.ini'), QSettings.Format.IniFormat)
        self.target = QSettings(str(self.path/'new.ini'), QSettings.Format.IniFormat)

    def tearDown(self):
        self.legacy = self.target = None
        self.tmp.cleanup()

    def test_imports_preferences_and_preserves_legacy_bytes(self):
        expected = {'resource_dir': '/test/cubes', 'last_dir': '/test/input',
                    'export_dir': '/test/export', 'reduce_transparency': False,
                    'window_geometry': QByteArray(b'geometry-fixture')}
        for key, value in expected.items(): self.legacy.setValue(key, value)
        self.legacy.sync()
        before = (self.path/'legacy.ini').read_bytes()
        self.assertTrue(settings.migrate_legacy_settings(self.target, self.legacy))
        for key, value in expected.items():
            self.assertEqual(self.target.value(key), value)
        self.assertTrue(self.target.value(settings.MIGRATION_MARKER, type=bool))
        self.assertEqual((self.path/'legacy.ini').read_bytes(), before)

    def test_new_values_win_including_empty_and_false(self):
        self.target.setValue('last_dir', '')
        self.target.setValue('reduce_transparency', False)
        self.legacy.setValue('last_dir', '/legacy/input')
        self.legacy.setValue('reduce_transparency', True)
        self.legacy.sync()
        settings.migrate_legacy_settings(self.target, self.legacy)
        self.assertEqual(self.target.value('last_dir'), '')
        self.assertFalse(self.target.value('reduce_transparency', type=bool))

    def test_marker_never_copied_from_legacy(self):
        self.legacy.setValue(settings.MIGRATION_MARKER, False)
        self.legacy.setValue('unknown_setting', 'do-not-copy')
        self.legacy.setValue('last_dir', '/legacy')
        self.legacy.sync()
        settings.migrate_legacy_settings(self.target, self.legacy)
        self.assertTrue(self.target.value(settings.MIGRATION_MARKER, type=bool))
        self.assertFalse(self.target.contains('unknown_setting'))
        self.assertFalse(self.legacy.value(settings.MIGRATION_MARKER, type=bool))

    def test_idempotent_and_removed_target_key_stays_removed_after_restart(self):
        self.legacy.setValue('resource_dir', '/legacy'); self.legacy.sync()
        settings.migrate_legacy_settings(self.target, self.legacy)
        self.target.remove('resource_dir'); self.target.sync()
        self.target = QSettings(str(self.path/'new.ini'), QSettings.Format.IniFormat)
        self.assertFalse(settings.migrate_legacy_settings(self.target, self.legacy))
        self.assertFalse(self.target.contains('resource_dir'))

    def test_missing_legacy_store_is_not_created(self):
        self.assertFalse((self.path/'legacy.ini').exists())
        settings.migrate_legacy_settings(self.target, self.legacy)
        self.assertFalse((self.path/'legacy.ini').exists())
        self.assertEqual(set(self.target.allKeys()), {settings.MIGRATION_MARKER})

    def test_factory_uses_new_identity_and_opens_legacy_only_once(self):
        self.legacy.setValue('export_dir', '/old-output'); self.legacy.sync()
        calls = []
        def factory(org, app):
            calls.append((org, app))
            return self.target if org == 'Feica Fotos' else self.legacy
        self.assertIs(settings.application_settings(factory=factory), self.target)
        self.assertEqual(calls, [('Feica Fotos', 'Feica Fotos'), ('LocalLooks', 'LocalLooks')])
        calls.clear()
        settings.application_settings(factory=factory)
        self.assertEqual(calls, [('Feica Fotos', 'Feica Fotos')])

    def test_failed_preference_flush_does_not_mark_migration_complete(self):
        self.legacy.setValue('last_dir', '/legacy'); self.legacy.sync()
        with patch.object(self.target, 'status', return_value=QSettings.Status.AccessError):
            with self.assertRaisesRegex(RuntimeError, 'could not be saved'):
                settings.migrate_legacy_settings(self.target, self.legacy)
        self.assertFalse(self.target.contains(settings.MIGRATION_MARKER))

    def test_environment_precedence_and_legacy_fallback(self):
        self.target.setValue('resource_dir', '/saved')
        env = {'FEICA_FOTOS_RESOURCE_DIR': '/new', 'LOCAL_LOOKS_RESOURCE_DIR': '/legacy'}
        resolve = lambda explicit=None: settings.resolve_resource_dir(explicit, self.target, '/default', env)
        self.assertEqual(resolve('/explicit'), '/explicit')
        self.assertEqual(resolve(), '/new')
        env.pop('FEICA_FOTOS_RESOURCE_DIR')
        self.assertEqual(resolve(), '/legacy')
        env.clear()
        self.assertEqual(resolve(), '/saved')
        self.target.remove('resource_dir')
        self.assertEqual(resolve(), '/default')

    def test_injected_window_settings_never_open_global_settings(self):
        app = QApplication.instance() or QApplication([])
        with patch.object(ui, 'application_settings', side_effect=AssertionError('global settings accessed')):
            window = ui.MainWindow(self.path, settings=self.target)
        self.assertIs(window.settings, self.target)
        self.assertEqual(window.windowTitle(), 'Feica Fotos')
        self.assertFalse(self.target.contains(settings.MIGRATION_MARKER))
        window._discard_close = True
        window.close(); window.deleteLater(); app.processEvents()


class BrandingTests(unittest.TestCase):
    def test_version_and_software(self):
        self.assertEqual(__version__, '0.3.1')
        self.assertEqual(engine.SOFTWARE, 'Feica Fotos')

    def test_export_software_tags_use_public_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); source = root/'source.png'
            Image.fromarray(np.full((3, 4, 3), 80, np.uint8)).save(source)
            backend = engine.ImageEngine(root)
            doc = backend.load_document(source)
            for suffix in ('png', 'jpg'):
                target = root/('export.'+suffix)
                meta = backend.export(doc, target, 'original')
                self.assertEqual(meta['software'], 'Feica Fotos')
                with Image.open(target) as image:
                    self.assertEqual(image.getexif()[305], 'Feica Fotos')
                    if suffix == 'png': self.assertEqual(image.info['Software'], 'Feica Fotos')


if __name__ == '__main__':
    unittest.main()
