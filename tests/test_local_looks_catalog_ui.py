"""Full-catalog UI tests: synthetic pixels only, offscreen Qt, no vendor LUTs."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
from PySide6.QtCore import QSettings, Qt, QTimer
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from apps.local_looks import ui
from apps.local_looks.engine import ImageDocument, CancelledError


class CatalogFakeEngine:
    """A deterministic fake that supports every current runtime Look ID."""
    def __init__(self):
        self.values = {spec.id: index for index, spec in enumerate(ui.LOOKS)}
        self.render_calls = []
        self.exports = []
        self.thread_ids = []
        self.gates = {}
        self.active = 0
        self.max_active = 0
        self.lock = threading.Lock()

    def load_document(self, path):
        self.thread_ids.append(threading.get_ident())
        with Image.open(path) as image:
            rgb = np.asarray(image).copy()
        return ImageDocument(Path(path), rgb, rgb, 'PNG', {}, rgb.shape[1], rgb.shape[0])

    def render(self, rgb, key, strength=None, *, color_filter=None, cancel=None, progress=None):
        assert key in self.values
        assert color_filter is None or color_filter in ui.LOOK_FILTER_OPTIONS[key]
        kind = 'thumb' if max(rgb.shape[:2]) <= 240 else 'preview'
        call = (kind, key, strength, color_filter, int(rgb[0, 0, 0]))
        with self.lock:
            self.thread_ids.append(threading.get_ident())
            self.render_calls.append(call)
            self.active += 1
            self.max_active = max(self.active, self.max_active)
        try:
            gate = self.gates.get((kind, key, call[4]))
            if gate:
                # Deliberately simulate one uninterruptible backend call; UI
                # must ignore its stale results after cancellation/new document.
                if not gate.wait(5):
                    raise RuntimeError('Test failed to release fake render gate')
            if cancel and cancel.is_set():
                raise CancelledError('cancelled')
            value = self.values[key] + (30 if color_filter else 0)
            return np.clip(rgb.astype(np.int16) + value, 0, 255).astype(np.uint8)
        finally:
            with self.lock:
                self.active -= 1

    def export(self, doc, path, key, strength=None, *, color_filter=None, cancel=None, progress=None, **kwargs):
        self.exports.append((key, strength, color_filter))
        rgb = self.render(doc.rgb, key, strength, color_filter=color_filter, cancel=cancel)
        with Path(path).open('xb') as handle:
            Image.fromarray(rgb).save(handle, format='PNG')
        if progress:
            progress(1)
        return {'path': str(path)}


class CatalogUITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name)
        self.source = self.path / 'synthetic.png'
        Image.fromarray(np.full((320, 480, 3), 40, np.uint8)).save(self.source)
        settings = QSettings(str(self.path / 'settings.ini'), QSettings.Format.IniFormat)
        self.window = ui.MainWindow(self.path, settings=settings)
        self.fake = CatalogFakeEngine()
        self.window.engine = self.fake
        self.window.show()
        self.app.processEvents()

    def tearDown(self):
        for gate in self.fake.gates.values():
            gate.set()
        self.window._after_export = None
        self.window._discard_close = True
        # Cleanup owns only disposable synthetic data. A debounce preview may
        # still be active, so bypass the real unsaved-changes modal explicitly.
        self.window._confirm_transition = lambda callback: True
        self.window.close()
        self.wait(lambda: not self.window.jobs)
        self.window.close()
        self.window.deleteLater()
        self.app.processEvents()
        self.tmp.cleanup()

    def wait(self, predicate, timeout=5):
        deadline = time.monotonic() + timeout
        while not predicate():
            if time.monotonic() > deadline:
                self.fail('Timed out waiting for offscreen UI state')
            self.app.processEvents()
            QTest.qWait(5)
        self.app.processEvents()

    def load(self, finish_thumbnails=True):
        self.window.open_path(str(self.source))
        self.wait(lambda: self.window.doc is not None)
        if finish_thumbnails:
            self.wait(lambda: not self.window.jobs)

    def group(self, name):
        for index in range(self.window.group_tabs.count()):
            if self.window.group_tabs.tabData(index) == name:
                self.window.group_tabs.setCurrentIndex(index)
                self.app.processEvents()
                return
        self.fail('Missing group ' + name)

    def mono(self):
        return next(key for key in self.fake.values
                    if ui.LOOK_GROUPS[key] == 'monochrome' and ui.LOOK_FILTER_OPTIONS[key])

    def render_selection(self):
        self.window._start_preview()
        self.wait(lambda: self.window._ready and not self.window.jobs)

    def test_catalog_titles_full_names_and_menu_cover_original_plus_21(self):
        self.load()
        self.assertEqual(len(ui.LOOKS), 22)
        self.assertEqual(set(self.window.look_buttons), set(ui.LOOK_BY_ID))
        self.assertEqual(set(self.window.look_actions), set(ui.LOOK_BY_ID))
        self.assertEqual([self.window.group_tabs.tabText(i) for i in range(4)],
                         ['全部', '颜色', '单色', 'Artist'])
        for spec in ui.LOOKS:
            button = self.window.look_buttons[spec.id]
            self.assertEqual(button.text(), spec.title)
            self.assertEqual(button.accessibleName(), spec.title)
            self.assertEqual(self.window.look_actions[spec.id].text(), spec.title)
            self.window.look_actions[spec.id].trigger()
            self.assertEqual(self.window.look_id, spec.id)
            self.assertTrue(button.isChecked())
            self.render_selection()
            self.assertEqual(self.window.canvas._image.pixelColor(0, 0).red(),
                             40 + self.fake.values[spec.id])
        # The primary Blue Look and blue attachment legitimately share an ID;
        # attachments must not add five extra cards beyond runtime LOOKS.
        self.assertEqual(len(self.window.look_buttons), len(ui.LOOKS))

    def test_group_browsing_does_not_apply_look_or_change_filter(self):
        self.load()
        key = self.mono()
        self.window.select_look(key)
        self.window.spin.setValue(31.27)
        self.window.color_filter_combo.setCurrentIndex(1)
        self.render_selection()
        signature = self.window._signature()
        frame = self.window.canvas._image.copy()
        rectangle = self.window.canvas.image_rect()
        for group in ('color', 'artist', 'monochrome', 'all'):
            self.group(group)
            expected = {spec.id for spec in ui.LOOKS if spec.id == 'original'
                        or group == 'all' or ui.LOOK_GROUPS[spec.id] == group}
            actual = {key for key, button in self.window.look_buttons.items() if not button.isHidden()}
            self.assertEqual(actual, expected)
            self.assertEqual(self.window._signature(), signature)
            self.assertEqual(self.window.canvas._image, frame)
            self.assertEqual(self.window.canvas.image_rect(), rectangle)
            self.assertTrue(self.window._ready)
            self.assertFalse(self.window.timer.isActive())

    def test_only_legacy_four_shortcuts_and_hidden_selection_is_revealed(self):
        self.load()
        expected = {'original': 'Ctrl+1', 'steve': 'Ctrl+2', 'eternal': 'Ctrl+3', 'vivid': 'Ctrl+4'}
        for key, action in self.window.look_actions.items():
            self.assertEqual(action.shortcut().toString(), expected.get(key, ''))
        self.group('monochrome')
        QTest.keyClick(self.window, Qt.Key.Key_2, Qt.KeyboardModifier.ControlModifier)
        self.wait(lambda: self.window.look_id == 'steve')
        self.assertEqual(self.window.browse_group, ui.LOOK_GROUPS['steve'])
        self.app.processEvents()
        button = self.window.look_buttons['steve']
        self.assertFalse(button.isHidden())
        left = button.mapTo(self.window.look_rail.viewport(), button.rect().topLeft()).x()
        self.assertGreaterEqual(left, 0)
        self.assertLess(left, self.window.look_rail.viewport().width())
        QTest.keyClick(self.window, Qt.Key.Key_1, Qt.KeyboardModifier.ControlModifier)
        self.wait(lambda: self.window.look_id == 'original')

    def test_rail_stays_one_row_at_compact_width_and_keyboard_reaches_end(self):
        self.load()
        for width in (720, 1180):
            self.window.resize(width, 700)
            self.app.processEvents()
            self.assertGreater(self.window.look_rail.horizontalScrollBar().maximum(), 0)
            buttons = list(self.window.look_buttons.values())
            # Checked borders can alter a size hint by two pixels; all cards
            # must occupy the same horizontal row, not an adaptive grid.
            tops = [button.geometry().top() for button in buttons]
            bottoms = [button.geometry().bottom() for button in buttons]
            self.assertLessEqual(max(tops), min(bottoms))
            self.assertLessEqual(max(tops) - min(tops), 2)
            for button in buttons:
                self.assertGreaterEqual(button.width(), button.fontMetrics().horizontalAdvance(button.text()) + 16)
            self.assertGreater(self.window.canvas.height(), 120)
        self.window.look_buttons['original'].setFocus()
        QTest.keyClick(self.window.look_buttons['original'], Qt.Key.Key_End)
        last = ui.LOOKS[-1].id
        self.assertEqual(self.window.look_id, last)
        self.assertTrue(self.window.look_buttons[last].hasFocus())
        self.app.processEvents()
        self.assertGreater(self.window.look_rail.horizontalScrollBar().value(), 0)

    def _assert_rail_height_survives_first_thumbnail_stream(self, width):
        # Set the initial mode while every icon is still empty. A later resize
        # would recompute size hints and conceal the initial clipping bug.
        self.window.resize(width, 780 if width == 1180 else 540)
        self.app.processEvents()
        self.assertTrue(all(button.icon().isNull() for button in self.window.look_buttons.values()))
        initial_size = self.window.size()
        for key, button in self.window.look_buttons.items():
            if not button.isHidden():
                with self.subTest(width=width, phase='empty', look=key):
                    self.assertGreaterEqual(self.window.look_rail.viewport().height(), button.minimumHeight())
        self.load()
        self.assertEqual(self.window.size(), initial_size)
        self.assertTrue(all(not button.icon().isNull() for button in self.window.look_buttons.values()))
        # Only scroll horizontally after the stream; never resize the window or
        # explicitly recalculate rail metrics to make the cards fit afterward.
        for key, button in self.window.look_buttons.items():
            if not button.isHidden():
                self.window.look_rail.ensureWidgetVisible(button, 0, 0)
                self.app.processEvents()
                with self.subTest(width=width, phase='streamed', look=key):
                    self.assertFalse(button.visibleRegion().isEmpty())
                    self.assertEqual(button.visibleRegion().boundingRect().height(), button.height())
                    self.assertEqual(self.window.size(), initial_size)

    def test_initial_wide_rail_does_not_clip_cards_after_stream_without_resize(self):
        self._assert_rail_height_survives_first_thumbnail_stream(1180)

    def test_initial_compact_rail_does_not_clip_cards_after_stream_without_resize(self):
        self._assert_rail_height_survives_first_thumbnail_stream(720)

    def test_each_look_remembers_fractional_strength_and_new_document_resets(self):
        self.load()
        for index, spec in enumerate(ui.LOOKS):
            self.window.select_look(spec.id)
            if spec.adjustable:
                self.window.spin.setValue(10 + index + .27)
                self.assertEqual(self.window.slider.value(), round((10 + index + .27) * 100))
        for index, spec in enumerate(ui.LOOKS):
            self.window.select_look(spec.id)
            self.assertEqual(self.window.spin.value(), 10 + index + .27 if spec.adjustable else spec.default_strength)
        self.window.open_path(str(self.source), skip_confirm=True)
        self.wait(lambda: self.window.doc_revision == 2 and not self.window.jobs)
        self.assertEqual(self.window.look_id, 'original')
        self.assertEqual(self.window.strengths, {spec.id: spec.default_strength for spec in ui.LOOKS})
        self.assertIsNone(self.window.color_filter)

    def test_filter_visibility_none_default_labels_and_reset(self):
        self.load()
        for spec in ui.LOOKS:
            self.window.select_look(spec.id)
            allowed = ui.LOOK_GROUPS[spec.id] == 'monochrome' and bool(ui.LOOK_FILTER_OPTIONS[spec.id])
            self.assertEqual(self.window.color_filter_combo.isVisible(), allowed)
            self.assertIsNone(self.window.color_filter_combo.itemData(0))
            self.assertIsNone(self.window.color_filter)
            if allowed:
                actual = [self.window.color_filter_combo.itemData(i)
                          for i in range(1, self.window.color_filter_combo.count())]
                self.assertEqual(actual, list(ui.LOOK_FILTER_OPTIONS[spec.id]))
        key = self.mono()
        option = ui.LOOK_FILTER_OPTIONS[key][0]
        custom = dict(ui.COLOR_FILTERS)
        custom[option] = dict(custom[option], label='Official Red Label')
        with patch.object(ui, 'COLOR_FILTERS', custom):
            self.window.select_look(key)
            self.assertEqual(self.window.color_filter_combo.itemText(1), 'Official Red Label')
        self.window.color_filter_combo.setCurrentIndex(1)
        self.assertEqual(self.window.color_filter, option)
        self.window.select_look('vivid')
        self.assertIsNone(self.window.color_filter)
        self.assertTrue(self.window.color_filter_combo.isHidden())
        self.window.select_look(key)
        self.assertIsNone(self.window.color_filter)
        self.window.color_filter_combo.setCurrentIndex(1)
        self.window.cancel_work()
        self.assertIsNone(self.window.color_filter)
        self.assertEqual(self.window.look_id, 'original')

    def test_filter_enters_preview_export_signature_and_dirty_state(self):
        self.load()
        key = self.mono()
        self.window.select_look(key)
        self.window.spin.setValue(42.37)
        without_filter = self.window._signature()
        option = ui.LOOK_FILTER_OPTIONS[key][-1]
        self.window.color_filter_combo.setCurrentIndex(self.window.color_filter_combo.findData(option))
        self.assertNotEqual(self.window._signature(), without_filter)
        self.assertEqual(self.window._signature()[-1], option)
        self.render_selection()
        self.assertIn(('preview', key, 42.37, option, 40), self.fake.render_calls)
        self.window.compare_button.setChecked(True)
        self.window.start_export(self.path / 'filtered.png')
        self.wait(lambda: not self.window.jobs)
        self.assertEqual(self.fake.exports[-1], (key, 42.37, option))
        self.assertFalse(self.window._is_dirty())
        self.window.color_filter_combo.setCurrentIndex(0)
        self.assertTrue(self.window._is_dirty())
        self.render_selection()
        self.assertEqual(self.fake.render_calls[-1][1:4], (key, 42.37, None))
        self.assertTrue(all(thread != threading.get_ident() for thread in self.fake.thread_ids))
        self.assertLessEqual(self.fake.max_active, self.window.pool.maxThreadCount())

    def test_rapid_filter_changes_ignore_stale_preview(self):
        self.load()
        key = self.mono()
        gate = threading.Event()
        self.fake.gates[('preview', key, 40)] = gate
        self.window.select_look(key)
        self.window.color_filter_combo.setCurrentIndex(1)
        self.window._start_preview()
        self.wait(lambda: any(call[:2] == ('preview', key) for call in self.fake.render_calls))
        self.window.color_filter_combo.setCurrentIndex(0)
        self.window._start_preview()
        gate.set()
        self.wait(lambda: self.window._ready and not self.window.jobs)
        self.assertIsNone(self.window.color_filter)
        self.assertEqual(self.window.canvas._image.pixelColor(0, 0).red(), 40 + self.fake.values[key])
        self.assertEqual(self.fake.render_calls[-1][3], None)

    def test_streaming_thumbnails_appear_before_completion_and_group_priority_updates(self):
        self.group('color')
        first = next(spec.id for spec in ui.LOOKS if ui.LOOK_GROUPS[spec.id] == 'color')
        gate = threading.Event()
        self.fake.gates[('thumb', first, 40)] = gate
        self.load(finish_thumbnails=False)
        self.wait(lambda: any(call[:2] == ('thumb', first) for call in self.fake.render_calls))
        self.wait(lambda: not self.window.look_buttons['original'].icon().isNull())
        self.assertIn(self.window._thumbnail_token, self.window.jobs)
        self.assertTrue(self.window._ready)
        self.assertEqual(self.window.canvas._image.pixelColor(0, 0).red(), 40)
        ticks = []
        QTimer.singleShot(0, lambda: ticks.append(True))
        self.wait(lambda: bool(ticks))
        self.group('artist')
        gate.set()
        self.wait(lambda: not self.window.jobs)
        thumbs = [call[1] for call in self.fake.render_calls if call[0] == 'thumb']
        self.assertEqual(thumbs[0], first)
        self.assertEqual(ui.LOOK_GROUPS[thumbs[1]], 'artist')
        self.assertEqual(len(thumbs), len(ui.LOOKS) - 1)
        self.assertEqual(set(self.window._thumbnail_cache), set(ui.LOOK_BY_ID))
        self.assertTrue(all(not button.icon().isNull() for button in self.window.look_buttons.values()))
        count = len(self.fake.render_calls)
        for group in ('all', 'color', 'monochrome', 'artist'):
            self.group(group)
        self.assertEqual(len(self.fake.render_calls), count)
        self.assertTrue(all(thread != threading.get_ident() for thread in self.fake.thread_ids))

    def test_old_thumbnail_streams_are_ignored_after_new_document_or_cancel(self):
        first = next(spec.id for spec in ui.LOOKS if spec.id != 'original')
        gate = threading.Event()
        self.fake.gates[('thumb', first, 40)] = gate
        self.load(finish_thumbnails=False)
        self.wait(lambda: any(call[:2] == ('thumb', first) for call in self.fake.render_calls))
        old_token = self.window._thumbnail_token
        new_source = self.path / 'second.png'
        Image.fromarray(np.full((320, 480, 3), 90, np.uint8)).save(new_source)
        self.window.open_path(str(new_source), skip_confirm=True)
        self.wait(lambda: self.window.doc.path == new_source)
        self.window._job_item(old_token, ('original', np.zeros((88, 240, 3), np.uint8)))
        gate.set()
        self.wait(lambda: not self.window.jobs)
        icon = self.window.look_buttons['original'].icon().pixmap(120, 44).toImage()
        self.assertEqual(icon.pixelColor(0, 0).red(), 90)
        self.assertEqual(self.window.canvas._image.pixelColor(0, 0).red(), 90)
        self.window._start_thumbnails()
        token = self.window._thumbnail_token
        self.window.cancel_work()
        self.window._job_item(token, ('original', np.zeros((88, 240, 3), np.uint8)))
        self.assertNotIn('original', self.window._thumbnail_cache)
        self.wait(lambda: not self.window.jobs)

    def test_official_body_is_complete_literal_text_and_only_footer_is_italic(self):
        self.load()
        key = self.mono()
        body = '  官方原文 <b>不是标签</b> &  双空格\n\n第二段。\n' + '完整原文。' * 250
        entry = dict(ui.LOOK_INFO[key], body=body, preview=True, official_name='Official <Title>')
        with patch.object(ui, 'LOOK_INFO', dict(ui.LOOK_INFO, **{key: entry})):
            self.window.select_look(key)
            rectangle = self.window.canvas.image_rect()
            self.window.set_filter_info_visible(True)
            self.app.processEvents()
            text = self.window.info_body.toPlainText()
            self.assertEqual(text, body + '\n' + ui.PREVIEW_NOTICE)
            self.assertEqual(text.count(ui.PREVIEW_NOTICE), 1)
            self.assertEqual(self.window.info_title.text(), entry['official_name'])
            self.assertIn('&lt;b&gt;', self.window.info_body.toHtml())
            cursor = self.window.info_body.document().find(ui.PREVIEW_NOTICE)
            self.assertTrue(cursor.charFormat().fontItalic())
            self.assertGreater(self.window.info_body.verticalScrollBar().maximum(), 0)
            self.assertEqual(self.window.canvas.image_rect(), rectangle)
            self.window._escape()
            self.assertEqual(self.window.look_id, key)
            self.assertFalse(self.window.info_panel.isVisible())

    def test_empty_body_is_not_invented_and_attachment_preview_uses_runtime(self):
        self.load()
        key = self.mono()
        entry = dict(ui.LOOK_INFO[key], body='', preview=False)
        with patch.object(ui, 'LOOK_INFO', dict(ui.LOOK_INFO, **{key: entry})), \
                patch.object(ui, 'look_preview', return_value=True) as preview:
            self.window.select_look(key)
            self.assertEqual(self.window.info_body.toPlainText(), '')
            self.window.color_filter_combo.setCurrentIndex(1)
            self.assertEqual(self.window.info_body.toPlainText(), ui.PREVIEW_NOTICE)
            preview.assert_called_with(key, strength=self.window._strength(), color_filter=self.window.color_filter)
        self.window.select_look('original')
        self.assertEqual(self.window.info_body.toPlainText(), '')


if __name__ == '__main__':
    unittest.main(verbosity=2)
