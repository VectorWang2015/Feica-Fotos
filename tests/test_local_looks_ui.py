"""Synthetic, offscreen UI state tests; no vendor resources or user pictures."""
import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
import tempfile
import threading
import time
import unittest
from pathlib import Path
import numpy as np
from PIL import Image
from PySide6.QtWidgets import QApplication, QFileDialog, QLabel
from PySide6.QtGui import QPalette
from PySide6.QtTest import QTest
from PySide6.QtCore import Qt, QSettings
from apps.local_looks.ui import MainWindow
from apps.local_looks.engine import ImageDocument, CancelledError, LOOKS
from apps.local_looks.look_info import LOOK_INFO


class FakeEngine:
    def __init__(self):self.exports=[];self.thread_ids=[]
    def load_document(self,path):
        self.thread_ids.append(threading.get_ident());p=Path(path)
        with Image.open(p) as im:rgb=np.asarray(im.convert('RGB')).copy()
        rgb.setflags(write=False)
        return ImageDocument(p,rgb,rgb,'JPEG',{'test':True},rgb.shape[1],rgb.shape[0])
    def render(self,rgb,key,strength=None,*,color_filter=None,cancel=None,progress=None):
        self.thread_ids.append(threading.get_ident())
        if key=='steve':
            for _ in range(20):
                if cancel and cancel.wait(.003):raise CancelledError('cancelled')
        if cancel and cancel.is_set():raise CancelledError('cancelled')
        values={spec.id:1 for spec in LOOKS}; values.update(original=0,steve=2,eternal=4,vivid=6)
        value=values[key]
        return np.clip(rgb.astype(np.int16)+value,0,255).astype(np.uint8)
    def export(self,doc,path,key,strength=None,*,cancel=None,progress=None,**kwargs):
        self.exports.append((path,key,strength))
        if cancel and cancel.is_set():raise CancelledError('cancelled')
        result=self.render(doc.rgb,key,strength,cancel=cancel)
        with Path(path).open('xb') as f:Image.fromarray(result).save(f,format='PNG')
        if progress:progress(1)
        return {'path':str(path)}


class UITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.app=QApplication.instance() or QApplication([])
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.path=Path(self.tmp.name)
        self.source=self.path/'input.png';Image.fromarray(np.full((24,32,3),30,np.uint8)).save(self.source)
        self.window=MainWindow(self.path,settings=QSettings(str(self.path/'settings.ini'),QSettings.Format.IniFormat));self.fake=FakeEngine();self.window.engine=self.fake
        self.window.show();self.app.processEvents()
    def tearDown(self):
        self.window._after_export=None;self.window._discard_close=True
        self.window.close();self.wait(lambda:not self.window.jobs)
        self.window.close();self.app.processEvents();self.window.deleteLater();self.app.processEvents();self.tmp.cleanup()
    def wait(self,predicate,timeout=4):
        end=time.monotonic()+timeout
        while not predicate():
            if time.monotonic()>end:self.fail('Timed out waiting for UI state')
            self.app.processEvents();QTest.qWait(5)
        self.app.processEvents()
    def load(self):
        self.window.open_path(str(self.source));self.wait(lambda:self.window.doc is not None and not self.window.jobs)
        self.assertFalse(self.window.canvas._image.isNull())
    def test_auxiliary_dialogs_keep_system_text_palette(self):
        dialog=QFileDialog(self.window);dialog.setOption(QFileDialog.Option.DontUseNativeDialog,True);dialog.setDirectory(str(self.path))
        dialog.show();self.app.processEvents()
        expected=self.app.palette().color(QPalette.ColorRole.WindowText)
        labels=dialog.findChildren(QLabel)
        self.assertTrue(labels)
        for label in labels:self.assertEqual(label.palette().color(QPalette.ColorRole.WindowText),expected)
        dialog.close();dialog.deleteLater();self.app.processEvents()
    def test_empty_state_has_disabled_export_and_look_controls(self):
        self.assertFalse(self.window.export_button.isEnabled())
        self.assertTrue(self.window.open_button.isEnabled())
        self.assertTrue(all(not b.isEnabled() for b in self.window.look_buttons.values()))
    def test_load_is_async_and_source_pixels_are_visible(self):
        self.load();self.assertTrue(self.window.export_button.isEnabled())
        self.assertEqual(self.window.canvas._image.pixelColor(0,0).red(),30)
        self.assertTrue(all(t!=threading.get_ident() for t in self.fake.thread_ids))
        self.assertEqual(self.window.look_id,'original')
    def test_compare_never_changes_export_look(self):
        self.load();self.window.select_look('vivid');self.window._start_preview()
        self.wait(lambda:self.window._ready and not self.window.jobs)
        self.assertEqual(self.window.canvas._image.pixelColor(0,0).red(),36)
        self.window.compare_button.setChecked(True)
        self.assertEqual(self.window.canvas._image.pixelColor(0,0).red(),30)
        target=self.path/'out.png';self.window.start_export(target)
        self.wait(lambda:not self.window.jobs)
        self.assertEqual(self.fake.exports[-1][1],'vivid')
        with Image.open(target) as im:self.assertEqual(im.getpixel((0,0)),(36,36,36))
        self.assertFalse(self.window._is_dirty())
    def test_rapid_selection_latest_result_wins(self):
        self.load();self.window.select_look('steve');self.window._start_preview()
        self.window.select_look('eternal');self.window._start_preview()
        self.window.select_look('vivid');self.window._start_preview()
        self.wait(lambda:self.window._ready and not self.window.jobs)
        self.assertEqual(self.window.look_id,'vivid')
        self.assertEqual(self.window.canvas._image.pixelColor(0,0).red(),36)
    def test_new_document_invalidates_old_preview(self):
        self.load();self.window.select_look('steve');self.window._start_preview()
        new=self.path/'new.png';Image.fromarray(np.full((20,30,3),80,np.uint8)).save(new)
        self.window.open_path(str(new),skip_confirm=True)
        self.wait(lambda:not self.window.jobs and self.window.doc.path==new)
        self.assertEqual(self.window.look_id,'original')
        self.assertEqual(self.window.canvas._image.pixelColor(0,0).red(),80)
    def test_failed_open_preserves_previous_document(self):
        self.load();self.window.open_path(str(self.path/'missing.png'))
        self.wait(lambda:not self.window.jobs)
        self.assertEqual(self.window.doc.path,self.source)
        self.assertEqual(self.window.canvas._image.pixelColor(0,0).red(),30)
        self.assertTrue(self.window.export_button.isEnabled())
    def test_existing_export_is_not_sent_to_worker(self):
        self.load();target=self.path/'existing.png';target.write_bytes(b'keep')
        self.window.start_export(target);self.app.processEvents()
        self.assertEqual(target.read_bytes(),b'keep');self.assertEqual(self.fake.exports,[])
    def test_all_looks_accept_fractional_strength_and_remember_it(self):
        self.load()
        for key in ['steve','eternal','vivid']:
            self.window.select_look(key);self.window.spin.setValue(37.25);self.window._start_preview()
            self.wait(lambda:self.window._ready and not self.window.jobs)
            self.assertEqual(self.window.slider.value(),3725)
            self.assertEqual(self.window._strength(),37.25)
        self.window.select_look('steve');self.assertEqual(self.window.spin.value(),37.25)
        self.window.select_look('original');self.assertFalse(self.window.slider.isVisible())
    def test_keyboard_fractional_entry_is_not_reformatted_mid_typing(self):
        self.load();self.window.select_look('eternal');self.window._start_preview()
        self.wait(lambda:self.window._ready and not self.window.jobs)
        self.window.spin.setFocus();self.window.spin.selectAll();QTest.keyClicks(self.window.spin,'37.25')
        QTest.keyClick(self.window.spin,Qt.Key.Key_Return)
        self.wait(lambda:self.window._ready and not self.window.jobs)
        self.assertEqual(self.window.spin.value(),37.25)
        self.assertEqual(self.window._strength(),37.25)
        self.assertEqual(self.window.slider.value(),3725)
    def test_fractional_export_uses_current_strength(self):
        self.load();self.window.select_look('eternal');self.window.spin.setValue(63.27);self.window._start_preview()
        self.wait(lambda:self.window._ready and not self.window.jobs)
        self.window.start_export(self.path/'fraction.png');self.wait(lambda:not self.window.jobs)
        self.assertEqual(self.fake.exports[-1][1:],('eternal',63.27))
    def test_info_survives_transient_hint_and_never_opens_modal(self):
        self.load();self.window.select_look('vivid');self.window._start_preview()
        self.wait(lambda:self.window._ready and not self.window.jobs)
        self.window.canvas.hint_timer.stop();self.window.canvas.hint.hide()
        self.window.filter_info_button.click();self.app.processEvents()
        self.assertTrue(self.window.info_panel.isVisible());self.assertTrue(self.window.slider.isVisible())
        self.assertIsNone(self.app.activeModalWidget())
        from apps.local_looks.look_info import LOOK_INFO, PREVIEW_NOTICE
        self.assertIn(LOOK_INFO['vivid']['body'],self.window.info_body.toPlainText())
        self.assertIn(PREVIEW_NOTICE,self.window.info_body.toPlainText())
        self.assertNotIn('本App操作说明',self.window.info_body.toPlainText())
        self.window.select_look('steve');self.assertEqual(self.window.info_title.text(),LOOK_INFO['steve'].get('official_name') or LOOK_INFO['steve']['title'])
        self.window._escape();self.assertFalse(self.window.info_panel.isVisible());self.assertEqual(self.window.look_id,'steve')
    def test_glass_does_not_resize_or_modify_the_displayed_photo(self):
        self.load();rect=self.window.canvas.image_rect();before=self.window.canvas._image.copy()
        self.window.set_filter_info_visible(True);self.app.processEvents()
        self.assertEqual(self.window.canvas.image_rect(),rect)
        self.assertEqual(self.window.canvas._image,before)
        blurred=self.window.info_panel._blurred_backdrop();self.assertFalse(blurred.isNull())
        first=self.window.info_panel._cache_key
        self.window.select_look('vivid');self.window._start_preview();self.wait(lambda:self.window._ready and not self.window.jobs)
        self.window.info_panel._blurred_backdrop();self.assertNotEqual(first,self.window.info_panel._cache_key)
        self.window.set_reduced_transparency(True);self.assertTrue(self.window.info_panel.reduced_transparency)
        self.assertTrue(self.window.canvas.hint.reduced_transparency)
    def test_transient_i_opens_persistent_description(self):
        self.load();self.window.select_look('eternal');self.window.canvas.hint_info.click();self.app.processEvents()
        self.assertTrue(self.window.info_panel.isVisible());self.assertFalse(self.window.canvas.hint.isVisible())
        self.assertEqual(self.window.info_title.text(),LOOK_INFO['eternal'].get('official_name') or LOOK_INFO['eternal']['title'])
    def test_canvas_space_is_temporary_comparison(self):
        self.load();self.window.select_look('vivid');self.window._start_preview()
        self.wait(lambda:self.window._ready and not self.window.jobs)
        self.window.canvas.setFocus();QTest.keyPress(self.window.canvas,Qt.Key.Key_Space)
        self.assertEqual(self.window.canvas._image.pixelColor(0,0).red(),30)
        QTest.keyRelease(self.window.canvas,Qt.Key.Key_Space)
        self.assertEqual(self.window.canvas._image.pixelColor(0,0).red(),36)

if __name__=='__main__':unittest.main(verbosity=2)
