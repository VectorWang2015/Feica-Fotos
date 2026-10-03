"""Single-window, offline Qt desktop frontend. No image processing on UI thread."""
from __future__ import annotations

import argparse
import itertools
from html import escape
import json
from pathlib import Path
import sys
import threading

import numpy as np
from PIL import Image, ImageOps
from PySide6.QtCore import Qt, QSize, QRectF, QTimer, QEvent, QObject, Signal, Slot, QRunnable, QThreadPool, QUrl
from PySide6.QtGui import QAction, QActionGroup, QFont, QIcon, QImage, QPixmap, QPainter, QColor, QColorSpace, QKeySequence, QDesktopServices
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLayout,
    QLabel, QPushButton, QToolButton, QButtonGroup, QSlider, QDoubleSpinBox, QTextBrowser, QProgressBar,
    QFileDialog, QMessageBox, QFrame, QSizePolicy, QTabBar, QScrollArea, QComboBox)

from .engine import (ImageEngine, LOOKS, LOOK_BY_ID, LOOK_GROUPS, LOOK_PREVIEW,
    LOOK_FILTER_OPTIONS, COLOR_FILTERS, look_preview, CancelledError)
from . import __version__
from .look_info import LOOK_INFO, PREVIEW_NOTICE
from .glass import GlassFrame
from .settings import application_settings, resolve_resource_dir

ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_RESOURCES = ROOT / 'filters/looks'

STYLE = """
QMainWindow, QWidget#workspace { background: #161616; color: #f3f3f3; }
QWidget#controls, QFrame#topbar { background: #242424; color: #f3f3f3; }
QLabel { color: #f3f3f3; background: transparent; }
QLabel[secondary="true"] { color: #b8b8b8; }
QPushButton, QToolButton { background: #303030; color: #f3f3f3; border: 1px solid #b8b8b8; border-radius: 6px; padding: 7px 14px; min-height: 20px; }
QPushButton:hover, QToolButton:hover { background: #414141; }
QPushButton:pressed, QToolButton:pressed { background: #555555; }
QPushButton:checked, QToolButton:checked { border: 2px solid #f3f3f3; background: #3b3b3b; }
QPushButton:focus, QToolButton:focus, QDoubleSpinBox:focus, QSlider:focus { border: 2px solid #9ec5ff; }
QPushButton#exportButton { background: #f3f3f3; color: #242424; border: 1px solid #f3f3f3; font-weight: 600; }
QPushButton:disabled, QToolButton:disabled { color: #8c8c8c; border-color: #606060; background: #242424; }
QPushButton#exportButton:disabled { background: #555555; color: #b8b8b8; border-color: #606060; }
QToolButton#lookCard { padding: 5px 8px; }
QScrollArea#lookRail, QWidget#lookRailContent { background: #242424; border: none; }
QTabBar::tab { background: #242424; color: #b8b8b8; padding: 5px 16px; border-bottom: 2px solid transparent; }
QTabBar::tab:selected { color: #f3f3f3; border-bottom: 2px solid #f3f3f3; }
QTabBar::tab:focus { border-bottom: 2px solid #9ec5ff; }
QComboBox { color: #f3f3f3; background: #303030; border: 1px solid #b8b8b8; border-radius: 5px; padding: 5px 8px; }
QComboBox QAbstractItemView { color: #f3f3f3; background: #303030; selection-background-color: #555555; }
QScrollBar:horizontal { background: transparent; height: 10px; margin: 0; }
QScrollBar::handle:horizontal { background: #b8b8b8; min-width: 24px; border-radius: 4px; }
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0; }
QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal { background: transparent; }
QSlider { min-height: 32px; border: 2px solid transparent; }
QSlider::groove:horizontal { background: #b8b8b8; height: 3px; border-radius: 1px; }
QSlider::handle:horizontal { background: #f3f3f3; width: 18px; margin: -8px 0; border-radius: 9px; }
QDoubleSpinBox { color: #f3f3f3; background: #303030; border: 1px solid #b8b8b8; border-radius: 5px; min-height: 32px; padding-left: 8px; }
QFrame#lookHint, QFrame#filterInfo { background: transparent; border: none; }
QFrame#lookHint QLabel, QFrame#filterInfo QLabel { color: #f3f3f3; }
QFrame#lookHint QToolButton, QFrame#filterInfo QToolButton { background: rgba(255,255,255,18); border: 1px solid rgba(255,255,255,105); color: #f3f3f3; }
QTextBrowser { color: #f3f3f3; background: transparent; border: none; }
QScrollBar:vertical { background: transparent; width: 8px; margin: 0; }
QScrollBar::handle:vertical { background: #b8b8b8; min-height: 24px; border-radius: 4px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: transparent; }
QProgressBar { background: #303030; border: none; max-height: 3px; }
QProgressBar::chunk { background: #b8b8b8; }
"""


# Keep workspace colors away from native/light file and message dialogs.
def _scope_workspace_styles(css):
    rules=[]
    for block in css.split('}'):
        if '{' not in block:continue
        selectors,body=block.split('{',1)
        scoped=[]
        for selector in selectors.split(','):
            selector=selector.strip()
            scoped.append(selector if selector in ('QMainWindow','QWidget#workspace') else 'QWidget#workspace '+selector)
        rules.append(', '.join(scoped)+' {'+body+'}')
    return '\n'.join(rules)

STYLE=_scope_workspace_styles(STYLE)


def application_icon() -> QIcon:
    icon=QIcon()
    folder=Path(__file__).resolve().parent/'assets'
    for size in (16,24,32,48,64,128,256,512):
        path=folder/f'icon-{size}.png'
        if path.is_file():icon.addFile(str(path),QSize(size,size))
    return icon


def image_from_rgb(rgb: np.ndarray) -> QImage:
    rgb = np.ascontiguousarray(rgb)
    h, w = rgb.shape[:2]
    image=QImage(rgb.data, w, h, rgb.strides[0], QImage.Format.Format_RGB888).copy()
    image.setColorSpace(QColorSpace(QColorSpace.NamedColorSpace.SRgb))
    return image


class PhotoCanvas(QWidget):
    dropped = Signal(str)
    invalid_drop = Signal()
    open_requested = Signal()
    compare_hold = Signal(bool)
    info_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._image = QImage()
        self._drag = False
        self.setAcceptDrops(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setAccessibleName('照片预览')
        self.setAccessibleDescription('等比显示完整照片。画布获得焦点后按住空格查看原图。')
        self.setMinimumSize(100, 120)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout = QVBoxLayout(self)
        layout.addStretch()
        self.empty = QWidget(self)
        inner = QVBoxLayout(self.empty)
        title = QLabel('打开一张照片'); font = title.font(); font.setPointSize(18); font.setWeight(QFont.Weight.DemiBold); title.setFont(font)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        desc = QLabel('拖入 JPG 或 DNG，选择一个 Look，再导出副本。')
        desc.setAlignment(Qt.AlignmentFlag.AlignCenter); desc.setWordWrap(True); desc.setProperty('secondary', True)
        button = QPushButton('打开照片…'); button.clicked.connect(self.open_requested)
        note = QLabel('全程离线 · DNG 使用内嵌 JPEG 预览')
        note.setAlignment(Qt.AlignmentFlag.AlignCenter); note.setWordWrap(True); note.setProperty('secondary', True)
        inner.addWidget(title); inner.addSpacing(8); inner.addWidget(desc); inner.addSpacing(16)
        row = QHBoxLayout(); row.addStretch(); row.addWidget(button); row.addStretch(); inner.addLayout(row)
        inner.addSpacing(8); inner.addWidget(note)
        layout.addWidget(self.empty); layout.addStretch()
        self.info_overlay=None
        self.hint=GlassFrame(self); self.hint.setObjectName('lookHint')
        hint_layout=QHBoxLayout(self.hint); hint_layout.setContentsMargins(12,8,12,8); hint_layout.setSpacing(12)
        text=QVBoxLayout(); text.setSpacing(2); self.hint_title=QLabel(); self.hint_detail=QLabel(); self.hint_detail.setProperty('secondary',True)
        self.hint_title.setTextFormat(Qt.TextFormat.PlainText)
        text.addWidget(self.hint_title); text.addWidget(self.hint_detail); hint_layout.addLayout(text)
        self.hint_info=QToolButton(); self.hint_info.setText('i'); self.hint_info.setAccessibleName('打开当前滤镜介绍'); self.hint_info.setToolTip('滤镜介绍')
        self.hint_info.clicked.connect(self.info_requested); hint_layout.addWidget(self.hint_info)
        self.hint.hide(); self.hint_timer=QTimer(self); self.hint_timer.setSingleShot(True); self.hint_timer.setInterval(2500); self.hint_timer.timeout.connect(self.hint.hide)
        for widget in [self.hint,self.hint_info]:widget.installEventFilter(self)

    def eventFilter(self,watched,event):
        if hasattr(self,'hint') and watched in (self.hint,self.hint_info):
            if event.type() in (QEvent.Type.Enter,QEvent.Type.FocusIn,QEvent.Type.MouseButtonPress):self.hint_timer.stop()
            elif event.type() in (QEvent.Type.Leave,QEvent.Type.FocusOut) and self.hint.isVisible():self.hint_timer.start()
        return super().eventFilter(watched,event)

    def show_look_hint(self,title,detail):
        if self._image.isNull():return
        self.hint_title.setText(title); self.hint_detail.setText(detail); self.hint.adjustSize()
        self._place_hint(); self.hint.show(); self.hint.raise_(); self.hint_timer.start()

    def _place_hint(self):
        self.hint.move(max(8,(self.width()-self.hint.width())//2),max(8,self.height()-self.hint.height()-24))

    def place_info_overlay(self):
        if self.info_overlay is None:return
        width=min(600,max(1,self.width()-32))
        height=min(240,max(1,self.height()-16))
        self.info_overlay.setGeometry((self.width()-width)//2,max(8,self.height()-height-8),width,height)
        self.info_overlay.invalidate_backdrop()

    def resizeEvent(self,event):
        super().resizeEvent(event)
        if hasattr(self,'hint'):self._place_hint(); self.hint.invalidate_backdrop()
        if hasattr(self,'info_overlay'):self.place_info_overlay()

    def image_rect(self):
        area=self.rect().adjusted(20,12,-20,-12)
        if self._image.isNull():return QRectF()
        size=self._image.size();size.scale(area.size(),Qt.AspectRatioMode.KeepAspectRatio)
        return QRectF(area.center().x()-size.width()/2,area.center().y()-size.height()/2,size.width(),size.height())

    def backdrop_image(self,area,ratio=1.0):
        image=QImage(round(area.width()*ratio),round(area.height()*ratio),QImage.Format.Format_RGBA8888);image.setDevicePixelRatio(ratio);image.fill(QColor('#161616'))
        p=QPainter(image);p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        p.translate(-area.x(),-area.y())
        if not self._image.isNull():p.drawImage(self.image_rect(),self._image)
        p.end();return image

    def set_image(self, image):
        self._image = image
        self.empty.setVisible(image.isNull())
        self.hint.invalidate_backdrop()
        if self.info_overlay is not None:self.info_overlay.invalidate_backdrop()
        self.update()

    def paintEvent(self, event):
        p = QPainter(self); p.fillRect(self.rect(), QColor('#161616'))
        if not self._image.isNull():
            p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
            p.drawImage(self.image_rect(), self._image)
        if self._drag or self.hasFocus():
            p.setPen(QColor('#9ec5ff') if self.hasFocus() else QColor('#f3f3f3'))
            p.drawRect(self.rect().adjusted(2, 2, -3, -3))

    def mousePressEvent(self, event):
        self.setFocus(); super().mousePressEvent(event)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            self.compare_hold.emit(True); event.accept(); return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            self.compare_hold.emit(False); event.accept(); return
        super().keyReleaseEvent(event)

    def focusOutEvent(self, event):
        self.compare_hold.emit(False); self.update(); super().focusOutEvent(event)

    def focusInEvent(self, event):
        self.update(); super().focusInEvent(event)

    def dragEnterEvent(self, event):
        urls = event.mimeData().urls()
        if len(urls) == 1 and urls[0].isLocalFile() and Path(urls[0].toLocalFile()).is_file():
            self._drag = True; self.update(); event.acceptProposedAction()
        else:
            event.ignore()

    def dragLeaveEvent(self, event):
        self._drag = False; self.update()

    def dropEvent(self, event):
        self._drag = False; self.update()
        urls = event.mimeData().urls()
        if len(urls) == 1 and urls[0].isLocalFile() and Path(urls[0].toLocalFile()).is_file():
            self.dropped.emit(urls[0].toLocalFile()); event.acceptProposedAction()
        else:
            self.invalid_drop.emit(); event.ignore()


class LookRail(QScrollArea):
    """A single row: wheel/trackpad browsing never changes the applied Look."""
    def wheelEvent(self,event):
        delta=event.pixelDelta()
        amount=delta.x() or delta.y()
        if not amount:
            delta=event.angleDelta(); amount=(delta.x() or delta.y())//2
        bar=self.horizontalScrollBar()
        if amount and bar.maximum()>0:
            bar.setValue(bar.value()-amount); event.accept()
        else:super().wheelEvent(event)


class JobSignals(QObject):
    result = Signal(int, object)
    item = Signal(int, object)
    failed = Signal(int, str, bool)
    progress = Signal(int, float)
    finished = Signal(int)


class Job(QRunnable):
    def __init__(self, token, kind, revision, function, *, stream=False):
        super().__init__()
        self.token, self.kind, self.revision, self.function = token, kind, revision, function
        self.stream = stream
        self.cancel = threading.Event()
        self.signals = JobSignals()

    @Slot()
    def run(self):
        try:
            if self.cancel.is_set(): raise CancelledError('已取消')
            progress = lambda p: self.signals.progress.emit(self.token, float(p))
            if self.stream:
                result = self.function(self.cancel, progress, lambda item: self.signals.item.emit(self.token, item))
            else:
                result = self.function(self.cancel, progress)
            if self.cancel.is_set() and self.kind != 'export': raise CancelledError('已取消')
            self.signals.result.emit(self.token, result)
        except Exception as exc:
            self.signals.failed.emit(self.token, str(exc), isinstance(exc, CancelledError))
        finally:
            self.signals.finished.emit(self.token)


class MainWindow(QMainWindow):
    def __init__(self, resource_dir=None, *, settings=None):
        super().__init__()
        self.settings = settings if settings is not None else application_settings()
        bundled = ROOT / 'look-resources'
        default_resources = bundled if bundled.is_dir() else WORKSPACE_RESOURCES
        resource_dir = resolve_resource_dir(resource_dir, self.settings, default_resources)
        self.engine = ImageEngine(Path(resource_dir))
        self.resource_dir = Path(resource_dir)
        self.pool = QThreadPool(self); self.pool.setMaxThreadCount(2)
        self.jobs = {}; self.tokens = itertools.count(1)
        self.doc = None; self.doc_revision = 0; self.render_revision = 0
        self.look_id = 'original'; self.strengths = {spec.id:float(spec.default_strength) for spec in LOOKS}
        self.color_filter = None; self.browse_group = 'all'
        self._thumbnail_token = None; self._thumbnail_queue = None; self._thumbnail_cache = {}
        self.source_image = QImage(); self.effect_image = QImage()
        self._held_compare = False; self._loading = False; self._exporting = False; self._ready = False
        self._preview_waiting = False; self._preview_token = None; self._load_token = None
        self._last_export_signature = None; self._after_export = None; self._closing = False; self._discard_close = False
        self.setWindowTitle('Feica Fotos'); self.setWindowIcon(application_icon()); self.setMinimumSize(720, 540); self.resize(1180, 780)
        self.setStyleSheet(STYLE)
        self._build_ui(); self._build_menu()
        screen=QApplication.primaryScreen()
        if screen:
            available=screen.availableGeometry()
            self.resize(min(1180,max(720,available.width()-32)),min(780,max(540,available.height()-64)))
            saved=self.settings.value('window_geometry')
            if saved:self.restoreGeometry(saved)
            if not saved or not available.intersects(self.frameGeometry()):self.move(available.center()-self.rect().center())
        self.timer = QTimer(self); self.timer.setSingleShot(True); self.timer.setInterval(80); self.timer.timeout.connect(self._start_preview)
        self._update_controls()

    def _build_ui(self):
        root = QWidget(); root.setObjectName('workspace'); self.setCentralWidget(root)
        layout = QVBoxLayout(root); layout.setContentsMargins(0,0,0,0); layout.setSpacing(0)
        top = QFrame(); top.setObjectName('topbar'); row = QHBoxLayout(top); row.setContentsMargins(20,8,20,8); row.setSpacing(8)
        self.open_button = QPushButton('打开照片…'); self.open_button.clicked.connect(self.choose_open)
        self.file_label = QLabel('Feica Fotos'); self.file_label.setSizePolicy(QSizePolicy.Policy.Ignored,QSizePolicy.Policy.Preferred)
        self.compare_button = QPushButton('查看原图'); self.compare_button.setCheckable(True); self.compare_button.toggled.connect(self._show_image)
        self.compare_button.setToolTip('B：切换原图；点击画布后按住空格临时对照')
        self.export_button = QPushButton('导出副本…'); self.export_button.setObjectName('exportButton'); self.export_button.clicked.connect(self.choose_export)
        for widget in [self.open_button,self.file_label,self.compare_button,self.export_button]: row.addWidget(widget,1 if widget is self.file_label else 0)
        layout.addWidget(top)
        context = QHBoxLayout(); context.setContentsMargins(20,6,20,6); context.setSpacing(12)
        self.source_label = QLabel('离线照片滤镜'); self.source_label.setProperty('secondary',True)
        self.status_label = QLabel(''); self.status_label.setWordWrap(True); self.status_label.setProperty('secondary',True)
        self.cancel_button = QPushButton('取消'); self.cancel_button.clicked.connect(self.cancel_work); self.cancel_button.hide()
        self.folder_button = QPushButton('打开所在文件夹'); self.folder_button.hide(); self.folder_button.clicked.connect(self.open_export_folder)
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        context.addWidget(self.source_label); context.addWidget(self.status_label,1); context.addWidget(self.cancel_button); context.addWidget(self.folder_button)
        layout.addLayout(context)
        self.progress_bar = QProgressBar(); self.progress_bar.setTextVisible(False); self.progress_bar.hide(); layout.addWidget(self.progress_bar)
        self.canvas = PhotoCanvas(); self.canvas.dropped.connect(self.open_path); self.canvas.open_requested.connect(self.choose_open)
        self.canvas.invalid_drop.connect(lambda:self._message('请一次打开一张 JPG 或 DNG 照片。',True))
        self.canvas.compare_hold.connect(self._hold_compare); self.canvas.info_requested.connect(lambda:self.set_filter_info_visible(True)); layout.addWidget(self.canvas,1)
        self.info_panel=GlassFrame(self.canvas,radius=14); self.canvas.info_overlay=self.info_panel; self.info_panel.setObjectName('filterInfo'); info_layout=QVBoxLayout(self.info_panel); info_layout.setContentsMargins(20,12,20,12); info_layout.setSpacing(8)
        info_header=QHBoxLayout(); self.info_title=QLabel(); self.info_title.setTextFormat(Qt.TextFormat.PlainText); info_header.addWidget(self.info_title); info_header.addStretch()
        close_info=QToolButton(); close_info.setText('关闭 ×'); close_info.setAccessibleName('关闭滤镜介绍'); close_info.clicked.connect(lambda:self.set_filter_info_visible(False)); info_header.addWidget(close_info)
        info_layout.addLayout(info_header)
        self.info_body=QTextBrowser(); self.info_body.setOpenExternalLinks(False); self.info_body.setOpenLinks(False); self.info_body.setMinimumHeight(0)
        self.info_body.setAutoFillBackground(False); self.info_body.viewport().setAutoFillBackground(False)
        self.info_body.setAccessibleName('滤镜介绍内容'); info_layout.addWidget(self.info_body,1)
        self.info_panel.hide()
        controls = QWidget(); controls.setObjectName('controls'); bottom=QVBoxLayout(controls); bottom.setContentsMargins(20,12,20,12); bottom.setSpacing(8)
        self.group_tabs=QTabBar(); self.group_tabs.setAccessibleName('Look 分组'); self.group_tabs.setExpanding(False)
        self.group_tabs.setDrawBase(False)
        for key,title in [('all','全部'),('color','颜色'),('monochrome','单色'),('artist','Artist')]:
            index=self.group_tabs.addTab(title); self.group_tabs.setTabData(index,key)
        self.group_tabs.currentChanged.connect(self._group_changed); bottom.addWidget(self.group_tabs,0,Qt.AlignmentFlag.AlignHCenter)
        self.look_rail=LookRail(); self.look_rail.setObjectName('lookRail'); self.look_rail.setAccessibleName('Look 选择列表')
        self.look_rail.setWidgetResizable(True); self.look_rail.setFrameShape(QFrame.Shape.NoFrame)
        self.look_rail.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self.look_rail.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.look_rail.setSizePolicy(QSizePolicy.Policy.Ignored,QSizePolicy.Policy.Fixed)
        self.card_box=QWidget(); self.card_box.setObjectName('lookRailContent')
        self.cards_layout=QHBoxLayout(self.card_box); self.cards_layout.setContentsMargins(0,0,0,2); self.cards_layout.setSpacing(12)
        self.cards_layout.setSizeConstraint(QLayout.SizeConstraint.SetMinimumSize)
        self.look_buttons={}; self.look_group=QButtonGroup(self); self.look_group.setExclusive(True)
        for spec in LOOKS:
            button=QToolButton(); button.setObjectName('lookCard'); button.setText(spec.title); button.setCheckable(True)
            button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon); button.setIconSize(QSize(120,44))
            button.setSizePolicy(QSizePolicy.Policy.Fixed,QSizePolicy.Policy.Fixed); button.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
            button.setAccessibleName(spec.title); button.setToolTip(spec.title)
            button.clicked.connect(lambda checked=False,key=spec.id:self.select_look(key))
            button.setProperty('lookTitle',spec.title); button.setProperty('lookId',spec.id); button.installEventFilter(self)
            self.look_group.addButton(button); self.look_buttons[spec.id]=button; self.cards_layout.addWidget(button)
        self.cards_layout.addStretch(); self.look_rail.setWidget(self.card_box); bottom.addWidget(self.look_rail)
        self._update_rail_metrics()
        param=QHBoxLayout(); param.addStretch(); self.strength_label=QLabel('强度'); param.addWidget(self.strength_label)
        self.slider=QSlider(Qt.Orientation.Horizontal); self.slider.setRange(0,10000); self.slider.setValue(5000); self.slider.setSingleStep(1); self.slider.setPageStep(100); self.slider.setMinimumWidth(180); self.slider.setMaximumWidth(330)
        self.slider.setAccessibleName('滤镜强度，0至100，百分之一精度'); self.slider.valueChanged.connect(lambda value:self._strength_changed(value/100)); self.slider.sliderReleased.connect(self._start_preview)
        self.spin=QDoubleSpinBox(); self.spin.setRange(0,100); self.spin.setDecimals(2); self.spin.setSingleStep(.01); self.spin.setKeyboardTracking(False); self.spin.setValue(50); self.spin.setAccessibleName('滤镜强度数值'); self.spin.valueChanged.connect(self._strength_changed)
        self.strength_label.setBuddy(self.spin); param.addWidget(self.slider); param.addWidget(self.spin)
        self.parameter_note=QLabel('未应用滤镜'); self.parameter_note.setProperty('secondary',True); self.parameter_note.setWordWrap(True)
        param.addWidget(self.parameter_note)
        self.color_filter_label=QLabel('滤色'); self.color_filter_combo=QComboBox()
        self.color_filter_combo.setAccessibleName('滤色'); self.color_filter_label.setBuddy(self.color_filter_combo)
        self.color_filter_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.color_filter_combo.currentIndexChanged.connect(self._color_filter_changed)
        param.addWidget(self.color_filter_label); param.addWidget(self.color_filter_combo)
        self._sync_color_filter_widgets()
        self.filter_info_button=QToolButton(); self.filter_info_button.setText('i  滤镜介绍'); self.filter_info_button.setCheckable(True); self.filter_info_button.setAccessibleName('滤镜介绍'); self.filter_info_button.setToolTip('Ctrl+I：展开或收起当前滤镜介绍')
        self.filter_info_button.toggled.connect(self.set_filter_info_visible); param.addWidget(self.filter_info_button)
        param.addStretch(); bottom.addLayout(param); layout.addWidget(controls)

    def _action(self, menu, text, callback, shortcut=None, checkable=False):
        action=QAction(text,self); action.setCheckable(checkable)
        if shortcut:action.setShortcut(QKeySequence(shortcut))
        action.triggered.connect(callback); menu.addAction(action); return action

    def _build_menu(self):
        file=self.menuBar().addMenu('文件')
        self.open_action=self._action(file,'打开照片…',self.choose_open,'Ctrl+O')
        self.export_action=self._action(file,'导出副本…',self.choose_export,'Ctrl+Shift+S')
        self.info_action=self._action(file,'照片信息',self.photo_info); file.addSeparator()
        self._action(file,'滤镜资源文件夹…',self.choose_resources)
        self._action(file,'退出',self.close,'Ctrl+Q')
        view=self.menuBar().addMenu('查看')
        self.compare_action=self._action(view,'查看原图',lambda checked:self.compare_button.setChecked(checked),'B',True)
        self.compare_button.toggled.connect(self.compare_action.setChecked)
        self._action(view,'关闭介绍／取消当前处理',self._escape,'Esc')
        self.reduce_transparency_action=self._action(view,'减少透明效果',self.set_reduced_transparency,None,True)
        reduced=self.settings.value('reduce_transparency',False,type=bool); self.reduce_transparency_action.setChecked(reduced); self.set_reduced_transparency(reduced)
        menu=self.menuBar().addMenu('滤镜'); self.look_actions={}; group=QActionGroup(self); group.setExclusive(True)
        shortcuts={'original':'Ctrl+1','steve':'Ctrl+2','eternal':'Ctrl+3','vivid':'Ctrl+4'}
        for spec in LOOKS:
            name=self.look_buttons[spec.id].text()
            a=self._action(menu,name,lambda checked=False,key=spec.id:self.select_look(key),shortcuts.get(spec.id),True)
            group.addAction(a); self.look_actions[spec.id]=a
        menu.addSeparator(); self.filter_info_action=self._action(menu,'滤镜介绍',self.set_filter_info_visible,'Ctrl+I',True)
        help_menu=self.menuBar().addMenu('帮助'); self._action(help_menu,'关于 Feica Fotos',self.about)

    def _visible_look_ids(self):
        return [spec.id for spec in LOOKS if spec.id=='original' or self.browse_group=='all' or LOOK_GROUPS[spec.id]==self.browse_group]

    @Slot(int)
    def _group_changed(self,index):
        self.browse_group=self.group_tabs.tabData(index)
        visible=set(self._visible_look_ids())
        for key,button in self.look_buttons.items():button.setVisible(key in visible)
        self.cards_layout.invalidate(); self.cards_layout.activate()
        self.look_rail.horizontalScrollBar().setValue(0)
        self._prioritize_thumbnails()
        # Browsing is not applying: leave look_id, strength and render state alone.

    def _reveal_look(self,key):
        if key not in self._visible_look_ids():
            group=LOOK_GROUPS[key]
            for index in range(self.group_tabs.count()):
                if self.group_tabs.tabData(index)==group:
                    self.group_tabs.setCurrentIndex(index); break
        self.cards_layout.activate()
        self.look_rail.ensureWidgetVisible(self.look_buttons[key],16,0)
        QTimer.singleShot(0,self._ensure_selected_look_visible)

    def _ensure_selected_look_visible(self):
        if not self._closing and self.look_id in self._visible_look_ids():
            self.look_rail.ensureWidgetVisible(self.look_buttons[self.look_id],16,0)

    def _update_rail_metrics(self):
        compact=self.width()<960
        height=0
        for button in self.look_buttons.values():
            button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly if compact else Qt.ToolButtonStyle.ToolButtonTextUnderIcon)
            button.setMinimumWidth(max(0 if compact else 136,button.fontMetrics().horizontalAdvance(button.text())+32))
            button.setMinimumHeight(button.fontMetrics().height()+(16 if compact else 60))
            height=max(height,button.sizeHint().height(),button.minimumHeight())
        self.look_rail.setFixedHeight(height+self.look_rail.horizontalScrollBar().sizeHint().height()+6)

    def eventFilter(self,watched,event):
        key=watched.property('lookId')
        if key and hasattr(self,'look_buttons'):
            if event.type()==QEvent.Type.FocusIn:
                self.look_rail.ensureWidgetVisible(watched,16,0)
            elif event.type()==QEvent.Type.KeyPress and not event.modifiers():
                keys=self._visible_look_ids()
                if event.key() in (Qt.Key.Key_Left,Qt.Key.Key_Right,Qt.Key.Key_Home,Qt.Key.Key_End):
                    index=keys.index(key)
                    if event.key()==Qt.Key.Key_Home:index=0
                    elif event.key()==Qt.Key.Key_End:index=len(keys)-1
                    else:index=max(0,min(len(keys)-1,index+(-1 if event.key()==Qt.Key.Key_Left else 1)))
                    target=keys[index]; self.look_buttons[target].setFocus(); self.select_look(target)
                    return True
        return super().eventFilter(watched,event)

    def _filter_options(self):
        return LOOK_FILTER_OPTIONS[self.look_id] if LOOK_GROUPS[self.look_id]=='monochrome' else ()

    def _sync_color_filter_widgets(self):
        options=self._filter_options()
        if self.color_filter not in options:self.color_filter=None
        self.color_filter_combo.blockSignals(True); self.color_filter_combo.clear()
        self.color_filter_combo.addItem('无',None)
        names={'red':'红','orange':'橙','yellow':'黄','green':'绿','blue':'蓝'}
        for key in options:
            item=COLOR_FILTERS[key]
            label=item.get('label') or names.get(key) or item.get('title') or key
            self.color_filter_combo.addItem(label,key)
        self.color_filter_combo.setCurrentIndex(max(0,self.color_filter_combo.findData(self.color_filter)))
        self.color_filter_combo.blockSignals(False)
        for widget in (self.color_filter_label,self.color_filter_combo):widget.setVisible(bool(options))

    @Slot(int)
    def _color_filter_changed(self,index):
        if self.doc is None or self._loading or self._exporting:return
        value=self.color_filter_combo.itemData(index)
        if value is not None and value not in self._filter_options():return
        if value==self.color_filter:return
        self.color_filter=value; self.render_revision+=1; self._ready=False
        self._update_info_content(); self._message('正在更新预览…'); self._update_controls(); self.timer.start()

    def _message(self,text,error=False):
        self.status_label.setText(text); self.status_label.setStyleSheet('color: #ffb4ab;' if error else 'color: #b8b8b8;')
        self.folder_button.hide()

    def _strength(self):
        return self.strengths[self.look_id]

    def _sync_strength_widgets(self):
        value=self._strength()
        self.slider.blockSignals(True); self.spin.blockSignals(True)
        self.slider.setValue(round(value*100)); self.spin.setValue(value)
        self.slider.blockSignals(False); self.spin.blockSignals(False)

    def _signature(self):
        return (self.doc_revision,self.look_id,self._strength(),self.color_filter)

    def _is_dirty(self):
        return self.doc is not None and self.look_id!='original' and self._signature()!=self._last_export_signature

    def _confirm_transition(self,callback):
        if not self._is_dirty():return True
        box=QMessageBox(self); box.setWindowTitle('尚未导出'); box.setText('当前效果尚未导出。')
        stay=box.addButton('继续编辑',QMessageBox.ButtonRole.RejectRole)
        discard=box.addButton('不导出',QMessageBox.ButtonRole.DestructiveRole)
        save=box.addButton('导出副本…',QMessageBox.ButtonRole.AcceptRole); box.setDefaultButton(stay); box.exec()
        if box.clickedButton() is discard:return True
        if box.clickedButton() is save:self._after_export=callback; self.choose_export()
        return False

    @Slot()
    def choose_open(self):
        if self._exporting:return
        path,_=QFileDialog.getOpenFileName(self,'打开照片',self.settings.value('last_dir',str(Path.home())),'照片 (*.jpg *.jpeg *.JPG *.JPEG *.dng *.DNG *.png *.PNG)')
        if path:self.open_path(path)

    @Slot(str)
    def open_path(self,path,skip_confirm=False):
        if self._exporting:return
        path=str(path)
        if not skip_confirm and not self._confirm_transition(lambda:self.open_path(path,True)):return
        self.timer.stop(); self._preview_waiting=False
        for job in self.jobs.values():
            if job.kind!='export':job.cancel.set()
        self._loading=True; self._message('正在读取照片…'); self.progress_bar.setRange(0,0); self.progress_bar.show()
        self._load_token=self._submit('load',self.doc_revision,lambda cancel,progress:self.engine.load_document(path))
        self._update_controls()

    def _submit(self,kind,revision,function,*,stream=False,priority=0):
        token=next(self.tokens); job=Job(token,kind,revision,function,stream=stream); self.jobs[token]=job
        job.signals.result.connect(self._job_result); job.signals.failed.connect(self._job_failed)
        job.signals.item.connect(self._job_item)
        job.signals.progress.connect(self._job_progress); job.signals.finished.connect(self._job_finished)
        self.pool.start(job,priority); return token

    def select_look(self,key):
        if self.doc is None or self._loading or self._exporting:return
        self.look_id=key; self.render_revision+=1; self._ready=False; self.compare_button.setChecked(False)
        self._sync_strength_widgets(); self._sync_color_filter_widgets(); self._update_info_content(); self._reveal_look(key)
        if not self.info_panel.isVisible():
            self.canvas.show_look_hint(LOOK_INFO[key]['title'],'未应用滤镜' if key=='original' else f'强度 {self._strength():.2f}')
        else:
            self.canvas.hint.hide(); self.canvas.hint_timer.stop()
        self._message('正在更新预览…'); self._update_controls(); self.timer.start()

    def _strength_changed(self,value):
        if self.doc is None or not LOOK_BY_ID[self.look_id].adjustable or self._exporting or self._loading:return
        self.strengths[self.look_id]=round(float(value),2)
        if self.canvas.hint.isVisible():self.canvas.hint_detail.setText(f'强度 {self._strength():.2f}')
        self._sync_strength_widgets()
        self.render_revision+=1; self._ready=False; self._message('正在更新预览…'); self._update_controls(); self.timer.start()

    def _start_preview(self):
        self.timer.stop()
        if self.doc is None or self._loading or self._exporting:return
        if self._preview_token in self.jobs:
            self.jobs[self._preview_token].cancel.set(); self._preview_waiting=True; return
        self._preview_waiting=False
        if self.look_id=='original':
            self.effect_image=self.source_image; self._ready=True; self._message('未应用滤镜'); self._show_image(); self._update_controls(); return
        rgb=self.doc.preview_rgb; key=self.look_id; strength=self._strength(); color_filter=self.color_filter; engine=self.engine
        self._preview_token=self._submit('preview',self.render_revision,lambda cancel,progress:engine.render(rgb,key,strength,color_filter=color_filter,progress=progress,cancel=cancel))
        self._update_controls()

    def _prioritize_thumbnails(self):
        if self._thumbnail_queue is None:return
        pending,lock=self._thumbnail_queue
        visible=set(self._visible_look_ids())
        with lock:pending.sort(key=lambda key:key not in visible)

    def _start_thumbnails(self):
        if self.doc is None:return
        for job in self.jobs.values():
            if job.kind=='thumbs':job.cancel.set()
        source=self.doc.preview_rgb; engine=self.engine
        self._thumbnail_cache={}
        for button in self.look_buttons.values():button.setIcon(QIcon())
        pending=[spec.id for spec in LOOKS]; lock=threading.Lock()
        self._thumbnail_queue=(pending,lock); self._prioritize_thumbnails()
        def work(cancel,progress,emit):
            proxy=Image.fromarray(source); proxy.thumbnail((240,240),Image.Resampling.LANCZOS)
            rgb=np.asarray(proxy)
            while True:
                if cancel.is_set():raise CancelledError('已取消')
                with lock:
                    if not pending:break
                    key=pending.pop(0)
                spec=LOOK_BY_ID[key]
                try:
                    rendered=rgb if key=='original' else engine.render(rgb,key,spec.default_strength,color_filter=None,cancel=cancel)
                    thumb=np.asarray(ImageOps.fit(Image.fromarray(rendered),(240,88),Image.Resampling.LANCZOS)).copy()
                except CancelledError:raise
                except Exception:thumb=None
                if cancel.is_set():raise CancelledError('已取消')
                emit((key,thumb))
        self._thumbnail_token=self._submit('thumbs',self.doc_revision,work,stream=True,priority=-1)

    @Slot(int,object)
    def _job_item(self,token,item):
        job=self.jobs.get(token)
        if (job is None or job.kind!='thumbs' or token!=self._thumbnail_token or job.cancel.is_set()
                or job.revision!=self.doc_revision or self._loading or self._closing):return
        key,rgb=item
        icon=QIcon() if rgb is None else QIcon(QPixmap.fromImage(image_from_rgb(rgb)))
        self._thumbnail_cache[key]=icon; self.look_buttons[key].setIcon(icon)

    @Slot(int,object)
    def _job_result(self,token,result):
        job=self.jobs.get(token)
        if job is None:return
        if job.kind!='export' and job.cancel.is_set():return
        if job.kind=='load' and token==self._load_token:
            self._loading=False; self.doc=result; self.doc_revision+=1; self.render_revision+=1
            self.look_id='original'; self.color_filter=None; self.strengths={spec.id:float(spec.default_strength) for spec in LOOKS}; self._last_export_signature=None; self._ready=True
            self._sync_strength_widgets(); self._sync_color_filter_widgets(); self.set_filter_info_visible(False); self.canvas.hint.hide(); self._update_info_content(); self._reveal_look('original')
            self.source_image=image_from_rgb(result.preview_rgb); self.effect_image=self.source_image
            self.compare_button.setChecked(False); self.file_label.setText(result.path.name); self.file_label.setToolTip(str(result.path))
            self.setWindowTitle(f'{result.path.name} — Feica Fotos')
            self.source_label.setText(f'{"DNG预览" if result.source_kind=="DNG preview" else "照片"} · {result.width} × {result.height}')
            self.settings.setValue('last_dir',str(result.path.parent))
            for button in self.look_buttons.values():button.setIcon(QIcon())
            self._message('未应用滤镜'); self.progress_bar.hide(); self._show_image(); self._update_controls(); self.canvas.setFocus(); self._start_thumbnails()
        elif job.kind=='preview' and job.revision==self.render_revision and not self._loading:
            self.effect_image=image_from_rgb(result); self._ready=True
            self._message(f'{LOOK_INFO[self.look_id]["title"]} · {self._strength():.2f}')
            self._show_image(); self._update_controls()
        elif job.kind=='export':
            self._exporting=False; self._last_export_signature=job.revision; self.progress_bar.hide()
            self._message(f'已导出：{self._export_path.name}'); self.folder_button.show(); self._update_controls()
            callback=self._after_export; self._after_export=None
            if callback:QTimer.singleShot(0,callback)

    @Slot(int,str,bool)
    def _job_failed(self,token,message,cancelled):
        job=self.jobs.get(token)
        if job is None:return
        relevant=job.kind=='export' or (job.kind=='load' and token==self._load_token) or (job.kind=='preview' and job.revision==self.render_revision and not self._loading)
        if not relevant:return
        if job.kind=='load':
            self._loading=False
            if self.doc is not None and not self._ready and not self._closing:self.timer.start()
        if job.kind=='export':self._exporting=False; self._after_export=None
        if job.kind=='preview':self._ready=False
        self.progress_bar.hide(); self._message('已取消' if cancelled else message,not cancelled); self._update_controls()

    @Slot(int,float)
    def _job_progress(self,token,value):
        job=self.jobs.get(token)
        if job is not None and job.kind=='export':
            self.progress_bar.setRange(0,100); self.progress_bar.setValue(max(0,min(100,round(value*100))))
            self._message(f'正在导出… {round(value*100)}%')

    @Slot(int)
    def _job_finished(self,token):
        job=self.jobs.pop(token,None)
        if job and job.kind=='preview':
            self._preview_token=None
            if self._preview_waiting and not self._loading and not self._closing:QTimer.singleShot(0,self._start_preview)
        if self._closing and not self.jobs:QTimer.singleShot(0,self.close)
        self._update_controls()

    def _update_controls(self):
        enabled=self.doc is not None and not self._loading and not self._exporting
        self.open_button.setEnabled(not self._exporting); self.open_action.setEnabled(not self._exporting) if hasattr(self,'open_action') else None
        self.compare_button.setEnabled(self.doc is not None)
        self.export_button.setEnabled(enabled and self._ready)
        if hasattr(self,'export_action'):
            self.export_action.setEnabled(enabled and self._ready); self.info_action.setEnabled(self.doc is not None); self.compare_action.setEnabled(self.doc is not None)
        for key,button in self.look_buttons.items():
            button.setEnabled(enabled); button.setChecked(key==self.look_id)
            if hasattr(self,'look_actions'):self.look_actions[key].setEnabled(enabled); self.look_actions[key].setChecked(key==self.look_id)
        adjustable=LOOK_BY_ID[self.look_id].adjustable
        for widget in [self.strength_label,self.slider,self.spin]:widget.setVisible(adjustable); widget.setEnabled(enabled)
        for widget in [self.color_filter_label,self.color_filter_combo]:widget.setEnabled(enabled)
        self.parameter_note.setText('未应用滤镜' if self.look_id=='original' else '')
        self.filter_info_button.setEnabled(self.doc is not None)
        self.filter_info_action.setEnabled(self.doc is not None)
        self.cancel_button.setVisible(self._loading or self._exporting or (not self._ready and self.doc is not None))

    @Slot()
    def _show_image(self,*args):
        compare=self.compare_button.isChecked() or self._held_compare
        self.canvas.set_image(self.source_image if compare else self.effect_image)

    @Slot(bool)
    def _hold_compare(self,held):self._held_compare=held; self._show_image()

    @Slot()
    def cancel_work(self):
        self.timer.stop(); self._preview_waiting=False; self._after_export=None
        for job in self.jobs.values():job.cancel.set()
        self._load_token=None; self._loading=False
        self.canvas.hint.hide(); self.canvas.hint_timer.stop()
        if self.doc is not None and not self._exporting:
            self.look_id='original'; self.color_filter=None; self.render_revision+=1; self.effect_image=self.source_image; self._ready=True
            self._sync_strength_widgets(); self._sync_color_filter_widgets(); self._update_info_content(); self._show_image(); self._reveal_look('original')
        if not self._exporting:self.progress_bar.hide()
        self._message('正在取消…' if self._exporting else '已取消'); self._update_controls()

    @Slot()
    def choose_export(self):
        if self.doc is None or not self._ready or self._loading or self._exporting:
            self._after_export=None; return
        suffix='original' if self.look_id=='original' else self.look_id+'-'+f'{self._strength():.2f}'.rstrip('0').rstrip('.').replace('.','p')
        if self.color_filter is not None:suffix+='-'+self.color_filter
        default=Path(self.settings.value('export_dir',str(self.doc.path.parent)))/f'{self.doc.path.stem}-{suffix}.jpg'
        path,chosen=QFileDialog.getSaveFileName(self,'导出副本',str(default),'JPEG 图像 (*.jpg *.jpeg);;PNG 图像 (*.png)',options=QFileDialog.Option.DontConfirmOverwrite)
        if not path:self._after_export=None; return
        target=Path(path)
        if not target.suffix:target=target.with_suffix('.png' if chosen.startswith('PNG') else '.jpg')
        self.start_export(target)

    def start_export(self,target):
        if self.doc is None or not self._ready or self._exporting:return
        target=Path(target)
        if target.exists() or target.is_symlink():self._message('目标文件已存在，请换一个文件名。原文件不会被覆盖。',True); self._after_export=None; return
        self._exporting=True; self._export_path=target; doc=self.doc; key=self.look_id; strength=self._strength(); color_filter=self.color_filter; engine=self.engine
        self.settings.setValue('export_dir',str(target.parent)); self.progress_bar.setRange(0,100); self.progress_bar.setValue(0); self.progress_bar.show()
        self._message('正在导出…'); self._update_controls()
        self._submit('export',self._signature(),lambda cancel,progress:engine.export(doc,target,key,strength,color_filter=color_filter,quality=95,progress=progress,cancel=cancel))

    def open_export_folder(self):
        if hasattr(self,'_export_path'):QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._export_path.parent.resolve())))

    def choose_resources(self):
        if self._loading or self._exporting:return
        path=QFileDialog.getExistingDirectory(self,'选择本地滤镜资源文件夹',str(self.resource_dir))
        if not path:return
        for job in self.jobs.values():job.cancel.set()
        self.resource_dir=Path(path); self.engine=ImageEngine(self.resource_dir); self.settings.setValue('resource_dir',path)
        self._message('已设置本地滤镜资源位置。')
        if self.doc is not None:self.render_revision+=1; self._ready=False; self.timer.start(); self._start_thumbnails()

    def _update_info_content(self):
        entry=LOOK_INFO[self.look_id]
        self.info_title.setText(entry.get('official_name') or entry['title'])
        body=escape(entry['body']).replace('\n','<br>')
        content='<p style="margin:0;white-space:pre-wrap">'+body+'</p>' if body else ''
        preview=entry.get('preview',LOOK_PREVIEW[self.look_id])
        if self.color_filter is not None:
            preview=preview or look_preview(self.look_id,strength=self._strength(),color_filter=self.color_filter)
        if preview:
            content+='<p style="margin:12px 0 0;font-style:italic">'+escape(PREVIEW_NOTICE)+'</p>'
        self.info_body.setHtml(content)
        self.info_body.setAccessibleName(entry['title']+' 滤镜介绍')
        self.info_body.verticalScrollBar().setValue(0)

    @Slot(bool)
    def set_filter_info_visible(self,visible):
        visible=bool(visible) and self.doc is not None
        self.canvas.hint.hide(); self.canvas.hint_timer.stop()
        self._update_info_content(); self.canvas.place_info_overlay(); self.info_panel.setVisible(visible)
        if visible:self.info_panel.raise_()
        self.filter_info_button.blockSignals(True); self.filter_info_button.setChecked(visible); self.filter_info_button.blockSignals(False)
        if hasattr(self,'filter_info_action'):
            self.filter_info_action.blockSignals(True); self.filter_info_action.setChecked(visible); self.filter_info_action.blockSignals(False)
        if visible:self.info_body.setFocus()
        elif self.doc is not None:self.filter_info_button.setFocus()

    def set_reduced_transparency(self,reduced):
        reduced=bool(reduced); self.canvas.hint.set_reduced_transparency(reduced); self.info_panel.set_reduced_transparency(reduced)
        self.settings.setValue('reduce_transparency',reduced)

    def _escape(self):
        if self.info_panel.isVisible():self.set_filter_info_visible(False)
        else:self.cancel_work()

    def photo_info(self):
        if self.doc is None:return
        info=dict(self.doc.source_info); info['path']=str(self.doc.path); info['dimensions']=[self.doc.width,self.doc.height]
        box=QMessageBox(self); box.setWindowTitle('照片信息'); box.setText(f'{self.doc.path.name}\n{self.source_label.text()}\n原文件只读；导出会创建新副本。')
        box.setDetailedText(json.dumps(info,ensure_ascii=False,indent=2,default=str)); box.exec()

    def about(self):
        QMessageBox.information(self,'关于 Feica Fotos',f'Feica Fotos {__version__}\n离线照片滤镜\n\n打开 → 选 Look → 对照 → 导出副本\nCtrl+O 打开 · Ctrl+Shift+S 导出\nCtrl+1…4 选滤镜 · B 对照 · Ctrl+I 介绍\n所有滤镜强度支持 0.01 精度\n画布获得焦点后按住空格临时查看原图\n\nDNG 使用内嵌 JPEG 预览，不进行 RAW 显影。\n独立的本地研究工具，与 Leica 无隶属或授权关系。\n厂商滤镜资源由用户本地提供，不自动下载或上传。')

    def resizeEvent(self,event):
        super().resizeEvent(event)
        if hasattr(self,'look_buttons'):
            compact=self.width()<960
            self.file_label.setVisible(not compact)
            self.open_button.setText('打开…' if compact else '打开照片…')
            self.compare_button.setText('原图对照' if compact else '查看原图')
            self.export_button.setText('导出…' if compact else '导出副本…')
            self._update_rail_metrics()
            if hasattr(self,'filter_info_button'):
                self.filter_info_button.setText('i' if compact else 'i  滤镜介绍')
                self.slider.setMinimumWidth(130 if compact else 180)

    def closeEvent(self,event):
        if self._discard_close and not self.jobs:self.settings.setValue('window_geometry',self.saveGeometry()); event.accept(); return
        if not self._closing and not self._confirm_transition(self._close_after_export):event.ignore(); return
        self._closing=True; self._discard_close=True
        self.timer.stop()
        for job in self.jobs.values():job.cancel.set()
        if self.jobs:event.ignore(); self._message('正在结束后台处理…'); return
        self.settings.setValue('window_geometry',self.saveGeometry())
        event.accept()

    def _close_after_export(self):self._discard_close=True; self.close()


def main(argv=None):
    parser=argparse.ArgumentParser(description='Feica Fotos — offline photo renderer')
    parser.add_argument('image',nargs='?'); parser.add_argument('--resource-dir',type=Path)
    args=parser.parse_args(argv)
    app=QApplication.instance() or QApplication(sys.argv[:1]); app.setApplicationName('Feica Fotos'); app.setOrganizationName('Feica Fotos'); app.setApplicationVersion(__version__); app.setWindowIcon(application_icon())
    font=app.font()
    if font.pointSizeF()<11:font.setPointSizeF(11); app.setFont(font)
    window=MainWindow(args.resource_dir); window.show()
    if args.image:QTimer.singleShot(0,lambda:window.open_path(args.image))
    return app.exec()

if __name__=='__main__':raise SystemExit(main())
