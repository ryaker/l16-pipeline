#!/usr/bin/env python3
"""
lri_lumen_app.py — Light L16 desktop Lumen application (PySide6).

Library browser → pick processed image set → click-to-focus editor →
export PNG / JPEG / DNG.

A "processed image set" is a directory containing:
  A1.png                                    (reference frame, from lri_extract.py)
  fused_10cams_median_depth.png             (fused depth, from lri_fuse_depth.py)

Usage:
  python3 lri_lumen_app.py [folder]
"""

import os
import sys
import time
import datetime
import argparse
import subprocess
import tempfile

import cv2
import numpy as np

from PySide6.QtCore import (Qt, QThread, Signal, QRunnable, QObject,
                             QThreadPool, QSize, QPoint, QRect, QTimer)
from PySide6.QtGui  import (QImage, QPixmap, QPainter, QPen, QColor,
                             QFont, QIcon, QAction, QKeySequence)
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QSplitter,
    QVBoxLayout, QHBoxLayout, QGridLayout, QFormLayout,
    QLabel, QSlider, QPushButton, QFileDialog, QListWidget,
    QListWidgetItem, QGroupBox, QScrollArea, QProgressBar,
    QStatusBar, QDockWidget, QToolBar, QSizePolicy,
    QMessageBox, QComboBox, QCheckBox, QDoubleSpinBox,
    QTabWidget, QFrame,
)

# ── Import core pipeline functions ────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
try:
    from lri_lumen import (load_data, apply_bokeh, apply_adjustments,
                           export_dng, depth_colormap)
except ImportError as e:
    print(f'Warning: could not import lri_lumen: {e}')
    # Minimal stubs so the app still launches
    def load_data(f, d):
        raise RuntimeError('lri_lumen.py not found')


# ── Helpers ───────────────────────────────────────────────────────────────────

def numpy_to_qpixmap(img_rgb: np.ndarray) -> QPixmap:
    """Convert uint8 RGB numpy array → QPixmap."""
    h, w, c = img_rgb.shape
    fmt = QImage.Format.Format_RGB888
    qimg = QImage(img_rgb.data, w, h, w * c, fmt).copy()
    return QPixmap.fromImage(qimg)


def find_image_sets(folder: str):
    """
    Walk *folder* and return list of dicts for every image set found.

    Two kinds of entries:
      type='processed'  — directory with A1.png (+ optional fused depth)
      type='lri'        — raw .lri file (thumbnail generated on demand)
    """
    sets = []

    _SKIP_DIRS = {'frames', 'depth', '__pycache__', '.git'}

    def _check_processed(path):
        if os.path.basename(path).lower() in _SKIP_DIRS:
            return None
        # Support both flat layout (PNGs directly in path) and
        # cache layout (PNGs in path/frames/, depth in path/fused_*.png)
        frames_subdir = os.path.join(path, 'frames')
        if os.path.isdir(frames_subdir):
            ref = first_frame(frames_subdir)
            img_path = os.path.join(frames_subdir, f'{ref}.png') if ref else None
        else:
            ref = first_frame(path)
            img_path = os.path.join(path, f'{ref}.png') if ref else None
        if ref is None:
            return None
        depth = os.path.join(path, 'fused_10cams_median_depth.png')
        return {'type':  'processed',
                'name':  os.path.basename(path),
                'path':  path,
                'img':   img_path,
                'depth': depth if os.path.isfile(depth) else None}

    def _scan_dir(path, depth=0):
        """Scan a directory for LRI files and processed sets (up to 2 levels deep)."""
        try:
            entries = sorted(os.listdir(path))
        except PermissionError:
            return
        for name in entries:
            full = os.path.join(path, name)
            if os.path.isdir(full):
                # Check if this dir is itself a processed set
                entry = _check_processed(full)
                if entry:
                    sets.append(entry)
                elif depth < 2:
                    # Recurse into date folders etc.
                    _scan_dir(full, depth + 1)
            elif (name.lower().endswith('.lri') and os.path.isfile(full)
                  and not name.startswith('.')):
                sets.append({'type':  'lri',
                             'name':  name,
                             'path':  full,
                             'img':   None,
                             'depth': None})

    # Also check the root itself
    entry = _check_processed(folder)
    if entry:
        sets.append(entry)

    _scan_dir(folder)
    return sets


# ── Pipeline worker (LRI → processed image set) ───────────────────────────────

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_EXTRACT_SCRIPT   = os.path.join(_SCRIPT_DIR, 'lri_extract.py')
_CAL_SCRIPT       = os.path.join(_SCRIPT_DIR, 'lri_calibration.py')
_DEPTH_PRO_SCRIPT = os.path.join(_SCRIPT_DIR, 'ml-depth-pro', 'src', 'depth_pro', 'cli', 'run.py')

CAMERAS_PRIORITY = ['A1','A2','A3','A4','A5','B1','B2','B3','B4','B5','C1','C2','C3','C4','C5','C6']


def first_frame(frames_dir: str) -> str | None:
    """Return the name (e.g. 'B1') of the first available extracted frame."""
    for name in CAMERAS_PRIORITY:
        if os.path.isfile(os.path.join(frames_dir, f'{name}.png')):
            return name
    return None


def lri_cache_dir(lri_path: str) -> str:
    """Return the cache directory path for a given LRI file."""
    stem = os.path.splitext(lri_path)[0]
    return stem + '_lumen'


class PipelineSignals(QObject):
    progress = Signal(str)          # human-readable status message
    finished = Signal(str)          # cache_dir path when complete
    error    = Signal(str)          # error message


class PipelineWorker(QRunnable):
    """
    Runs the LRI → viewer pipeline in a background thread.

    Stages (each skipped if outputs already exist):
      1. Extract  — lri_extract.py → cache/frames/A1…B5.png
      2. Depth    — depth_pro on A1 only → cache/depth/A1.npz
      3. Convert  — A1.npz (metres) → cache/fused_10cams_median_depth.png (16-bit mm)

    The cache dir is <lri_stem>_lumen/ next to the LRI file.
    """

    def __init__(self, lri_path: str):
        super().__init__()
        self.signals   = PipelineSignals()
        self._lri_path = lri_path
        self._cache    = lri_cache_dir(lri_path)

    # ── helpers ───────────────────────────────────────────────────────────────

    _DEPTH_PRO_DIR = os.path.join(_SCRIPT_DIR, 'ml-depth-pro')

    def _run(self, cmd, stage_name, cwd=None):
        """Run a subprocess; raise RuntimeError on failure."""
        result = subprocess.run(cmd, capture_output=True, timeout=600, cwd=cwd)
        if result.returncode != 0:
            raise RuntimeError(
                f'{stage_name} failed:\n{result.stderr.decode()[:400]}')

    # ── stages ────────────────────────────────────────────────────────────────

    def _stage_extract(self):
        frames_dir = os.path.join(self._cache, 'frames')
        if first_frame(frames_dir) is not None:
            self.signals.progress.emit('Extract: already done, skipping')
            return
        os.makedirs(frames_dir, exist_ok=True)
        self.signals.progress.emit('Extracting frames from LRI...')
        self._run([sys.executable, _EXTRACT_SCRIPT,
                   self._lri_path, frames_dir], 'Extract')
        self.signals.progress.emit('Extract complete')

    def _stage_depth_pro(self):
        frames_dir = os.path.join(self._cache, 'frames')
        depth_dir  = os.path.join(self._cache, 'depth')
        ref_name   = first_frame(frames_dir)
        if ref_name is None:
            raise RuntimeError('No extracted frames found — extract stage must have failed')
        ref_npz = os.path.join(depth_dir, f'{ref_name}.npz')
        if os.path.isfile(ref_npz):
            self.signals.progress.emit('Depth Pro: already done, skipping')
            return
        os.makedirs(depth_dir, exist_ok=True)
        ref_png = os.path.join(frames_dir, f'{ref_name}.png')
        self.signals.progress.emit(f'Running Depth Pro on {ref_name} (this takes ~30–60 s)...')
        self._run([sys.executable, _DEPTH_PRO_SCRIPT,
                   '-i', ref_png, '-o', depth_dir, '--skip-display'],
                  'Depth Pro',
                  cwd=self._DEPTH_PRO_DIR)
        self.signals.progress.emit('Depth Pro complete')

    def _stage_convert_depth(self):
        frames_dir = os.path.join(self._cache, 'frames')
        depth_dir  = os.path.join(self._cache, 'depth')
        fused_path = os.path.join(self._cache, 'fused_10cams_median_depth.png')
        if os.path.isfile(fused_path):
            self.signals.progress.emit('Depth convert: already done, skipping')
            return
        ref_name = first_frame(frames_dir)
        if ref_name is None:
            raise RuntimeError('No extracted frames found')
        ref_npz = os.path.join(depth_dir, f'{ref_name}.npz')
        self.signals.progress.emit('Converting depth to 16-bit PNG...')
        data     = np.load(ref_npz)
        depth_m  = data['depth'].astype(np.float32)   # metres
        depth_mm = depth_m * 1000.0                    # → mm

        # Resize to match reference frame if shapes differ
        ref_png = os.path.join(frames_dir, f'{ref_name}.png')
        ref = cv2.imread(ref_png)
        if ref is not None:
            rh, rw = ref.shape[:2]
            if depth_mm.shape != (rh, rw):
                depth_mm = cv2.resize(depth_mm, (rw, rh),
                                      interpolation=cv2.INTER_LINEAR)

        depth16 = np.clip(depth_mm, 0, 65535).astype(np.uint16)
        cv2.imwrite(fused_path, depth16)
        self.signals.progress.emit('Depth convert complete')

    # ── main ──────────────────────────────────────────────────────────────────

    def run(self):
        try:
            self._stage_extract()
            self._stage_depth_pro()
            self._stage_convert_depth()
            self.signals.finished.emit(self._cache)
        except Exception as e:
            self.signals.error.emit(str(e))


# ── LRI thumbnail worker ───────────────────────────────────────────────────────

class ThumbSignals(QObject):
    done  = Signal(str, QPixmap)   # (lri_path, thumbnail)
    error = Signal(str, str)       # (lri_path, message)


class LRIThumbnailWorker(QRunnable):
    """
    Background worker: runs lri_extract.py --scale 8 on one .lri file,
    reads the A1.png it writes to a temp dir, builds a QPixmap thumbnail,
    then emits done(lri_path, pixmap).
    """
    _extract_script = os.path.join(os.path.dirname(__file__), 'lri_extract.py')

    def __init__(self, lri_path: str, thumb_size: int = 140):
        super().__init__()
        self.signals    = ThumbSignals()
        self._lri_path  = lri_path
        self._thumb_size = thumb_size

    def run(self):
        lri = self._lri_path
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                cmd = [sys.executable, self._extract_script,
                       lri, tmpdir, '--scale', '8']
                result = subprocess.run(cmd, capture_output=True, timeout=120)
                if result.returncode != 0:
                    err = result.stderr.decode().strip()
                    out = result.stdout.decode().strip()
                    msg = err or out or f'returncode {result.returncode}'
                    print(f'[thumb] FAIL {os.path.basename(lri)}: {msg[:300]}',
                          flush=True)
                    raise RuntimeError(msg[:200])
                ref = first_frame(tmpdir)
                if ref is None:
                    files = os.listdir(tmpdir)
                    print(f'[thumb] no frames in {tmpdir}: {files}', flush=True)
                    raise RuntimeError('No frames produced by lri_extract')
                px = self._display_thumbnail(os.path.join(tmpdir, f'{ref}.png'),
                                             self._thumb_size)
            self.signals.done.emit(lri, px)
        except Exception as e:
            print(f'[thumb] ERROR {os.path.basename(lri)}: {e}', flush=True)
            self.signals.error.emit(lri, str(e))

    @staticmethod
    def _display_thumbnail(img_path: str, size: int) -> QPixmap:
        """
        Load a linear raw-extracted PNG, apply grey-world AWB + gamma 2.2,
        and return a square-cropped thumbnail QPixmap suitable for display.
        """
        img = cv2.imread(img_path)
        if img is None:
            ph = QPixmap(size, size)
            ph.fill(QColor(60, 60, 60))
            return ph
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)

        # Grey-world auto white balance: scale R and B to match G mean
        means = img.mean(axis=(0, 1))          # [R_mean, G_mean, B_mean]
        g_mean = means[1]
        if means[0] > 0:
            img[:, :, 0] *= g_mean / means[0]  # scale R
        if means[2] > 0:
            img[:, :, 2] *= g_mean / means[2]  # scale B
        img = np.clip(img, 0, 255)

        # Gamma 2.2: raw data is linear light; displays expect ~2.2 encoding
        img = (img / 255.0) ** (1.0 / 2.2) * 255.0
        img = np.clip(img, 0, 255).astype(np.uint8)

        # Square-crop and resize to thumbnail
        h, w = img.shape[:2]
        s = min(h, w)
        y0, x0 = (h - s) // 2, (w - s) // 2
        crop = img[y0:y0+s, x0:x0+s]
        crop = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
        return numpy_to_qpixmap(crop)


def make_thumbnail(img_path: str, size: int = 160) -> QPixmap:
    """Load image and return a square-cropped thumbnail QPixmap."""
    img = cv2.imread(img_path)
    if img is None:
        px = QPixmap(size, size)
        px.fill(QColor(60, 60, 60))
        return px
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    # Centre-crop to square
    s = min(h, w)
    y0, x0 = (h - s) // 2, (w - s) // 2
    crop = img[y0:y0+s, x0:x0+s]
    crop = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
    return numpy_to_qpixmap(crop)


# ── Image viewer widget ───────────────────────────────────────────────────────

class ImageViewer(QLabel):
    """
    Zoomable image display with click-to-focus crosshair.
    Emits focus_clicked(x_frac, y_frac) with normalised [0,1] coords.
    """
    focus_clicked = Signal(float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(400, 300)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.setStyleSheet('background: #1a1a1a;')

        self._pixmap_orig = None   # full-res QPixmap
        self._focus_xf    = None   # normalised focus coords
        self._focus_yf    = None

    def set_image(self, img_rgb: np.ndarray):
        self._pixmap_orig = numpy_to_qpixmap(img_rgb)
        self._focus_xf = None
        self._focus_yf = None
        self._update_display()

    def set_focus_point(self, xf: float, yf: float):
        self._focus_xf = xf
        self._focus_yf = yf
        self._update_display()

    def _update_display(self):
        if self._pixmap_orig is None:
            return
        scaled = self._pixmap_orig.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        if self._focus_xf is not None:
            # Draw crosshair on a copy
            img = scaled.toImage().copy()
            painter = QPainter(img)
            # Compute display coords from fractions
            x_off = (self.width()  - scaled.width())  // 2
            y_off = (self.height() - scaled.height()) // 2
            cx = int(self._focus_xf * scaled.width())
            cy = int(self._focus_yf * scaled.height())
            r = 16
            pen = QPen(QColor(255, 220, 0), 2)
            painter.setPen(pen)
            painter.drawEllipse(cx - r, cy - r, 2*r, 2*r)
            painter.drawLine(cx - r - 6, cy, cx + r + 6, cy)
            painter.drawLine(cx, cy - r - 6, cx, cy + r + 6)
            painter.end()
            scaled = QPixmap.fromImage(img)

        canvas = QPixmap(self.size())
        canvas.fill(QColor(26, 26, 26))
        painter = QPainter(canvas)
        x_off = (self.width()  - scaled.width())  // 2
        y_off = (self.height() - scaled.height()) // 2
        painter.drawPixmap(x_off, y_off, scaled)
        painter.end()
        self.setPixmap(canvas)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_display()

    def mousePressEvent(self, event):
        if self._pixmap_orig is None:
            return
        if event.button() != Qt.MouseButton.LeftButton:
            return
        scaled_w = min(self.width(),
                       int(self._pixmap_orig.width() * self.height()
                           / self._pixmap_orig.height()))
        scaled_h = min(self.height(),
                       int(self._pixmap_orig.height() * self.width()
                           / self._pixmap_orig.width()))
        x_off = (self.width()  - scaled_w) // 2
        y_off = (self.height() - scaled_h) // 2
        px = event.position().x() - x_off
        py = event.position().y() - y_off
        if 0 <= px <= scaled_w and 0 <= py <= scaled_h:
            xf = px / scaled_w
            yf = py / scaled_h
            self.set_focus_point(xf, yf)
            self.focus_clicked.emit(xf, yf)


# ── Library panel ─────────────────────────────────────────────────────────────

class LibraryPanel(QWidget):
    image_selected = Signal(dict)   # emits the image-set dict

    def __init__(self, parent=None):
        super().__init__(parent)
        self._sets = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        btn_open = QPushButton('Open Folder...')
        btn_open.clicked.connect(self._pick_folder)
        layout.addWidget(btn_open)

        self._list = QListWidget()
        self._list.setViewMode(QListWidget.ViewMode.IconMode)
        self._list.setIconSize(QSize(140, 140))
        self._list.setSpacing(6)
        self._list.setResizeMode(QListWidget.ResizeMode.Adjust)
        self._list.setMovement(QListWidget.Movement.Static)
        self._list.itemDoubleClicked.connect(self._on_double_click)
        layout.addWidget(self._list)

        self._status = QLabel('No folder loaded')
        self._status.setStyleSheet('color: #888; font-size: 11px;')
        layout.addWidget(self._status)

    # Max concurrent thumbnail extractions (each runs lri_extract subprocess)
    _MAX_THUMB_WORKERS = 4

    def load_folder(self, folder: str):
        self._sets = find_image_sets(folder)
        self._list.clear()
        self._lri_items  = {}   # lri_path → QListWidgetItem
        self._thumb_queue = []  # lri paths waiting for thumbnail
        self._thumb_active = 0  # currently running workers

        for s in self._sets:
            if s['type'] == 'processed':
                icon  = QIcon(make_thumbnail(s['img']))
                label = s['name']
                if s['depth'] is None:
                    label += '\n(no depth)'
            else:
                ph = QPixmap(140, 140)
                ph.fill(QColor(45, 45, 55))
                icon  = QIcon(ph)
                label = s['name']

            item = QListWidgetItem(icon, label)
            item.setData(Qt.ItemDataRole.UserRole, s)
            item.setSizeHint(QSize(150, 170))
            self._list.addItem(item)

            if s['type'] == 'lri':
                self._lri_items[s['path']] = item
                self._thumb_queue.append(s['path'])

        # Kick off initial batch (visible items)
        self._pump_thumb_queue()

        # Load more as user scrolls
        self._list.verticalScrollBar().valueChanged.connect(self._pump_thumb_queue)

        n = len(self._sets)
        lri_count = sum(1 for s in self._sets if s['type'] == 'lri')
        proc_count = n - lri_count
        parts = []
        if proc_count:
            parts.append(f'{proc_count} processed')
        if lri_count:
            parts.append(f'{lri_count} LRI')
        summary = ', '.join(parts) if parts else '0'
        self._status.setText(f'{summary} in {os.path.basename(folder)}')

    def _pump_thumb_queue(self, _=None):
        """Start thumbnail workers up to _MAX_THUMB_WORKERS, prioritising visible items."""
        if not self._thumb_queue:
            return
        pool = QThreadPool.globalInstance()
        # Figure out which items are currently visible
        visible = set()
        for i in range(self._list.count()):
            item = self._list.item(i)
            if not self._list.visualItemRect(item).isEmpty():
                d = item.data(Qt.ItemDataRole.UserRole)
                if d and d['type'] == 'lri':
                    visible.add(d['path'])
        # Re-order queue: visible items first
        visible_first = [p for p in self._thumb_queue if p in visible]
        rest = [p for p in self._thumb_queue if p not in visible]
        self._thumb_queue = visible_first + rest

        while self._thumb_active < self._MAX_THUMB_WORKERS and self._thumb_queue:
            lri_path = self._thumb_queue.pop(0)
            self._thumb_active += 1
            worker = LRIThumbnailWorker(lri_path)
            worker.signals.done.connect(self._on_thumb_ready)
            worker.signals.error.connect(self._on_thumb_error)
            pool.start(worker)

    def _on_thumb_ready(self, lri_path: str, pixmap: QPixmap):
        self._thumb_active = max(0, self._thumb_active - 1)
        item = self._lri_items.get(lri_path)
        if item:
            item.setIcon(QIcon(pixmap))
        self._pump_thumb_queue()

    def _on_thumb_error(self, lri_path: str, msg: str):
        self._thumb_active = max(0, self._thumb_active - 1)
        item = self._lri_items.get(lri_path)
        if item:
            ph = QPixmap(140, 140)
            ph.fill(QColor(60, 30, 30))
            painter = QPainter(ph)
            painter.setPen(QColor(200, 80, 80))
            painter.drawText(ph.rect(), Qt.AlignmentFlag.AlignCenter, '⚠ read\nerror')
            painter.end()
            item.setIcon(QIcon(ph))
            item.setToolTip(msg)   # hover to see actual error
        self._pump_thumb_queue()

    def _pick_folder(self):
        folder = QFileDialog.getExistingDirectory(self, 'Open Image Folder')
        if folder:
            self.load_folder(folder)

    def _on_double_click(self, item: QListWidgetItem):
        data = item.data(Qt.ItemDataRole.UserRole)
        if data:
            self.image_selected.emit(data)


# ── Slider row helper ─────────────────────────────────────────────────────────

class LabeledSlider(QWidget):
    value_changed = Signal(float)

    def __init__(self, label, lo, hi, default, step=0.01, decimals=2, parent=None):
        super().__init__(parent)
        self._lo = lo
        self._hi = hi
        self._step = step
        self._scale = max(1, round(1.0 / step))

        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(4)

        lbl = QLabel(label)
        lbl.setFixedWidth(90)
        lbl.setStyleSheet('font-size: 11px;')
        row.addWidget(lbl)

        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setRange(int(lo * self._scale), int(hi * self._scale))
        self._slider.setValue(int(default * self._scale))
        self._slider.valueChanged.connect(self._on_slider)
        row.addWidget(self._slider)

        self._val_lbl = QLabel(f'{default:.{decimals}f}')
        self._val_lbl.setFixedWidth(38)
        self._val_lbl.setAlignment(Qt.AlignmentFlag.AlignRight)
        self._val_lbl.setStyleSheet('font-size: 11px;')
        row.addWidget(self._val_lbl)

        self._decimals = decimals

    @property
    def value(self) -> float:
        return self._slider.value() / self._scale

    def set_value(self, v: float):
        self._slider.setValue(int(v * self._scale))

    def _on_slider(self, v: int):
        fv = v / self._scale
        self._val_lbl.setText(f'{fv:.{self._decimals}f}')
        self.value_changed.emit(fv)


# ── Render worker ─────────────────────────────────────────────────────────────

class RenderSignals(QObject):
    finished = Signal(np.ndarray)
    error    = Signal(str)


class RenderWorker(QRunnable):
    """Runs apply_bokeh in a thread; emits raw bokeh (no tone adjustments).
    Tone adjustments are always applied in the main thread so _bokeh_preview
    stays un-adjusted and sliders can re-apply without stacking."""

    def __init__(self, img_rgb, depth, focus_mm, f_equiv, f_number, scale):
        super().__init__()
        self.signals = RenderSignals()
        self._args = (img_rgb, depth, focus_mm, f_equiv, f_number, scale)

    def run(self):
        try:
            img_rgb, depth, focus_mm, f_equiv, f_number, scale = self._args
            bokeh = apply_bokeh(img_rgb, depth, focus_mm, f_number, f_equiv,
                                preview_scale=scale)
            self.signals.finished.emit(bokeh)
        except Exception as e:
            self.signals.error.emit(str(e))


# ── Main window ───────────────────────────────────────────────────────────────

class LumenWindow(QMainWindow):
    def __init__(self, start_folder: str | None = None):
        super().__init__()
        self.setWindowTitle('L16 Lumen')
        self.resize(1400, 900)
        self.setStyleSheet(DARK_STYLE)

        self._img_rgb   = None
        self._depth     = None
        self._focus_mm  = 3000.0
        self._bokeh_preview = None       # bokeh at 0.25 scale — reused by tone sliders
        self._current_result = None
        self._export_fmt_pending = None  # set when export needs a full-res render first
        self._export_stem = 'L16_lumen'  # updated when an image is loaded
        self._pool = QThreadPool.globalInstance()
        self._render_pending = False

        # Debounce timer for bokeh sliders (150 ms)
        self._bokeh_debounce = QTimer(self)
        self._bokeh_debounce.setSingleShot(True)
        self._bokeh_debounce.setInterval(150)
        self._bokeh_debounce.timeout.connect(self._trigger_bokeh_preview)

        self._build_ui()
        self._build_menu()

        if start_folder:
            self._library.load_folder(start_folder)

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_row = QHBoxLayout(central)
        main_row.setContentsMargins(0, 0, 0, 0)
        main_row.setSpacing(0)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Library (left dock)
        self._library = LibraryPanel()
        self._library.setMinimumWidth(180)
        self._library.setMaximumWidth(300)
        self._library.image_selected.connect(self._load_image_set)
        splitter.addWidget(self._library)

        # Image viewer (centre)
        viewer_widget = QWidget()
        vlay = QVBoxLayout(viewer_widget)
        vlay.setContentsMargins(0, 0, 0, 0)
        vlay.setSpacing(2)

        self._viewer = ImageViewer()
        self._viewer.focus_clicked.connect(self._on_focus_click)
        vlay.addWidget(self._viewer)

        self._focus_label = QLabel('Focus: — mm    Click the image to set focus point')
        self._focus_label.setStyleSheet('color: #ccc; font-size: 12px; padding: 4px 8px;')
        vlay.addWidget(self._focus_label)

        self._progress = QProgressBar()
        self._progress.setRange(0, 0)
        self._progress.setFixedHeight(4)
        self._progress.hide()
        vlay.addWidget(self._progress)

        splitter.addWidget(viewer_widget)

        # Controls (right panel)
        ctrl_scroll = QScrollArea()
        ctrl_scroll.setWidgetResizable(True)
        ctrl_scroll.setFixedWidth(260)
        ctrl_scroll.setFrameShape(QFrame.Shape.NoFrame)
        ctrl_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        ctrl_inner = QWidget()
        ctrl_scroll.setWidget(ctrl_inner)
        ctrl_layout = QVBoxLayout(ctrl_inner)
        ctrl_layout.setContentsMargins(8, 8, 8, 8)
        ctrl_layout.setSpacing(8)

        # Lens
        lens_grp = QGroupBox('Lens')
        lens_lay = QVBoxLayout(lens_grp)
        lens_lay.setSpacing(3)
        self._sl_focal   = LabeledSlider('Focal (mm)', 24, 200, 35, step=1, decimals=0)
        self._sl_fnum    = LabeledSlider('f-number',  0.7, 22, 1.8, step=0.1, decimals=1)
        self._sl_focus   = LabeledSlider('Focus (mm)', 200, 10000, 3000, step=10, decimals=0)
        for sl in (self._sl_focal, self._sl_fnum, self._sl_focus):
            lens_lay.addWidget(sl)
        ctrl_layout.addWidget(lens_grp)

        # Tone
        tone_grp = QGroupBox('Tone & Colour')
        tone_lay = QVBoxLayout(tone_grp)
        tone_lay.setSpacing(3)
        self._sl_exp    = LabeledSlider('Exposure',    -3,  3,  0,    step=0.1, decimals=1)
        self._sl_cont   = LabeledSlider('Contrast',   0.5, 2.5, 1.0, step=0.05, decimals=2)
        self._sl_hi     = LabeledSlider('Highlights', -1,  0,  0,    step=0.05, decimals=2)
        self._sl_sh     = LabeledSlider('Shadows',     0,  1,  0,    step=0.05, decimals=2)
        self._sl_wb_r   = LabeledSlider('WB Red',     0.5, 2, 1.0,   step=0.05, decimals=2)
        self._sl_wb_b   = LabeledSlider('WB Blue',    0.5, 2, 1.0,   step=0.05, decimals=2)
        self._sl_sat    = LabeledSlider('Saturation',  0,  3, 1.0,   step=0.05, decimals=2)
        self._sl_sharp  = LabeledSlider('Sharpness',   0,  5, 0,     step=0.1,  decimals=1)
        for sl in (self._sl_exp, self._sl_cont, self._sl_hi, self._sl_sh,
                   self._sl_wb_r, self._sl_wb_b, self._sl_sat, self._sl_sharp):
            tone_lay.addWidget(sl)
        ctrl_layout.addWidget(tone_grp)

        # View
        view_grp = QGroupBox('View')
        view_lay = QVBoxLayout(view_grp)
        self._view_combo = QComboBox()
        self._view_combo.addItems(['Photo', 'Depth map', 'Depth overlay'])
        view_lay.addWidget(self._view_combo)
        self._chk_preview = QCheckBox('Fast preview (1/4 res)')
        self._chk_preview.setChecked(True)
        view_lay.addWidget(self._chk_preview)
        ctrl_layout.addWidget(view_grp)

        # Apply
        self._btn_apply = QPushButton('Render Full Res')
        self._btn_apply.setFixedHeight(36)
        self._btn_apply.setEnabled(False)
        self._btn_apply.clicked.connect(self._apply_bokeh)
        self._btn_apply.setStyleSheet(
            'background: #2a6db5; color: white; font-weight: bold; border-radius: 4px;')
        ctrl_layout.addWidget(self._btn_apply)

        self._btn_reset = QPushButton('Reset')
        self._btn_reset.setEnabled(False)
        self._btn_reset.clicked.connect(self._reset)
        ctrl_layout.addWidget(self._btn_reset)

        ctrl_layout.addSpacing(8)

        # Export
        exp_grp = QGroupBox('Export')
        exp_lay = QVBoxLayout(exp_grp)
        self._btn_png  = QPushButton('Save PNG')
        self._btn_dng  = QPushButton('Save 16-bit DNG')
        self._btn_jpg  = QPushButton('Save JPEG')
        for btn in (self._btn_png, self._btn_dng, self._btn_jpg):
            btn.setEnabled(False)
            exp_lay.addWidget(btn)
        self._btn_png.clicked.connect(lambda: self._export('png'))
        self._btn_dng.clicked.connect(lambda: self._export('dng'))
        self._btn_jpg.clicked.connect(lambda: self._export('jpg'))
        ctrl_layout.addWidget(exp_grp)

        ctrl_layout.addStretch()
        splitter.addWidget(ctrl_scroll)

        splitter.setSizes([220, 920, 260])
        main_row.addWidget(splitter)

        self.setStatusBar(QStatusBar())
        self._status('Ready — open a folder to begin')

        # Connect slider changes — two tiers
        bokeh_sliders = [self._sl_focal, self._sl_fnum, self._sl_focus]
        tone_sliders  = [self._sl_exp, self._sl_cont, self._sl_hi, self._sl_sh,
                         self._sl_wb_r, self._sl_wb_b, self._sl_sat, self._sl_sharp]
        for sl in bokeh_sliders:
            sl.value_changed.connect(self._on_bokeh_slider_change)
        for sl in tone_sliders:
            sl.value_changed.connect(self._on_tone_slider_change)
        self._view_combo.currentIndexChanged.connect(self._on_view_change)

    def _build_menu(self):
        mb = self.menuBar()
        file_menu = mb.addMenu('File')
        open_act  = QAction('Open Folder...', self)
        open_act.setShortcut(QKeySequence.StandardKey.Open)
        open_act.triggered.connect(self._library._pick_folder)
        file_menu.addAction(open_act)
        file_menu.addSeparator()
        quit_act  = QAction('Quit', self)
        quit_act.setShortcut(QKeySequence.StandardKey.Quit)
        quit_act.triggered.connect(self.close)
        file_menu.addAction(quit_act)

    # ── Data loading ──────────────────────────────────────────────────────────

    def _load_image_set(self, data: dict):
        if data['type'] == 'lri':
            self._run_pipeline(data['path'])
            return

        if data['depth'] is None:
            QMessageBox.warning(self, 'No depth map',
                f'No fused depth map found in:\n{data["path"]}\n\n'
                'Run lri_fuse_depth.py on this image set first.')
            return

        self._status(f'Loading {data["name"]}...')
        self._progress.show()
        QApplication.processEvents()
        # Track stem for export filename (e.g. "L16_01922" from "L16_01922.lri")
        self._export_stem = os.path.splitext(data['name'])[0]

        try:
            # data['img'] and data['depth'] hold explicit file paths
            img_path   = data['img']
            depth_path = data['depth']
            import cv2 as _cv2
            img_bgr = _cv2.imread(img_path)
            if img_bgr is None:
                raise FileNotFoundError(f'Cannot load image: {img_path}')
            img_rgb = _cv2.cvtColor(img_bgr, _cv2.COLOR_BGR2RGB).astype(np.float32)
            # Grey-world AWB + gamma 2.2 so linear raw looks correct on display
            means = img_rgb.mean(axis=(0, 1))
            g = means[1]
            if means[0] > 0: img_rgb[:, :, 0] *= g / means[0]
            if means[2] > 0: img_rgb[:, :, 2] *= g / means[2]
            img_rgb = np.clip(img_rgb, 0, 255)
            img_rgb = (img_rgb / 255.0) ** (1.0 / 2.2) * 255.0
            self._img_rgb = np.clip(img_rgb, 0, 255).astype(np.uint8)

            depth_raw = _cv2.imread(depth_path, _cv2.IMREAD_UNCHANGED)
            if depth_raw is None:
                raise FileNotFoundError(f'Cannot load depth: {depth_path}')
            self._depth = depth_raw.astype(np.float32)
            valid = self._depth[self._depth > 0]
            self._focus_mm = float(np.median(valid))

            # Update focus slider range
            self._sl_focus._slider.setRange(int(valid.min() * 0.8),
                                            int(valid.max() * 1.2))
            self._sl_focus.set_value(self._focus_mm)
            self._focus_label.setText(
                f'Focus: {self._focus_mm:.0f} mm ({self._focus_mm/1000:.2f} m)   '
                f'Image: {self._img_rgb.shape[1]}×{self._img_rgb.shape[0]}   '
                f'Depth: {valid.min():.0f}–{valid.max():.0f} mm'
            )

            self._bokeh_preview = None  # invalidate old preview
            self._current_result = self._img_rgb.copy()
            self._show_image(self._img_rgb)

            for btn in (self._btn_apply, self._btn_reset,
                        self._btn_png, self._btn_dng, self._btn_jpg):
                btn.setEnabled(True)

            self._status(f'Loaded: {data["name"]} — adjusting sliders will auto-preview')
            # Auto-trigger initial bokeh preview
            self._bokeh_debounce.start()
        except Exception as e:
            QMessageBox.critical(self, 'Load error', str(e))
            self._status('Load failed')
        finally:
            self._progress.hide()

    # ── Display helpers ───────────────────────────────────────────────────────

    def _show_image(self, img_rgb: np.ndarray):
        mode = self._view_combo.currentText()
        if mode == 'Depth map' and self._depth is not None:
            disp = depth_colormap(self._depth)
        elif mode == 'Depth overlay' and self._depth is not None:
            disp = depth_colormap(self._depth, alpha=0.5, img_rgb=img_rgb)
        else:
            disp = img_rgb
        self._viewer.set_image(disp)

    def _status(self, msg: str):
        self.statusBar().showMessage(msg)

    # ── LRI pipeline ──────────────────────────────────────────────────────────

    def _run_pipeline(self, lri_path: str):
        cache = lri_cache_dir(lri_path)
        name  = os.path.basename(lri_path)

        # Check if already fully processed
        fused = os.path.join(cache, 'fused_10cams_median_depth.png')
        if os.path.isfile(fused) and first_frame(os.path.join(cache, 'frames')) is not None:
            self._status(f'Loading cached: {name}')
            self._load_processed_cache(cache, name)
            return

        self._status(f'Processing {name}…')
        self._progress.show()

        worker = PipelineWorker(lri_path)
        worker.signals.progress.connect(self._on_pipeline_progress)
        worker.signals.finished.connect(self._on_pipeline_done)
        worker.signals.error.connect(self._on_pipeline_error)
        self._pool.start(worker)

    def _on_pipeline_progress(self, msg: str):
        self._status(msg)

    def _on_pipeline_done(self, cache_dir: str):
        self._progress.hide()
        name = os.path.basename(cache_dir)
        self._status(f'Pipeline complete — loading {name}')
        self._load_processed_cache(cache_dir, name)

    def _on_pipeline_error(self, msg: str):
        self._progress.hide()
        self._status('Pipeline failed')
        QMessageBox.critical(self, 'Pipeline error', msg)

    def _load_processed_cache(self, cache_dir: str, name: str):
        """Load a completed cache dir (frames/ + fused PNG) into the viewer."""
        frames_dir = os.path.join(cache_dir, 'frames')
        ref = first_frame(frames_dir)
        if ref is None:
            QMessageBox.critical(self, 'Load error', f'No frames found in {frames_dir}')
            return
        data = {
            'type':  'processed',
            'name':  name,
            'path':  cache_dir,
            'img':   os.path.join(frames_dir, f'{ref}.png'),
            'depth': os.path.join(cache_dir, 'fused_10cams_median_depth.png'),
        }
        self._load_image_set(data)

    # ── Interaction ───────────────────────────────────────────────────────────

    def _on_focus_click(self, xf: float, yf: float):
        if self._depth is None:
            return
        H, W = self._depth.shape
        xi = min(int(xf * W), W - 1)
        yi = min(int(yf * H), H - 1)
        d = self._depth[yi, xi]
        if d == 0:
            r = 30
            patch = self._depth[max(0, yi-r):yi+r, max(0, xi-r):xi+r]
            valid = patch[patch > 0]
            d = float(np.median(valid)) if len(valid) else self._focus_mm
        self._focus_mm = float(d)
        self._sl_focus.set_value(self._focus_mm)
        self._focus_label.setText(
            f'Focus: {self._focus_mm:.0f} mm ({self._focus_mm/1000:.2f} m)   '
            f'pixel ({xi}, {yi})'
        )
        self._viewer.set_focus_point(xf, yf)
        # Trigger live bokeh preview after click-to-focus
        self._bokeh_debounce.start()

    def _on_bokeh_slider_change(self, _val):
        """Bokeh slider moved — sync state and restart debounce."""
        self._focus_mm = self._sl_focus.value
        self._bokeh_debounce.start()

    def _on_tone_slider_change(self, _val):
        """Tone slider moved — instantly re-apply adjustments on stored bokeh preview."""
        if self._bokeh_preview is None:
            return
        result = apply_adjustments(
            self._bokeh_preview,
            self._sl_exp.value,
            self._sl_cont.value,
            self._sl_hi.value,
            self._sl_sh.value,
            self._sl_wb_r.value,
            self._sl_wb_b.value,
            self._sl_sat.value,
            self._sl_sharp.value,
        )
        self._current_result = result
        self._show_image(result)

    def _on_view_change(self, _idx):
        if self._current_result is not None:
            self._show_image(self._current_result)

    # ── Bokeh rendering ───────────────────────────────────────────────────────

    def _trigger_bokeh_preview(self):
        """Called by debounce timer — render bokeh at 0.25 scale for live preview."""
        if self._img_rgb is None:
            return
        self._render_pending = True
        worker = RenderWorker(
            self._img_rgb, self._depth,
            self._sl_focus.value,
            self._sl_focal.value,
            self._sl_fnum.value,
            0.25,  # always preview scale
        )
        worker.signals.finished.connect(self._on_bokeh_preview_done)
        worker.signals.error.connect(self._on_render_error)
        self._pool.start(worker)

    def _on_bokeh_preview_done(self, bokeh_raw: np.ndarray):
        """Bokeh preview render done — store and apply current tone settings."""
        self._bokeh_preview = bokeh_raw
        self._render_pending = False
        result = apply_adjustments(
            bokeh_raw,
            self._sl_exp.value,
            self._sl_cont.value,
            self._sl_hi.value,
            self._sl_sh.value,
            self._sl_wb_r.value,
            self._sl_wb_b.value,
            self._sl_sat.value,
            self._sl_sharp.value,
        )
        self._current_result = result
        self._show_image(result)
        self._status(
            f'Preview — focus {self._sl_focus.value:.0f} mm, '
            f'f/{self._sl_fnum.value:.1f}  (click "Render Full Res" to export)'
        )

    def _apply_bokeh(self):
        """Render Full Res — full scale render for export."""
        if self._img_rgb is None:
            return
        self._btn_apply.setEnabled(False)
        self._btn_apply.setText('Rendering...')
        self._progress.show()

        worker = RenderWorker(
            self._img_rgb, self._depth,
            self._sl_focus.value,
            self._sl_focal.value,
            self._sl_fnum.value,
            1.0,  # full resolution
        )
        worker.signals.finished.connect(self._on_render_done)
        worker.signals.error.connect(self._on_render_error)
        self._pool.start(worker)

    def _on_render_done(self, bokeh_raw: np.ndarray):
        # Store raw bokeh (no adjustments) so tone sliders can re-apply without stacking
        self._bokeh_preview = bokeh_raw
        result = apply_adjustments(
            bokeh_raw,
            self._sl_exp.value,
            self._sl_cont.value,
            self._sl_hi.value,
            self._sl_sh.value,
            self._sl_wb_r.value,
            self._sl_wb_b.value,
            self._sl_sat.value,
            self._sl_sharp.value,
        )
        self._current_result = result
        self._show_image(result)
        self._btn_apply.setEnabled(True)
        self._btn_apply.setText('Render Full Res')
        self._progress.hide()
        self._status(
            f'Full res — focus {self._sl_focus.value:.0f} mm, '
            f'f/{self._sl_fnum.value:.1f}'
        )
        # If an export was waiting for full-res render, fire it now
        if self._export_fmt_pending:
            fmt = self._export_fmt_pending
            self._export_fmt_pending = None
            self._do_export(fmt, result)

    def _on_render_error(self, msg: str):
        QMessageBox.critical(self, 'Render error', msg)
        self._btn_apply.setEnabled(True)
        self._btn_apply.setText('Render Full Res')
        self._export_fmt_pending = None
        self._progress.hide()

    def _reset(self):
        if self._img_rgb is not None:
            self._current_result = self._img_rgb.copy()
            self._show_image(self._img_rgb)
            self._viewer.set_focus_point(None, None)  # clears crosshair
            self._status('Reset')

    # ── Export ────────────────────────────────────────────────────────────────

    def _export(self, fmt: str):
        if self._img_rgb is None:
            return

        # If current result is lower than full res, require a full-res render first
        is_preview = (self._current_result is None or
                      self._current_result.shape[:2] != self._img_rgb.shape[:2])
        if is_preview:
            reply = QMessageBox.question(
                self, 'Full-resolution export?',
                'Current view is a live preview (1/4 resolution).\n'
                'Click "Render Full Res" first, then export — or export at preview quality?',
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No |
                QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Yes,
            )
            if reply == QMessageBox.StandardButton.Cancel:
                return
            if reply == QMessageBox.StandardButton.Yes:
                # Kick off full-res render, then export when done
                self._export_fmt_pending = fmt
                self._apply_bokeh()
                return
            # No = export whatever we have (preview quality)

        self._do_export(fmt, self._current_result)

    def _do_export(self, fmt: str, img_rgb: np.ndarray):
        ext_map = {'png': 'PNG files (*.png)', 'dng': 'DNG files (*.dng)',
                   'jpg': 'JPEG files (*.jpg)'}
        default_name = f'{self._export_stem}_lumen.{fmt}'
        path, _ = QFileDialog.getSaveFileName(
            self, 'Save Image', default_name, ext_map.get(fmt, '*.*'))
        if not path:
            return

        try:
            self._progress.show()
            self._status(f'Saving {fmt.upper()}...')
            QApplication.processEvents()

            if fmt == 'dng':
                export_dng(img_rgb, path,
                           focus_mm=self._sl_focus.value,
                           f_equiv_mm=self._sl_focal.value,
                           f_number=self._sl_fnum.value)
            elif fmt == 'jpg':
                bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
                cv2.imwrite(path, bgr, [cv2.IMWRITE_JPEG_QUALITY, 97])
            else:  # png
                bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
                cv2.imwrite(path, bgr)

            size_mb = os.path.getsize(path) / 1e6
            self._status(f'Saved: {os.path.basename(path)}  ({size_mb:.1f} MB)')
        except Exception as e:
            QMessageBox.critical(self, 'Export error', str(e))
        finally:
            self._progress.hide()


# ── Dark stylesheet ───────────────────────────────────────────────────────────

DARK_STYLE = """
QWidget          { background: #232323; color: #ddd; font-size: 12px; }
QMainWindow      { background: #1a1a1a; }
QMenuBar         { background: #2a2a2a; }
QMenuBar::item:selected { background: #3a3a3a; }
QMenu            { background: #2a2a2a; border: 1px solid #444; }
QGroupBox        { border: 1px solid #444; border-radius: 4px; margin-top: 8px;
                   padding-top: 6px; font-size: 11px; color: #aaa; }
QGroupBox::title { subcontrol-origin: margin; left: 8px; color: #bbb; }
QSlider::groove:horizontal { height: 4px; background: #444; border-radius: 2px; }
QSlider::handle:horizontal { width: 12px; height: 12px; margin: -4px 0;
                              background: #4a9; border-radius: 6px; }
QSlider::sub-page:horizontal { background: #4a9; border-radius: 2px; }
QPushButton      { background: #3a3a3a; border: 1px solid #555;
                   border-radius: 3px; padding: 4px 8px; }
QPushButton:hover  { background: #484848; }
QPushButton:pressed{ background: #2a2a2a; }
QPushButton:disabled { color: #666; background: #2a2a2a; border-color: #3a3a3a; }
QScrollBar:vertical { width: 8px; background: #2a2a2a; }
QScrollBar::handle:vertical { background: #555; border-radius: 4px; min-height: 20px; }
QListWidget      { background: #1e1e1e; border: none; }
QListWidget::item:selected { background: #2a5a8a; }
QComboBox        { background: #3a3a3a; border: 1px solid #555; border-radius: 3px;
                   padding: 2px 6px; }
QComboBox::drop-down { border: none; }
QCheckBox        { color: #ccc; }
QCheckBox::indicator { width: 14px; height: 14px; }
QStatusBar       { background: #1e1e1e; color: #888; font-size: 11px; }
QScrollArea      { border: none; }
QProgressBar     { border: none; background: #333; }
QProgressBar::chunk { background: #4a9; }
"""


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='L16 Lumen desktop app')
    parser.add_argument('folder', nargs='?', default=None,
                        help='Folder to open on startup')
    args = parser.parse_args()

    app = QApplication(sys.argv)
    app.setApplicationName('L16 Lumen')

    win = LumenWindow(start_folder=args.folder or '/tmp')
    win.show()

    # Auto-load /tmp if it has processed image sets
    if not args.folder:
        sets = find_image_sets('/tmp')
        if sets:
            win._library.load_folder('/tmp')

    sys.exit(app.exec())


if __name__ == '__main__':
    main()
