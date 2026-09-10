#!/usr/bin/env python3
"""Lightweight PySide6 bounding-box annotator.

Boxes come from a YOLO labels folder and/or from running a YOLO model.
Everything is editable: drag to move, drag handles to resize, drag on empty
canvas to create, Del to remove.

Pick a folder of loose images and it is reshaped into the YOLO layout —
images move to <folder>/images, labels are written to <folder>/labels (ask
by default; --organize always|never).

On the first launch the three default checkpoints (yolov8x.pt, yolo11x.pt,
yolo26x.pt) are downloaded into ./weights next to this script; later launches
see them on disk and skip it. --no-download opts out, --weights-dir relocates.

    python bbox_annotator.py --images IMGS [--labels LBLS] [--classes classes.txt] [--model yolo11n.pt]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, QThread, Signal
from PySide6.QtGui import (QAction, QBrush, QColor, QFont, QIcon, QImage,
                           QKeySequence, QPainter, QPen, QPixmap)
from PySide6.QtWidgets import (QApplication, QComboBox, QDockWidget,
                               QDoubleSpinBox,
                               QFileDialog, QGraphicsItem, QGraphicsRectItem,
                               QGraphicsScene, QGraphicsView, QHBoxLayout,
                               QInputDialog, QLabel, QListWidget,
                               QListWidgetItem, QMainWindow, QMessageBox,
                               QProgressDialog, QPushButton, QSlider,
                               QVBoxLayout, QWidget)

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# Weights fetched automatically the first time the app runs.
DEFAULT_WEIGHTS = ["yolov8x.pt", "yolo11x.pt", "yolo26x.pt"]
WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"
POSE_WEIGHT = "yolo11x-pose.pt"
ASSET_RELEASE = "v8.4.0"
ASSET_URLS = (
    "https://github.com/ultralytics/assets/releases/download/{rel}/{name}",
    "https://github.com/ultralytics/assets/releases/latest/download/{name}",
)
MIN_WEIGHT_BYTES = 1_000_000

# Detections overlapping an existing same-class box by at least this IoU update
# that box instead of being added next to it.
MERGE_IOU = 0.55

PALETTE = [
    "#ff3838", "#2ecc71", "#3498db", "#ff9f1a", "#9b59b6", "#1abc9c",
    "#e84393", "#f1c40f", "#00b8d4", "#8e44ad", "#d35400", "#16a085",
    "#c0392b", "#27ae60", "#2980b9", "#f39c12",
]


def cls_color(idx: int) -> QColor:
    return QColor(PALETTE[idx % len(PALETTE)])


# ------------------------------------------------------------ weight fetching

class WeightDownloader(QThread):
    """Streams the default weights into WEIGHTS_DIR, reporting progress."""

    progress = Signal(str, int, float, float)   # name, index, got_mb, total_mb
    finished_one = Signal(str)
    failed = Signal(str, str)
    done = Signal()

    def __init__(self, names: list[str], dest: Path, parent=None):
        super().__init__(parent)
        self.names = names
        self.dest = dest
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        import urllib.error
        import urllib.request
        self.dest.mkdir(parents=True, exist_ok=True)
        for i, name in enumerate(self.names):
            out = self.dest / name
            if weight_ok(out):
                self.finished_one.emit(name)
                continue
            last_err = "no url worked"
            for tpl in ASSET_URLS:
                url = tpl.format(rel=ASSET_RELEASE, name=name)
                tmp = out.with_suffix(out.suffix + ".part")
                try:
                    req = urllib.request.Request(
                        url, headers={"User-Agent": "bbox-annotator"})
                    with urllib.request.urlopen(req, timeout=30) as r, \
                            open(tmp, "wb") as fh:
                        total = float(r.headers.get("Content-Length") or 0)
                        got = 0
                        while True:
                            if self._cancel:
                                tmp.unlink(missing_ok=True)
                                return
                            chunk = r.read(1 << 20)
                            if not chunk:
                                break
                            fh.write(chunk)
                            got += len(chunk)
                            self.progress.emit(name, i, got / 1e6, total / 1e6)
                    if tmp.stat().st_size < MIN_WEIGHT_BYTES:
                        raise OSError(f"only {tmp.stat().st_size} bytes")
                    tmp.replace(out)
                    last_err = ""
                    break
                except Exception as e:                          # noqa: BLE001
                    tmp.unlink(missing_ok=True)
                    last_err = str(e)
            if last_err:
                self.failed.emit(name, last_err)
            else:
                self.finished_one.emit(name)
        self.done.emit()


def iou(a: "Box", b: "Box") -> float:
    ax1, ay1, ax2, ay2 = min(a.x1, a.x2), min(a.y1, a.y2), max(a.x1, a.x2), max(a.y1, a.y2)
    bx1, by1, bx2, by2 = min(b.x1, b.x2), min(b.y1, b.y2), max(b.x1, b.x2), max(b.y1, b.y2)
    iw = min(ax2, bx2) - max(ax1, bx1)
    ih = min(ay2, by2) - max(ay1, by1)
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def merge_detections(existing: list["Box"], new: list["Box"],
                     iou_thr: float = MERGE_IOU) -> tuple[int, int]:
    """Fold `new` into `existing` in place: overlapping same-class boxes are
    updated rather than duplicated. Returns (updated, added)."""
    updated = added = 0
    for nb in new:
        best, best_iou = None, iou_thr
        for eb in existing:
            if eb.cls != nb.cls:
                continue
            v = iou(eb, nb)
            if v >= best_iou:
                best, best_iou = eb, v
        if best is None:
            existing.append(nb)
            added += 1
        else:
            best.x1, best.y1, best.x2, best.y2 = nb.x1, nb.y1, nb.x2, nb.y2
            best.conf = nb.conf
            updated += 1
    return updated, added


def weight_ok(path: Path) -> bool:
    return path.exists() and path.stat().st_size >= MIN_WEIGHT_BYTES


# --------------------------------------------------------------- mining job

class MinerThread(QThread):
    """Runs the whole mine_person_crops pipeline off the GUI thread."""

    progress = Signal(str)
    done = Signal(str, int, int)      # out_dir, images, crops
    picked_ready = Signal(list)       # in-place mode: the selected candidates
    failed = Signal(str)

    def __init__(self, source: Path, out: Path, budget: int, per_track: int,
                 weights_dir: Path, allow_partial: bool, parent=None,
                 in_place: bool = False):
        super().__init__(parent)
        self.source, self.out = source, out
        self.budget, self.per_track = budget, per_track
        self.weights_dir = weights_dir
        self.allow_partial = allow_partial
        self.in_place = in_place
        self.stats: dict = {}
        self._stop = False

    def stop(self):
        self._stop = True

    def _args(self):
        import mine_person_crops as M
        ap = M.build_parser()
        args = ap.parse_args(["--source", str(self.source),
                              "--out", str(self.out)])
        args.budget = self.budget
        args.per_track = self.per_track
        args.allow_partial = self.allow_partial
        args.det_model = str(self.weights_dir / "yolo11x.pt")
        pose = self.weights_dir / "yolo11x-pose.pt"
        args.pose_model = str(pose) if pose.exists() else None
        try:
            import torch
            args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        except ImportError:
            args.device = "cpu"
        return M, args

    def run(self):
        try:
            M, args = self._args()
        except Exception as e:                                  # noqa: BLE001
            self.failed.emit(f"cannot set up the miner: {e}")
            return
        try:
            self.progress.emit("loading models…")

            def cb(src, frame, n):
                self.progress.emit(f"{src} · frame {frame} · {n} candidates")
                return not self._stop

            cands, meta = M.scan(args, on_progress=cb)
            if self._stop:
                self.failed.emit("stopped")
                return
            if not cands:
                self.failed.emit("no person detections passed the filters")
                return
            self.progress.emit(f"selecting from {len(cands)} candidates…")
            picked, stats = M.select(cands, args)
            self.stats = stats
            if not picked:
                self.failed.emit(
                    "every candidate was filtered out — try Allow partial "
                    "bodies, or a lower budget/quality gate")
                return
            if self.in_place:
                self.progress.emit(f"{len(picked)} unique people selected")
                self.picked_ready.emit(picked)
                return
            self.progress.emit(f"writing {len(picked)} frames…")
            man = M.write_review_set(picked, Path(args.out), args, stats)
            self.done.emit(str(args.out), man["images"], man["crops"])
        except BaseException as e:          # SystemExit included: never die mute
            import traceback
            traceback.print_exc()
            self.failed.emit(str(e) or type(e).__name__)


# ---------------------------------------------------------------- data model

@dataclass
class Box:
    cls: int
    # pixel coords in image space
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float | None = None

    def norm(self, w: int, h: int) -> tuple[float, float, float, float]:
        cx = (self.x1 + self.x2) / 2 / w
        cy = (self.y1 + self.y2) / 2 / h
        bw = abs(self.x2 - self.x1) / w
        bh = abs(self.y2 - self.y1) / h
        return cx, cy, bw, bh


@dataclass
class Frame:
    path: Path
    boxes: list[Box] = field(default_factory=list)
    loaded: bool = False
    dirty: bool = False


# ------------------------------------------------------------- graphics item

class BoxItem(QGraphicsRectItem):
    """Rect stored in scene(=image pixel) coords; pos stays at origin."""

    HANDLE_PX = 8.0          # on-screen handle size
    view_scale = 1.0         # updated by the view on zoom

    NONE, MOVE = 0, 1
    TL, T, TR, R, BR, B, BL, L = range(2, 10)

    def __init__(self, box: Box, owner: "Canvas"):
        super().__init__()
        self.box = box
        self.owner = owner
        self.setFlags(QGraphicsItem.ItemIsSelectable | QGraphicsItem.ItemIsFocusable)
        self.setAcceptHoverEvents(True)
        self._mode = self.NONE
        self._start = QPointF()
        self._orig = QRectF()
        self.sync_from_box()

    # -- geometry helpers -------------------------------------------------
    def sync_from_box(self):
        b = self.box
        self.setRect(QRectF(QPointF(min(b.x1, b.x2), min(b.y1, b.y2)),
                            QPointF(max(b.x1, b.x2), max(b.y1, b.y2))))
        self.update()

    def push_to_box(self):
        r = self.rect().normalized()
        self.box.x1, self.box.y1 = r.left(), r.top()
        self.box.x2, self.box.y2 = r.right(), r.bottom()

    def _hs(self) -> float:
        return self.HANDLE_PX / max(self.view_scale, 1e-6)

    def _handle_rects(self) -> dict[int, QRectF]:
        r = self.rect().normalized()
        s = self._hs()
        cx, cy = r.center().x(), r.center().y()
        pts = {
            self.TL: (r.left(), r.top()), self.T: (cx, r.top()),
            self.TR: (r.right(), r.top()), self.R: (r.right(), cy),
            self.BR: (r.right(), r.bottom()), self.B: (cx, r.bottom()),
            self.BL: (r.left(), r.bottom()), self.L: (r.left(), cy),
        }
        return {k: QRectF(x - s / 2, y - s / 2, s, s) for k, (x, y) in pts.items()}

    def handle_at(self, p: QPointF) -> int:
        if self.isSelected():
            for k, hr in self._handle_rects().items():
                if hr.contains(p):
                    return k
        return self.MOVE if self.rect().normalized().contains(p) else self.NONE

    def boundingRect(self) -> QRectF:
        return self.rect().normalized().adjusted(-self._hs(), -self._hs(),
                                                 self._hs(), self._hs())

    def shape(self):
        from PySide6.QtGui import QPainterPath
        path = QPainterPath()
        path.addRect(self.boundingRect())
        return path

    # -- painting ---------------------------------------------------------
    def paint(self, painter: QPainter, option, widget=None):
        r = self.rect().normalized()
        col = cls_color(self.box.cls)
        lw = (2.5 if self.isSelected() else 1.6) / max(self.view_scale, 1e-6)
        pen = QPen(col, lw)
        pen.setCosmetic(False)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(r)

        # translucent fill only when selected, keeps view clean
        if self.isSelected():
            fill = QColor(col)
            fill.setAlpha(45)
            painter.setBrush(QBrush(fill))
            painter.setPen(Qt.NoPen)
            painter.drawRect(r)
            painter.setBrush(QBrush(col))
            painter.setPen(QPen(QColor("#ffffff"), lw * 0.5))
            for hr in self._handle_rects().values():
                painter.drawRect(hr)

        # Label drawn in device space (constant on-screen size), clipped to this
        # item's boundingRect: painting outside it would leave smears when the
        # box is dragged, since Qt only repaints the area the item claims.
        name = self.owner.class_name(self.box.cls)
        if self.box.conf is not None:
            name = f"{name} {self.box.conf:.2f}"
        painter.save()
        painter.resetTransform()
        clip = QRectF(self.owner.view.mapFromScene(
            self.mapToScene(self.boundingRect())).boundingRect())
        painter.setClipRect(clip)
        dev = self.owner.view.mapFromScene(self.mapToScene(r.topLeft()))
        f = QFont()
        f.setPointSize(9)
        painter.setFont(f)
        fm = painter.fontMetrics()
        tw, th = fm.horizontalAdvance(name) + 8, fm.height() + 2
        bg = QRectF(dev.x() + 1, dev.y() + 1, tw, th)   # inside the box
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(col))
        painter.drawRect(bg)
        painter.setPen(QPen(QColor("#101010")))
        painter.drawText(bg.adjusted(4, 0, 0, 0), Qt.AlignVCenter | Qt.AlignLeft, name)
        painter.restore()

    # -- interaction ------------------------------------------------------
    def hoverMoveEvent(self, ev):
        cursors = {
            self.TL: Qt.SizeFDiagCursor, self.BR: Qt.SizeFDiagCursor,
            self.TR: Qt.SizeBDiagCursor, self.BL: Qt.SizeBDiagCursor,
            self.T: Qt.SizeVerCursor, self.B: Qt.SizeVerCursor,
            self.L: Qt.SizeHorCursor, self.R: Qt.SizeHorCursor,
            self.MOVE: Qt.SizeAllCursor, self.NONE: Qt.ArrowCursor,
        }
        self.setCursor(cursors[self.handle_at(ev.pos())])
        super().hoverMoveEvent(ev)

    def mousePressEvent(self, ev):
        if ev.button() != Qt.LeftButton:
            ev.ignore()
            return
        self._mode = self.handle_at(ev.pos())
        if self._mode == self.NONE:
            ev.ignore()
            return
        self.setSelected(True)
        self._start = ev.pos()
        self._orig = self.rect().normalized()
        ev.accept()

    def mouseMoveEvent(self, ev):
        if self._mode == self.NONE:
            return
        d = ev.pos() - self._start
        r = QRectF(self._orig)
        if self._mode == self.MOVE:
            r.translate(d)
        else:
            if self._mode in (self.TL, self.T, self.TR):
                r.setTop(r.top() + d.y())
            if self._mode in (self.BL, self.B, self.BR):
                r.setBottom(r.bottom() + d.y())
            if self._mode in (self.TL, self.L, self.BL):
                r.setLeft(r.left() + d.x())
            if self._mode in (self.TR, self.R, self.BR):
                r.setRight(r.right() + d.x())
        r = self.owner.clamp_rect(r.normalized(), keep_size=self._mode == self.MOVE)
        self.prepareGeometryChange()
        self.setRect(r)
        self.update()
        self.push_to_box()
        self.owner.box_changed(self)
        ev.accept()

    def mouseReleaseEvent(self, ev):
        if self._mode != self.NONE:
            self._mode = self.NONE
            self.owner.commit()
        ev.accept()


# -------------------------------------------------------------------- canvas

class Canvas(QGraphicsView):
    boxes_changed = Signal()
    selection_changed = Signal()

    def __init__(self, app: "MainWindow"):
        super().__init__()
        self.app = app
        self.view = self
        self.scene_ = QGraphicsScene(self)
        self.setScene(self.scene_)
        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setViewportUpdateMode(QGraphicsView.SmartViewportUpdate)
        self.setBackgroundBrush(QBrush(QColor("#1e1e1e")))
        self.setMouseTracking(True)
        self.pix_item = None
        self.img_w = self.img_h = 0
        self._new_item: BoxItem | None = None
        self._new_origin = QPointF()
        self._panning = False
        self._pan_from = QPointF()
        self.scene_.selectionChanged.connect(self.selection_changed)

    # -- plumbing ---------------------------------------------------------
    def class_name(self, i: int) -> str:
        return self.app.class_name(i)

    def clamp_point(self, p: QPointF) -> QPointF:
        if not self.img_w:
            return p
        return QPointF(min(max(p.x(), 0.0), float(self.img_w)),
                       min(max(p.y(), 0.0), float(self.img_h)))

    def clamp_rect(self, r: QRectF, keep_size=False) -> QRectF:
        if not self.img_w:
            return r
        if keep_size:
            dx = min(0.0, self.img_w - r.right()) + max(0.0, -r.left())
            dy = min(0.0, self.img_h - r.bottom()) + max(0.0, -r.top())
            r.translate(dx, dy)
            return r
        r.setLeft(max(0.0, min(r.left(), self.img_w)))
        r.setTop(max(0.0, min(r.top(), self.img_h)))
        r.setRight(max(0.0, min(r.right(), self.img_w)))
        r.setBottom(max(0.0, min(r.bottom(), self.img_h)))
        return r

    def box_changed(self, item: BoxItem):
        self.app.mark_dirty()
        self.app.refresh_box_row(item.box)

    def commit(self):
        self.boxes_changed.emit()

    # -- content ----------------------------------------------------------
    def show_image(self, pixmap: QPixmap, boxes: list[Box], fit=True):
        self.scene_.clear()
        self.pix_item = self.scene_.addPixmap(pixmap)
        self.pix_item.setZValue(-10)
        self.img_w, self.img_h = pixmap.width(), pixmap.height()
        self.scene_.setSceneRect(QRectF(0, 0, self.img_w, self.img_h))
        for b in boxes:
            self.scene_.addItem(BoxItem(b, self))
        if fit:
            self.fit()

    def items_boxes(self) -> list[BoxItem]:
        return [i for i in self.scene_.items() if isinstance(i, BoxItem)]

    def item_for(self, box: Box) -> BoxItem | None:
        for i in self.items_boxes():
            if i.box is box:
                return i
        return None

    def add_box_item(self, box: Box) -> BoxItem:
        it = BoxItem(box, self)
        self.scene_.addItem(it)
        return it

    def fit(self):
        if self.pix_item:
            self.fitInView(self.scene_.sceneRect(), Qt.KeepAspectRatio)
            self._sync_scale()

    def _sync_scale(self):
        BoxItem.view_scale = self.transform().m11()
        for i in self.items_boxes():
            i.prepareGeometryChange()
            i.update()

    # -- events -----------------------------------------------------------
    def wheelEvent(self, ev):
        if not self.pix_item:
            return
        f = 1.0015 ** ev.angleDelta().y()
        self.scale(f, f)
        self._sync_scale()

    def mousePressEvent(self, ev):
        if ev.button() == Qt.MiddleButton or (
                ev.button() == Qt.LeftButton and ev.modifiers() & Qt.ShiftModifier):
            self._panning = True
            self._pan_from = ev.position()
            self.setCursor(Qt.ClosedHandCursor)
            ev.accept()
            return
        super().mousePressEvent(ev)
        if ev.isAccepted() and self.scene_.mouseGrabberItem():
            return
        if ev.button() == Qt.LeftButton and self.pix_item:
            p = self.clamp_point(self.mapToScene(ev.position().toPoint()))
            self.scene_.clearSelection()
            box = Box(self.app.current_class(), p.x(), p.y(), p.x(), p.y())
            self._new_origin = p
            self._new_item = self.add_box_item(box)
            self._new_item.setSelected(True)
            ev.accept()

    def mouseMoveEvent(self, ev):
        if self._panning:
            d = ev.position() - self._pan_from
            self._pan_from = ev.position()
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - int(d.x()))
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - int(d.y()))
            return
        if self._new_item is not None:
            p = self.clamp_point(self.mapToScene(ev.position().toPoint()))
            r = QRectF(self._new_origin, p).normalized()
            self._new_item.prepareGeometryChange()
            self._new_item.setRect(r)
            self._new_item.push_to_box()
            self.app.status(f"new box  {r.width():.0f}x{r.height():.0f}")
            return
        super().mouseMoveEvent(ev)

    def mouseReleaseEvent(self, ev):
        if self._panning:
            self._panning = False
            self.setCursor(Qt.ArrowCursor)
            return
        if self._new_item is not None:
            it, self._new_item = self._new_item, None
            r = it.rect().normalized()
            if r.width() < 3 or r.height() < 3:
                self.scene_.removeItem(it)
            else:
                self.app.register_new_box(it.box)
                it.setSelected(True)
            self.commit()
            return
        super().mouseReleaseEvent(ev)

    def keyPressEvent(self, ev):
        if ev.key() in (Qt.Key_Delete, Qt.Key_Backspace):
            self.app.delete_selected()
            return
        super().keyPressEvent(ev)


# --------------------------------------------------------------- main window

class MainWindow(QMainWindow):
    def __init__(self, args):
        super().__init__()
        self.setWindowTitle("BBox Annotator")
        self.resize(1360, 860)

        self.frames: list[Frame] = []
        self.idx = -1
        self.classes: list[str] = []
        self.images_dir: Path | None = None
        self.labels_dir: Path | None = None
        self.model = None
        self.model_path: str | None = None
        self.conf = 0.25
        self._suppress = False
        self.weights_dir = Path(args.weights_dir) if args.weights_dir else WEIGHTS_DIR
        self._dl: WeightDownloader | None = None
        self.organize = args.organize
        self.manifest: dict | None = None
        self.review: dict[str, str] = {}      # image name -> approved/rejected
        self.crops_out = Path(args.crops_out) if args.crops_out else None
        self.crop_margin = args.crop_margin
        self._miner: MinerThread | None = None

        self.canvas = Canvas(self)
        self.setCentralWidget(self.canvas)
        self.canvas.selection_changed.connect(self.on_scene_selection)
        self.canvas.boxes_changed.connect(self.rebuild_box_list)

        self._build_docks()
        self._build_actions()
        self.statusBar().showMessage("Open an images folder to start  (Ctrl+O)")

        self.margin_spin.setValue(self.crop_margin)
        self.refresh_weight_combo()
        if not args.no_download:
            self.ensure_weights()

        if args.classes:
            self.load_classes(Path(args.classes))
        if args.images:
            self.open_images(Path(args.images),
                             Path(args.labels) if args.labels else None)
        if args.model:
            self.load_model(args.model)
        self.conf = args.conf
        self.conf_slider.setValue(int(self.conf * 100))

    # -- ui ---------------------------------------------------------------
    def _build_docks(self):
        # files
        self.file_list = QListWidget()
        self.file_list.currentRowChanged.connect(self.goto)
        d = QDockWidget("Images", self)
        d.setWidget(self.file_list)
        self.addDockWidget(Qt.LeftDockWidgetArea, d)

        # right panel
        right = QWidget()
        lay = QVBoxLayout(right)
        lay.setContentsMargins(6, 6, 6, 6)

        lay.addWidget(QLabel("<b>① Build a dataset</b>"))
        b = QPushButton("Mine person crops from video/folder…")
        b.setToolTip("With images open: detect people on all of them in place.\n"
                     "Otherwise: pick a video/folder, track + score diversity, "
                     "and build a review set.")
        b.clicked.connect(self.mine_dataset)
        lay.addWidget(b)
        self.mine_label = QLabel("")
        self.mine_label.setWordWrap(True)
        self.mine_label.setStyleSheet("color:#9a9a9a")
        lay.addWidget(self.mine_label)

        lay.addWidget(QLabel("<b>Class for new boxes</b>"))
        self.class_combo = QComboBox()
        lay.addWidget(self.class_combo)
        row = QHBoxLayout()
        b = QPushButton("+ class")
        b.clicked.connect(self.add_class)
        row.addWidget(b)
        b = QPushButton("Set on selected")
        b.clicked.connect(self.apply_class_to_selection)
        row.addWidget(b)
        lay.addLayout(row)

        lay.addWidget(QLabel("<b>Boxes</b>"))
        self.box_list = QListWidget()
        self.box_list.currentRowChanged.connect(self.on_list_selection)
        self.box_list.itemClicked.connect(self.on_row_clicked)
        lay.addWidget(self.box_list, 1)

        row = QHBoxLayout()
        b = QPushButton("Delete (Del)")
        b.clicked.connect(self.delete_selected)
        row.addWidget(b)
        b = QPushButton("Clear all")
        b.clicked.connect(self.clear_boxes)
        row.addWidget(b)
        lay.addLayout(row)

        lay.addWidget(QLabel("<b>YOLO model</b>"))
        self.weight_combo = QComboBox()
        self.weight_combo.currentIndexChanged.connect(self.on_weight_picked)
        lay.addWidget(self.weight_combo)
        row = QHBoxLayout()
        b = QPushButton("Browse .pt…")
        b.clicked.connect(self.browse_model)
        row.addWidget(b)
        b = QPushButton("Re-download")
        b.clicked.connect(lambda: self.ensure_weights(force=True))
        row.addWidget(b)
        lay.addLayout(row)
        b = QPushButton("Load selected model")
        b.clicked.connect(self.load_selected_model)
        lay.addWidget(b)
        self.model_label = QLabel("<i>no model loaded yet</i>")
        self.model_label.setWordWrap(True)
        lay.addWidget(self.model_label)
        self.conf_label = QLabel("conf 0.25")
        lay.addWidget(self.conf_label)
        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setRange(1, 95)
        self.conf_slider.setValue(25)
        self.conf_slider.valueChanged.connect(self.set_conf)
        lay.addWidget(self.conf_slider)
        b = QPushButton("Detect this image  (R)")
        b.clicked.connect(self.detect_current)
        lay.addWidget(b)
        b = QPushButton("Detect on ALL images…  (Ctrl+R)")
        b.clicked.connect(self.detect_all)
        lay.addWidget(b)

        lay.addWidget(QLabel("<b>② Manual review</b>"))
        self.review_label = QLabel("—")
        self.review_label.setWordWrap(True)
        lay.addWidget(self.review_label)
        self.cand_label = QLabel("")
        self.cand_label.setWordWrap(True)
        self.cand_label.setStyleSheet("color:#9a9a9a")
        lay.addWidget(self.cand_label)
        row = QHBoxLayout()
        b = QPushButton("✓ Approve (Space)")
        b.clicked.connect(lambda: self.set_review("approved"))
        row.addWidget(b)
        b = QPushButton("✗ Reject (X)")
        b.clicked.connect(lambda: self.set_review("rejected"))
        row.addWidget(b)
        lay.addLayout(row)
        row = QHBoxLayout()
        b = QPushButton("Clear mark")
        b.clicked.connect(lambda: self.set_review("pending", advance=False))
        row.addWidget(b)
        b = QPushButton("Approve all shown")
        b.clicked.connect(self.approve_all)
        row.addWidget(b)
        lay.addLayout(row)
        row = QHBoxLayout()
        row.addWidget(QLabel("crop padding"))
        self.margin_spin = QDoubleSpinBox()
        self.margin_spin.setRange(0.0, 1.0)
        self.margin_spin.setSingleStep(0.05)
        self.margin_spin.setDecimals(2)
        self.margin_spin.valueChanged.connect(self.set_crop_margin)
        row.addWidget(self.margin_spin)
        lay.addLayout(row)
        b = QPushButton("③ Export approved person crops  (Ctrl+E)")
        b.clicked.connect(self.export_crops)
        lay.addWidget(b)

        d = QDockWidget("Tools", self)
        d.setWidget(right)
        d.setMinimumWidth(250)
        self.addDockWidget(Qt.RightDockWidgetArea, d)

    def _build_actions(self):
        m = self.menuBar().addMenu("&File")

        def act(text, slot, shortcut=None, menu=m):
            a = QAction(text, self)
            if shortcut:
                a.setShortcut(QKeySequence(shortcut))
            a.triggered.connect(slot)
            menu.addAction(a)
            self.addAction(a)
            return a

        act("Open images folder…", self.pick_images, "Ctrl+O")
        act("Set labels folder…", self.pick_labels, "Ctrl+L")
        act("Load classes.txt…", lambda: self.load_classes(), "Ctrl+K")
        m.addSeparator()
        act("Save labels", self.save_current, "Ctrl+S")
        act("Save all changed", self.save_all, "Ctrl+Shift+S")

        r = self.menuBar().addMenu("&Review")
        act("Approve frame", lambda: self.set_review("approved"), "Space", r)
        act("Reject frame", lambda: self.set_review("rejected"), "X", r)
        act("Next unreviewed", self.next_pending, "N", r)
        act("Export approved crops…", self.export_crops, "Ctrl+E", r)

        n = self.menuBar().addMenu("&Navigate")
        act("Next image", self.next_image, "D", n)
        act("Next image (arrow)", self.next_image, "Right", n)
        act("Previous image", self.prev_image, "A", n)
        act("Previous image (arrow)", self.prev_image, "Left", n)
        act("Fit to window", self.canvas.fit, "F", n)
        act("Detect (model)", self.detect_current, "R", n)
        act("Detect on all images", self.detect_all, "Ctrl+R", n)
        act("Load selected model", self.load_selected_model, "Ctrl+M", n)
        for i in range(9):
            a = QAction(f"Select class {i}", self)
            a.setShortcut(QKeySequence(str(i + 1)))
            a.triggered.connect(lambda _=False, k=i: self.pick_class(k))
            self.addAction(a)

    # -- classes ----------------------------------------------------------
    def class_name(self, i: int) -> str:
        return self.classes[i] if 0 <= i < len(self.classes) else str(i)

    def pick_class(self, k: int) -> None:
        if k < self.class_combo.count():
            self.class_combo.setCurrentIndex(k)

    def current_class(self) -> int:
        return max(0, self.class_combo.currentIndex())

    def ensure_classes(self, n: int):
        changed = False
        while len(self.classes) < n:
            self.classes.append(f"class{len(self.classes)}")
            changed = True
        if changed:
            self.refresh_class_combo()

    def refresh_class_combo(self):
        keep = self.class_combo.currentIndex()
        self.class_combo.blockSignals(True)
        self.class_combo.clear()
        for i, c in enumerate(self.classes):
            pm = QPixmap(12, 12)
            pm.fill(cls_color(i))
            self.class_combo.addItem(QIcon(pm), f"{i}: {c}")
        if 0 <= keep < self.class_combo.count():
            self.class_combo.setCurrentIndex(keep)
        self.class_combo.blockSignals(False)

    def load_classes(self, path: Path | None = None):
        if path is None:
            f, _ = QFileDialog.getOpenFileName(self, "classes.txt", "",
                                               "Text (*.txt *.names);;All (*)")
            if not f:
                return
            path = Path(f)
        if path.exists():
            self.classes = [ln.strip() for ln in
                            path.read_text().splitlines() if ln.strip()]
            self.refresh_class_combo()
            self.status(f"{len(self.classes)} classes from {path.name}")
            self.reload_current(fit=False)

    def add_class(self):
        name, ok = QInputDialog.getText(self, "New class", "Name:")
        if ok and name.strip():
            self.classes.append(name.strip())
            self.refresh_class_combo()
            self.class_combo.setCurrentIndex(len(self.classes) - 1)

    def apply_class_to_selection(self):
        c = self.current_class()
        for it in self.canvas.scene_.selectedItems():
            if isinstance(it, BoxItem):
                it.box.cls = c
                it.update()
        self.mark_dirty()
        self.rebuild_box_list()

    # -- folders / io -----------------------------------------------------
    def pick_images(self):
        d = QFileDialog.getExistingDirectory(self, "Images folder")
        if d:
            self.open_images(Path(d), None)

    def pick_labels(self):
        d = QFileDialog.getExistingDirectory(self, "Labels folder")
        if d:
            self.labels_dir = Path(d)
            for f in self.frames:
                f.loaded, f.boxes = False, []
            self.reload_current()
            self.status(f"labels: {self.labels_dir}")

    def guess_labels_dir(self, images: Path) -> Path:
        if images.name == "images":
            return images.parent / "labels"    # canonical YOLO layout
        for cand in (images.parent / "labels",
                     Path(str(images).replace("images", "labels"))
                     if "images" in str(images) else images / "labels"):
            if Path(cand).is_dir():
                return Path(cand)
        # flat folder: keep existing sidecar .txt files where they are,
        # otherwise start a labels/ subfolder rather than littering the images
        stems = {p.stem for p in images.iterdir()
                 if p.suffix.lower() in IMG_EXT}
        if any(p.stem in stems for p in images.glob("*.txt")):
            return images
        return images / "labels"

    def organize_folder(self, folder: Path) -> Path | None:
        """Turn a folder of loose images into <folder>/images + <folder>/labels.

        Returns the images folder to use, or None if the user cancelled.
        Folders already in YOLO layout are left untouched.
        """
        loose = sorted(p for p in folder.iterdir()
                       if p.is_file() and p.suffix.lower() in IMG_EXT)
        sub = folder / "images"
        if not loose:
            # nothing to move; use <folder>/images when that is where they are
            return sub if sub.is_dir() else folder
        if folder.name == "images":
            return folder                      # already the images folder

        if self.organize == "never":
            return folder
        if self.organize == "ask":
            n_in_sub = len(list(sub.glob("*"))) if sub.is_dir() else 0
            r = QMessageBox.question(
                self, "Organize folder",
                f"{folder} holds {len(loose)} loose image(s).\n\n"
                f"Move them into:\n    {sub}\n"
                f"and write labels to:\n    {folder / 'labels'}\n"
                + (f"\n({sub} already exists with {n_in_sub} entries)\n"
                   if n_in_sub else "")
                + "\nChoose No to leave the files where they are.",
                QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
                QMessageBox.Yes)
            if r == QMessageBox.Cancel:
                return None
            if r == QMessageBox.No:
                return folder

        sub.mkdir(exist_ok=True)
        (folder / "labels").mkdir(exist_ok=True)
        moved = skipped = 0
        for src in loose:
            dst = sub / src.name
            if dst.exists():
                if dst.stat().st_size == src.stat().st_size:
                    src.unlink()               # identical copy already there
                else:
                    skipped += 1
                    continue
            else:
                try:
                    src.replace(dst)           # same filesystem: instant
                except OSError:
                    shutil.move(str(src), str(dst))
            moved += 1
            # carry an existing sidecar label across, if any
            side = src.with_suffix(".txt")
            if side.exists() and side.name != "classes.txt":
                tgt = folder / "labels" / side.name
                if not tgt.exists():
                    try:
                        side.replace(tgt)
                    except OSError:
                        shutil.move(str(side), str(tgt))
        self.status(f"moved {moved} image(s) into {sub}"
                    + (f" · {skipped} name clash(es) left in place" if skipped else ""))
        if skipped:
            QMessageBox.warning(
                self, "Name clashes",
                f"{skipped} image(s) already exist in {sub} with a different "
                f"size and were left in {folder}.")
        return sub

    def open_images(self, images: Path, labels: Path | None):
        if not images.is_dir():
            QMessageBox.warning(self, "Not a folder", str(images))
            return
        organized = self.organize_folder(images)
        if organized is None:
            return
        images = organized
        files = sorted(p for p in images.iterdir()
                       if p.is_file() and p.suffix.lower() in IMG_EXT)
        if not files:
            QMessageBox.warning(self, "Empty", f"No images in {images}")
            return
        self.images_dir = images
        self.labels_dir = labels or self.guess_labels_dir(images)
        self.labels_dir.mkdir(parents=True, exist_ok=True)
        self.frames = [Frame(p) for p in files]
        self.idx = -1
        for cf in (images / "classes.txt", images.parent / "classes.txt",
                   self.labels_dir / "classes.txt"):
            if not self.classes and cf.exists():
                self.load_classes(cf)
                break
        if not self.classes:
            self.classes = ["class0"]
            self.refresh_class_combo()
        self.file_list.blockSignals(True)
        self.file_list.clear()
        for p in files:
            self.file_list.addItem(QListWidgetItem(p.name))
        self.file_list.blockSignals(False)
        self.load_review()
        self.goto(0)
        self.status(f"{len(files)} images · labels {self.labels_dir}")

    def label_path(self, frame: Frame) -> Path:
        return (self.labels_dir or frame.path.parent) / (frame.path.stem + ".txt")

    def load_labels(self, frame: Frame, w: int, h: int):
        frame.boxes = []
        lp = self.label_path(frame)
        if lp.exists():
            for ln in lp.read_text().splitlines():
                parts = ln.split()
                if len(parts) < 5:
                    continue
                try:
                    c = int(float(parts[0]))
                    cx, cy, bw, bh = (float(v) for v in parts[1:5])
                except ValueError:
                    continue
                frame.boxes.append(Box(c, (cx - bw / 2) * w, (cy - bh / 2) * h,
                                       (cx + bw / 2) * w, (cy + bh / 2) * h))
        frame.loaded = True
        if frame.boxes:
            self.ensure_classes(max(b.cls for b in frame.boxes) + 1)

    def save_current(self):
        if self.idx < 0:
            return
        self._save(self.frames[self.idx])
        self.status(f"saved {self.label_path(self.frames[self.idx])}")

    def save_all(self):
        n = 0
        for f in self.frames:
            if f.dirty:
                self._save(f)
                n += 1
        self.status(f"saved {n} label file(s)")

    def _save(self, frame: Frame):
        pm = QPixmap(str(frame.path))
        w, h = pm.width(), pm.height()
        if not w:
            return
        out = self.label_path(frame)
        out.parent.mkdir(parents=True, exist_ok=True)
        lines = []
        for b in frame.boxes:
            cx, cy, bw, bh = b.norm(w, h)
            if bw <= 0 or bh <= 0:
                continue
            lines.append(f"{b.cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        out.write_text("\n".join(lines) + ("\n" if lines else ""))
        frame.dirty = False
        self.refresh_file_row(frame)

    # -- navigation -------------------------------------------------------
    def goto(self, row: int, fit=True):
        if self._suppress or not self.frames or not (0 <= row < len(self.frames)):
            return
        if self.idx >= 0 and self.idx != row and self.frames[self.idx].dirty:
            self._save(self.frames[self.idx])
        self.idx = row
        f = self.frames[row]
        pm = QPixmap(str(f.path))
        if pm.isNull():
            self.status(f"cannot read {f.path.name}")
            return
        if not f.loaded:
            self.load_labels(f, pm.width(), pm.height())
        self.canvas.show_image(pm, f.boxes, fit=fit)
        self._suppress = True
        self.file_list.setCurrentRow(row)
        self._suppress = False
        self.rebuild_box_list()
        self.update_review_ui()
        self.setWindowTitle(f"BBox Annotator — {f.path.name} "
                            f"[{row + 1}/{len(self.frames)}]")
        self.status(f"{f.path.name}  {pm.width()}x{pm.height()}  "
                    f"{len(f.boxes)} boxes")

    def reload_current(self, fit=True):
        if self.idx >= 0:
            row, self.idx = self.idx, -1
            self.goto(row, fit=fit)

    def next_image(self):
        if self.frames:
            self.goto(min(self.idx + 1, len(self.frames) - 1))

    def prev_image(self):
        if self.frames:
            self.goto(max(self.idx - 1, 0))

    # -- box list ---------------------------------------------------------
    def cur_frame(self) -> Frame | None:
        return self.frames[self.idx] if 0 <= self.idx < len(self.frames) else None

    def rebuild_box_list(self):
        f = self.cur_frame()
        self._suppress = True
        self.box_list.clear()
        if f:
            for i, b in enumerate(f.boxes):
                self.box_list.addItem(self._row_text(i, b))
                self.box_list.item(i).setForeground(QBrush(cls_color(b.cls)))
            sel = [it.box for it in self.canvas.scene_.selectedItems()
                   if isinstance(it, BoxItem)]
            # -1 unless something is really selected, so clicking row 0 works
            self.box_list.setCurrentRow(f.boxes.index(sel[0]) if sel else -1)
        self._suppress = False

    def _row_text(self, i: int, b: Box) -> str:
        return (f"{i}  {self.class_name(b.cls)}  "
                f"[{b.x1:.0f},{b.y1:.0f} {abs(b.x2-b.x1):.0f}x{abs(b.y2-b.y1):.0f}]")

    def refresh_box_row(self, box: Box):
        f = self.cur_frame()
        if not f or box not in f.boxes:
            return
        i = f.boxes.index(box)
        if i < self.box_list.count():
            self.box_list.item(i).setText(self._row_text(i, box))

    def refresh_file_row(self, frame: Frame):
        if frame in self.frames:
            i = self.frames.index(frame)
            n = len(frame.boxes)
            mark = {"approved": "✓ ", "rejected": "✗ "}.get(
                self.review.get(frame.path.name), "")
            if frame.dirty:
                mark += "* "
            self.file_list.item(i).setText(
                f"{mark}{frame.path.name}" + (f"  ({n})" if n else ""))

    def on_row_clicked(self, item: QListWidgetItem):
        self.on_list_selection(self.box_list.row(item))

    def on_list_selection(self, row: int):
        f = self.cur_frame()
        if self._suppress or not f or not (0 <= row < len(f.boxes)):
            return
        self.canvas.scene_.clearSelection()
        it = self.canvas.item_for(f.boxes[row])
        if it:
            it.setSelected(True)
            self.canvas.ensureVisible(it, 40, 40)

    def on_scene_selection(self):
        f = self.cur_frame()
        sel = [it.box for it in self.canvas.scene_.selectedItems()
               if isinstance(it, BoxItem)]
        if not f or not sel or self._suppress:
            return
        self._suppress = True
        try:
            self.box_list.setCurrentRow(f.boxes.index(sel[0]))
        except ValueError:
            pass
        self._suppress = False

    def register_new_box(self, box: Box):
        f = self.cur_frame()
        if f and box not in f.boxes:
            f.boxes.append(box)
            self.mark_dirty()
            self.rebuild_box_list()

    def delete_selected(self):
        f = self.cur_frame()
        if not f:
            return
        gone = False
        for it in list(self.canvas.scene_.selectedItems()):
            if isinstance(it, BoxItem):
                if it.box in f.boxes:
                    f.boxes.remove(it.box)
                self.canvas.scene_.removeItem(it)
                gone = True
        if not gone and self.box_list.currentRow() >= 0:
            r = self.box_list.currentRow()
            if r < len(f.boxes):
                b = f.boxes.pop(r)
                it = self.canvas.item_for(b)
                if it:
                    self.canvas.scene_.removeItem(it)
                gone = True
        if gone:
            self.mark_dirty()
            self.rebuild_box_list()

    def clear_boxes(self):
        f = self.cur_frame()
        if not f or not f.boxes:
            return
        if QMessageBox.question(self, "Clear", f"Remove all {len(f.boxes)} boxes?") \
                != QMessageBox.Yes:
            return
        f.boxes.clear()
        self.mark_dirty()
        self.reload_current(fit=False)

    def mark_dirty(self):
        f = self.cur_frame()
        if f:
            f.dirty = True
            self.refresh_file_row(f)

    # -- model ------------------------------------------------------------
    def set_conf(self, v: int):
        self.conf = v / 100
        self.conf_label.setText(f"conf {self.conf:.2f}")

    # weights: discovery, download, selection
    def available_weights(self) -> list[Path]:
        found = [self.weights_dir / n for n in DEFAULT_WEIGHTS
                 if weight_ok(self.weights_dir / n)]
        if self.weights_dir.is_dir():
            for p in sorted(self.weights_dir.glob("*.pt")):
                if p not in found and weight_ok(p):
                    found.append(p)
        return found

    def refresh_weight_combo(self):
        keep = self.weight_combo.currentData()
        self.weight_combo.blockSignals(True)
        self.weight_combo.clear()
        for p in self.available_weights():
            self.weight_combo.addItem(
                f"{p.name}  ({p.stat().st_size / 1e6:.0f} MB)", str(p))
        missing = [n for n in DEFAULT_WEIGHTS
                   if not weight_ok(self.weights_dir / n)]
        for n in missing:
            self.weight_combo.addItem(f"{n}  (not downloaded)", None)
        if self.model_path and self.weight_combo.findData(self.model_path) < 0:
            self.weight_combo.addItem(Path(self.model_path).name, self.model_path)
        i = self.weight_combo.findData(keep or self.model_path)
        self.weight_combo.setCurrentIndex(max(i, 0))
        self.weight_combo.blockSignals(False)

    def ensure_weights(self, force=False):
        names = DEFAULT_WEIGHTS if force else [
            n for n in DEFAULT_WEIGHTS if not weight_ok(self.weights_dir / n)]
        if not names:
            self.status(f"weights ready: {', '.join(DEFAULT_WEIGHTS)}")
            return
        if self._miner and self._miner.isRunning():
            self._miner.stop()
            self._miner.wait(5000)
        if self._dl and self._dl.isRunning():
            return
        if force:
            for n in names:
                (self.weights_dir / n).unlink(missing_ok=True)
        dlg = QProgressDialog(
            f"Downloading {len(names)} YOLO weight file(s) to\n"
            f"{self.weights_dir}\n\nThis happens once; the UI is usable "
            f"meanwhile (detection needs the files).",
            "Cancel", 0, 100, self)
        dlg.setWindowTitle("First-run model download")
        dlg.setMinimumWidth(460)
        dlg.setAutoClose(True)
        dlg.setWindowModality(Qt.NonModal)

        dl = WeightDownloader(names, self.weights_dir, self)
        self._dl = dl
        n_total = len(names)

        def on_progress(name, i, got, total):
            pct = (i + (got / total if total else 0)) / n_total * 100
            dlg.setValue(int(pct))
            dlg.setLabelText(f"{name}   {got:.0f} / {total:.0f} MB"
                             f"   ({i + 1} of {n_total})")

        def on_one(name):
            self.refresh_weight_combo()
            self.status(f"downloaded {name}")

        def on_failed(name, err):
            self.status(f"download failed for {name}: {err}")
            QMessageBox.warning(
                self, "Download failed",
                f"Could not download {name}:\n{err}\n\n"
                f"Put the file in {self.weights_dir} manually, or use Browse .pt…")

        def on_done():
            dlg.setValue(100)
            dlg.close()
            self.refresh_weight_combo()
            ok = [n for n in DEFAULT_WEIGHTS if weight_ok(self.weights_dir / n)]
            self.status(f"weights ready: {', '.join(ok) or 'none'}")

        dl.progress.connect(on_progress)
        dl.finished_one.connect(on_one)
        dl.failed.connect(on_failed)
        dl.done.connect(on_done)
        dlg.canceled.connect(dl.cancel)
        dlg.show()
        dl.start()

    def on_weight_picked(self, _idx: int):
        path = self.weight_combo.currentData()
        if not path:
            self.model, self.model_path = None, None
            self.model_label.setText("<i>file not downloaded yet</i>")
            return
        if path != self.model_path:
            self.model, self.model_path = None, None
            self.model_label.setText(
                f"<i>{Path(path).name} — loads on first detect</i>")

    def browse_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "YOLO weights", str(self.weights_dir),
            "Weights (*.pt *.onnx *.engine);;All (*)")
        if path:
            self.load_model(path)
            self.refresh_weight_combo()

    def load_selected_model(self) -> bool:
        """Load the picked weights into memory right now."""
        if not self.ensure_model():
            return False
        dev = "?"
        try:
            dev = str(next(self.model.model.parameters()).device)
        except Exception:                                       # noqa: BLE001
            pass
        self.model_label.setText(
            f"<b>{Path(self.model_path).name}</b><br>"
            f"{len(getattr(self.model, 'names', {}) or {})} classes · {dev}")
        return True

    def ensure_model(self) -> bool:
        """Load the weights picked in the combo, if not already in memory."""
        want = self.weight_combo.currentData()
        if self.model is not None and self.model_path == want:
            return True
        if not want:
            QMessageBox.information(
                self, "No weights",
                "The selected weights are not on disk yet.\n"
                "Wait for the download to finish, hit Re-download, "
                "or pick a file with Browse .pt…")
            return False
        return self.load_model(want)

    def load_model(self, path: str | None = None) -> bool:
        if path is None:
            path, _ = QFileDialog.getOpenFileName(self, "YOLO weights", "",
                                                  "Weights (*.pt *.onnx *.engine);;All (*)")
            if not path:
                return False
        try:
            from ultralytics import YOLO
        except ImportError:
            QMessageBox.warning(self, "ultralytics missing",
                                "pip install ultralytics to use model detection")
            return False
        self.status(f"loading {Path(path).name}…")
        QApplication.processEvents()
        try:
            self.model = YOLO(path)
        except Exception as e:                                  # noqa: BLE001
            QMessageBox.critical(self, "Load failed", str(e))
            return False
        self.model_path = path
        self.model_label.setText(f"<b>{Path(path).name}</b>")
        names = getattr(self.model, "names", None)
        # adopt the model's class names only if the user has none of their own
        if names and self.classes in ([], ["class0"]):
            self.classes = [names[i] for i in sorted(names)]
            self.refresh_class_combo()
            self.reload_current(fit=False)
        self.status(f"model loaded: {Path(path).name}")
        return True

    def _predict(self, frame: Frame) -> list[Box]:
        res = self.model.predict(str(frame.path), conf=self.conf, verbose=False)[0]
        out = []
        for b in res.boxes:
            x1, y1, x2, y2 = (float(v) for v in b.xyxy[0].tolist())
            out.append(Box(int(b.cls.item()), x1, y1, x2, y2,
                           conf=float(b.conf.item())))
        return out

    def detect_current(self):
        f = self.cur_frame()
        if not f:
            return
        if not self.ensure_model():
            return
        self.status("detecting…")
        QApplication.processEvents()
        try:
            new = self._predict(f)
        except Exception as e:                                  # noqa: BLE001
            QMessageBox.critical(self, "Inference failed", str(e))
            return
        upd, add = merge_detections(f.boxes, new)
        if f.boxes:
            self.ensure_classes(max(b.cls for b in f.boxes) + 1)
        self.mark_dirty()
        self._save(f)                      # straight to labels/<stem>.txt
        self.reload_current(fit=False)
        self.status(f"{len(new)} detections: {add} new, {upd} updated → "
                    f"{self.label_path(f)}")

    MODES = ["Only images with no label file (skip the rest)",
             "All images — replace any existing boxes",
             "All images — merge into existing boxes (update overlaps, "
             "no duplicates)"]

    def detect_all(self):
        """Load the picked model and run it over the whole folder."""
        if not self.frames:
            QMessageBox.information(self, "No images", "Open an images folder first.")
            return
        if not self.load_selected_model():
            return

        mode, ok = QInputDialog.getItem(
            self, "Detect on all images",
            f"{len(self.frames)} images · model "
            f"{Path(self.model_path).name} · conf {self.conf:.2f}\n"
            f"Labels are written to:\n{self.labels_dir}\n\nRun on:",
            self.MODES, 0, False)
        if not ok:
            return
        m = self.MODES.index(mode)
        todo = [f for f in self.frames
                if m > 0 or not self.label_path(f).exists()]
        if not todo:
            self.status("every image already has a label file")
            return

        # save whatever is pending so nothing is silently overwritten
        if self.idx >= 0 and self.frames[self.idx].dirty:
            self._save(self.frames[self.idx])

        dlg = QProgressDialog(f"Detecting on {len(todo)} images…",
                              "Stop", 0, len(todo), self)
        dlg.setWindowTitle("Batch detection")
        dlg.setMinimumWidth(430)
        dlg.setWindowModality(Qt.ApplicationModal)
        dlg.setMinimumDuration(0)
        dlg.setValue(0)

        n_boxes = done = 0
        for i, f in enumerate(todo, 1):
            if dlg.wasCanceled():
                break
            dlg.setLabelText(f"{i}/{len(todo)}   {f.path.name}\n"
                             f"{n_boxes} boxes so far")
            QApplication.processEvents()
            try:
                new = self._predict(f)
            except Exception as e:                              # noqa: BLE001
                dlg.close()
                QMessageBox.critical(self, "Inference failed", str(e))
                break
            if m == 2:
                if not f.loaded:
                    pm = QPixmap(str(f.path))
                    self.load_labels(f, pm.width(), pm.height())
                merge_detections(f.boxes, new)
            else:
                f.boxes = new
            f.loaded, f.dirty = True, True
            self._save(f)
            n_boxes += len(new)
            done = i
            dlg.setValue(i)
        dlg.close()

        if self.frames:
            self.ensure_classes(1)
            names = getattr(self.model, "names", None)
            if names and len(self.classes) < len(names):
                self.classes = [names[k] for k in sorted(names)]
                self.refresh_class_combo()
        self.reload_current(fit=False)
        for f in self.frames:
            self.refresh_file_row(f)
        self.status(f"detected {n_boxes} boxes across {done} image(s) → "
                    f"{self.labels_dir}")
        QMessageBox.information(
            self, "Batch detection finished",
            f"{done} of {len(todo)} images processed\n"
            f"{n_boxes} boxes written to {self.labels_dir}"
            + ("\n\nStopped early." if done < len(todo) else ""))

    # -- one-button mining ------------------------------------------------
    def mine_dataset(self):
        """① — detect people, keep whole bodies, prepare them for review.

        With a folder already open this runs over those frames in place; only
        an empty session asks where the footage is.
        """
        if self._miner and self._miner.isRunning():
            QMessageBox.information(self, "Already running",
                                    "A mining run is in progress.")
            return
        if self.frames:
            box = QMessageBox(self)
            box.setWindowTitle("Build person crops")
            box.setText(f"Run on the {len(self.frames)} images already open?")
            box.setInformativeText(
                f"{self.images_dir}\n\nPeople are detected on every loaded "
                f"frame, boxes are widened to the full body, and labels are "
                f"written next to them — then you review and export.")
            b_here = box.addButton(f"Run on these {len(self.frames)} images",
                                   QMessageBox.AcceptRole)
            box.addButton("Pick another source…", QMessageBox.AcceptRole)
            box.addButton(QMessageBox.Cancel)
            box.setDefaultButton(b_here)
            box.exec()
            clicked = box.clickedButton()
            if clicked is None or box.buttonRole(clicked) == QMessageBox.RejectRole:
                return
            if clicked is b_here:
                self.detect_loaded_frames()
                return
        pick = QMessageBox(self)
        pick.setWindowTitle("Source")
        pick.setText("Where do the people come from?")
        b_vid = pick.addButton("Video file…", QMessageBox.AcceptRole)
        b_dir = pick.addButton("Folder (videos or frames)", QMessageBox.AcceptRole)
        pick.addButton(QMessageBox.Cancel)
        pick.exec()
        if pick.clickedButton() is b_vid:
            f, _ = QFileDialog.getOpenFileName(
                self, "Recording", "",
                "Video (*.mp4 *.mov *.mkv *.avi *.m4v *.webm *.ts);;All (*)")
            src = Path(f) if f else None
        elif pick.clickedButton() is b_dir:
            d = QFileDialog.getExistingDirectory(self, "Folder of videos or frames")
            src = Path(d) if d else None
        else:
            return
        if not src:
            return

        out = QFileDialog.getExistingDirectory(
            self, "Where to put the review set",
            str(src.parent if src.is_file() else src))
        if not out:
            return
        out = Path(out)
        if out.resolve() == (src if src.is_dir() else src.parent).resolve():
            out = out / f"{src.stem}_review"

        budget, ok = QInputDialog.getInt(
            self, "Budget", "How many frames to keep?", 400, 1, 100000, 50)
        if not ok:
            return
        per_track, ok = QInputDialog.getInt(
            self, "Per person", "Max crops from one tracked person:", 6, 1, 200, 1)
        if not ok:
            return
        partial = QMessageBox.question(
            self, "Full bodies only?",
            "Keep only people whose whole body is visible?\n\n"
            "Yes = full-body crops only (head, knees and an ankle must be "
            "detected, and the box must not touch the frame edge)\n"
            "No = also keep partial / truncated people",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes) == QMessageBox.No

        if not weight_ok(self.weights_dir / "yolo11x.pt"):
            QMessageBox.warning(
                self, "Detector missing",
                f"weights/yolo11x.pt is not in {self.weights_dir}.\n"
                f"Let the first-run download finish, or hit Re-download.")
            return
        pose = self.weights_dir / "yolo11x-pose.pt"
        if not weight_ok(pose) and not partial:
            if QMessageBox.question(
                    self, "Pose model missing",
                    f"{pose.name} is not downloaded, so full-body checking and "
                    f"pose diversity are unavailable.\n\nRun without pose?",
                    QMessageBox.Yes | QMessageBox.No) != QMessageBox.Yes:
                return

        dlg = QProgressDialog("Starting…", "Stop", 0, 0, self)
        dlg.setWindowTitle("Mining person crops")
        dlg.setMinimumWidth(480)
        dlg.setWindowModality(Qt.NonModal)
        dlg.setMinimumDuration(0)
        dlg.show()

        th = MinerThread(src, out, budget, per_track, self.weights_dir,
                         partial, self)
        self._miner = th

        def on_prog(msg):
            dlg.setLabelText(msg)
            self.mine_label.setText(msg)

        def on_done(out_dir, n_img, n_crops):
            dlg.close()
            self.mine_label.setText(
                f"{n_img} frames · {n_crops} person boxes → {out_dir}")
            self.open_images(Path(out_dir) / "images", None)
            QMessageBox.information(
                self, "Mining finished",
                f"{n_img} frames with {n_crops} person boxes written to\n"
                f"{out_dir}\n\nReview them now: Space approves, X rejects, "
                f"then ③ exports the crops.")

        def on_failed(err):
            dlg.close()
            self.mine_label.setText(f"mining failed: {err}")
            if err != "stopped":
                QMessageBox.critical(self, "Mining failed", err)

        th.progress.connect(on_prog)
        th.done.connect(on_done)
        th.failed.connect(on_failed)
        dlg.canceled.connect(th.stop)
        th.start()

    def person_class(self) -> int:
        """Class id to write person boxes under, without inventing extra ids."""
        for i, c in enumerate(self.classes):
            if c.lower() == "person":
                return i
        if self.classes in ([], ["class0"]):
            self.classes = ["person"]
        else:
            self.classes.append("person")
        self.refresh_class_combo()
        return self.classes.index("person")

    def detect_loaded_frames(self):
        """Run the FULL pipeline over the open frames, writing labels in place.

        Same selection as the video path — track, pose + appearance descriptor,
        per-track cap, farthest-point sampling — so what lands in the labels is
        one entry per distinct person/pose, not every detection.
        """
        if not weight_ok(self.weights_dir / "yolo11x.pt"):
            QMessageBox.warning(
                self, "Detector missing",
                f"weights/yolo11x.pt is not in {self.weights_dir}.")
            return
        pose_ok = weight_ok(self.weights_dir / POSE_WEIGHT)

        budget, ok = QInputDialog.getInt(
            self, "How many people to keep?",
            f"{len(self.frames)} frames are open.\n\n"
            f"Maximum person crops to keep (the most varied ones win):",
            min(200, max(10, len(self.frames) * 2)), 1, 100000, 10)
        if not ok:
            return
        per_track, ok = QInputDialog.getInt(
            self, "Per person",
            "Max crops of one tracked person:", 4, 1, 200, 1)
        if not ok:
            return
        partial = QMessageBox.question(
            self, "Full bodies only?",
            "Keep only people whose whole body is visible?\n\n"
            "Yes = skip truncated people\nNo = keep partial bodies too"
            + ("" if pose_ok else
               f"\n\n({POSE_WEIGHT} is missing — pose diversity and the "
               f"full-body check will be skipped)"),
            QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes) == QMessageBox.No

        if any(f.dirty for f in self.frames):
            if QMessageBox.question(
                    self, "Overwrite labels?",
                    "Selected boxes replace the labels of the open frames.\n"
                    "Unsaved edits are saved first, then overwritten. Go on?",
                    QMessageBox.Yes | QMessageBox.No) != QMessageBox.Yes:
                return
        if self.idx >= 0 and self.frames[self.idx].dirty:
            self._save(self.frames[self.idx])

        dlg = QProgressDialog("Loading models…", "Stop", 0, 0, self)
        dlg.setWindowTitle("① Build person crops")
        dlg.setMinimumWidth(480)
        dlg.setWindowModality(Qt.NonModal)
        dlg.setMinimumDuration(0)
        dlg.show()

        th = MinerThread(self.images_dir, self.images_dir, budget, per_track,
                         self.weights_dir, partial, self, in_place=True)
        self._miner = th
        th.progress.connect(lambda m: (dlg.setLabelText(m),
                                       self.mine_label.setText(m)))
        th.picked_ready.connect(lambda picked: (dlg.close(),
                                                self.apply_picks(picked)))

        def on_failed(err):
            dlg.close()
            self.mine_label.setText(f"failed: {err}")
            if err != "stopped":
                QMessageBox.critical(self, "Mining failed", err)

        th.failed.connect(on_failed)
        dlg.canceled.connect(th.stop)
        th.start()

    def apply_picks(self, picked: list[dict]):
        """Write the selected people into the open frames' label files."""
        cls = self.person_class()
        by_name: dict[str, list[dict]] = defaultdict(list)
        for c in picked:
            by_name[Path(c["source"]).name].append(c)

        entries, n_box, n_frames = [], 0, 0
        for f in self.frames:
            got = by_name.get(f.path.name, [])
            f.boxes = [Box(cls, *c["box"], conf=c.get("conf")) for c in got]
            f.loaded, f.dirty = True, True
            self._save(f)
            if got:
                n_box += len(got)
                n_frames += 1
                entries.append(dict(
                    image=f.path.name, source=str(f.path), frame=0,
                    width=0, height=0,
                    boxes=[{k: c.get(k) for k in
                            ("track", "conf", "h_rel", "sharp", "occl",
                             "bright", "motion", "hard", "rank",
                             "full_body", "dhash")}
                           for c in got]))
            elif self.review.get(f.path.name) is None:
                # nothing distinctive here: mark it out of the way, reversible
                self.review[f.path.name] = "rejected"

        base = self.images_dir.parent if self.images_dir else Path(".")
        # keep the class names with the data, so reopening it later still says
        # "person" instead of falling back to a placeholder
        (base / "classes.txt").write_text("\n".join(self.classes) + "\n")
        (base / "manifest.json").write_text(json.dumps(dict(
            created=time.strftime("%Y-%m-%d %H:%M:%S"),
            images=n_frames, crops=n_box, in_place=True,
            args=dict(conf=self.conf, crop_margin=self.crop_margin),
            entries=entries), indent=1))
        self.save_review()
        self.load_review()
        self.reload_current(fit=False)
        for f in self.frames:
            self.refresh_file_row(f)
        self.update_review_ui()
        st = getattr(self._miner, "stats", {}) or {}
        skipped = len(self.frames) - n_frames
        self.mine_label.setText(
            f"{n_box} distinct people kept across {n_frames} frames"
            + (f" · {skipped} frames had nothing new" if skipped else ""))
        self.status(("① " + " → ".join(f"{k.replace('_', ' ')} {v}"
                                       for k, v in st.items()) + f"  ·  → {self.labels_dir}")
                    if st else f"{n_box} person boxes → {self.labels_dir}")
        QMessageBox.information(
            self, "Ready for review",
            f"{n_box} distinct person crop(s) kept, spread over "
            f"{n_frames} frame(s).\n"
            + (f"{skipped} frame(s) added nothing new and were marked "
               f"rejected (Clear mark undoes that).\n" if skipped else "")
            + (("\nFilter chain: " + "  ".join(
                f"{k.replace('_', ' ')} {v}" for k, v in st.items()) + "\n")
               if st else "")
            + f"\nLabels: {self.labels_dir}\n\nNow ② review "
              f"(Space approves, X rejects) and ③ export the crops.")

    def set_crop_margin(self, v: float):
        self.crop_margin = float(v)

    # -- manual review ----------------------------------------------------
    def review_file(self) -> Path:
        base = self.images_dir.parent if self.images_dir else Path(".")
        return base / "review.json"

    def load_review(self):
        self.manifest, self.review = None, {}
        if not self.images_dir:
            return
        for cand in (self.images_dir.parent / "manifest.json",
                     self.images_dir / "manifest.json"):
            if cand.exists():
                try:
                    self.manifest = json.loads(cand.read_text())
                except Exception:                               # noqa: BLE001
                    self.manifest = None
                break
        rf = self.review_file()
        if rf.exists():
            try:
                self.review = json.loads(rf.read_text()).get("status", {})
            except Exception:                                   # noqa: BLE001
                self.review = {}

    def save_review(self):
        if not self.images_dir:
            return
        self.review_file().write_text(json.dumps(
            {"images_dir": str(self.images_dir), "status": self.review}, indent=1))

    def counts(self) -> tuple[int, int, int]:
        a = sum(1 for f in self.frames if self.review.get(f.path.name) == "approved")
        r = sum(1 for f in self.frames if self.review.get(f.path.name) == "rejected")
        return a, r, len(self.frames) - a - r

    def set_review(self, status: str, advance=True):
        f = self.cur_frame()
        if not f:
            return
        if f.dirty:
            self._save(f)                      # box edits count as part of approval
        if status == "pending":
            self.review.pop(f.path.name, None)
        else:
            self.review[f.path.name] = status
        self.save_review()
        self.refresh_file_row(f)
        self.update_review_ui()
        if advance and status != "pending":
            self.next_pending()

    def approve_all(self):
        if not self.frames:
            return
        pend = [f for f in self.frames
                if self.review.get(f.path.name, "pending") == "pending"]
        if not pend or QMessageBox.question(
                self, "Approve all",
                f"Mark the {len(pend)} unreviewed frame(s) approved?") \
                != QMessageBox.Yes:
            return
        for f in pend:
            self.review[f.path.name] = "approved"
            self.refresh_file_row(f)
        self.save_review()
        self.update_review_ui()

    def next_pending(self):
        n = len(self.frames)
        for k in range(1, n + 1):
            j = (self.idx + k) % n
            if self.review.get(self.frames[j].path.name, "pending") == "pending":
                self.goto(j)
                return
        self.status("no unreviewed frames left — Ctrl+E exports the approved crops")

    def update_review_ui(self):
        a, r, p = self.counts()
        f = self.cur_frame()
        mine = self.review.get(f.path.name, "pending") if f else "—"
        mark = {"approved": "<b style='color:#2ecc71'>APPROVED</b>",
                "rejected": "<b style='color:#ff3838'>REJECTED</b>",
                "pending": "<b>unreviewed</b>"}.get(mine, mine)
        n_crops = sum(len(x.boxes) for x in self.frames
                      if self.review.get(x.path.name) == "approved" and x.loaded)
        self.review_label.setText(
            f"this frame: {mark}<br>{a} approved · {r} rejected · {p} to go"
            + (f"<br>{n_crops}+ crops queued" if n_crops else ""))
        self.cand_label.setText(self.candidate_info())

    def candidate_info(self) -> str:
        f = self.cur_frame()
        if not (self.manifest and f):
            return ""
        for e in self.manifest.get("entries", []):
            if e.get("image") == f.path.name:
                bits = [f"source {Path(e.get('source', '?')).name} · "
                        f"frame {e.get('frame')}"]
                for b in e.get("boxes", []):
                    bits.append(
                        f"track {b.get('track')} · conf {b.get('conf', 0):.2f} · "
                        f"h {b.get('h_rel', 0) * 100:.0f}% · "
                        f"occl {b.get('occl', 0):.2f}"
                        + ("  ⚠ hard" if b.get("hard") else ""))
                return "<br>".join(bits)
        return ""

    # -- crop export ------------------------------------------------------
    def export_crops(self):
        appr = [f for f in self.frames
                if self.review.get(f.path.name) == "approved"]
        if not appr:
            QMessageBox.information(
                self, "Nothing approved",
                "Approve some frames first (Space approves and jumps to the "
                "next unreviewed one).")
            return
        out = self.crops_out
        if out is None:
            d = QFileDialog.getExistingDirectory(
                self, "Save crops to",
                str((self.images_dir.parent if self.images_dir else Path(".")) / "crops"))
            if not d:
                return
            out = Path(d)
        out.mkdir(parents=True, exist_ok=True)

        dlg = QProgressDialog(f"Cropping {len(appr)} approved frame(s)…",
                              "Stop", 0, len(appr), self)
        dlg.setWindowTitle("Export crops")
        dlg.setWindowModality(Qt.ApplicationModal)
        dlg.setMinimumDuration(0)

        rows, n = [], 0
        for i, f in enumerate(appr, 1):
            if dlg.wasCanceled():
                break
            dlg.setValue(i - 1)
            dlg.setLabelText(f"{i}/{len(appr)}  {f.path.name}  ({n} crops)")
            QApplication.processEvents()
            img = QImage(str(f.path))
            if img.isNull():
                continue
            iw, ih = img.width(), img.height()
            if not f.loaded:
                self.load_labels(f, iw, ih)
            for j, b in enumerate(f.boxes):
                x1, y1 = min(b.x1, b.x2), min(b.y1, b.y2)
                x2, y2 = max(b.x1, b.x2), max(b.y1, b.y2)
                mx, my = (x2 - x1) * self.crop_margin, (y2 - y1) * self.crop_margin
                rx1 = int(round(max(0.0, x1 - mx))); ry1 = int(round(max(0.0, y1 - my)))
                rx2 = int(round(min(float(iw), x2 + mx)))
                ry2 = int(round(min(float(ih), y2 + my)))
                if rx2 - rx1 < 8 or ry2 - ry1 < 8:
                    continue
                crop = img.copy(rx1, ry1, rx2 - rx1, ry2 - ry1)
                name = f"{f.path.stem}_{self.class_name(b.cls)}{j:02d}.jpg"
                if not crop.save(str(out / name), "JPG", 95):
                    continue
                rows.append(f"{name},{f.path.name},{self.class_name(b.cls)},"
                            f"{b.cls},{rx1},{ry1},{rx2},{ry2},{iw},{ih}")
                n += 1
        dlg.close()

        (out / "crops.csv").write_text(
            "crop,frame,class_name,class_id,x1,y1,x2,y2,frame_w,frame_h\n"
            + "\n".join(rows) + "\n")
        self.status(f"exported {n} crop(s) → {out}")
        QMessageBox.information(
            self, "Crops exported",
            f"{n} crop(s) from {len(appr)} approved frame(s)\n"
            f"written to {out}\n\nIndex: crops.csv")

    # -- misc -------------------------------------------------------------
    def status(self, msg: str):
        self.statusBar().showMessage(msg)

    def closeEvent(self, ev):
        if self._miner and self._miner.isRunning():
            self._miner.stop()
            self._miner.wait(5000)
        if self._dl and self._dl.isRunning():
            self._dl.cancel()
            self._dl.wait(3000)
        n = sum(1 for f in self.frames if f.dirty)
        if n:
            r = QMessageBox.question(
                self, "Unsaved", f"{n} image(s) have unsaved boxes. Save now?",
                QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel)
            if r == QMessageBox.Cancel:
                ev.ignore()
                return
            if r == QMessageBox.Yes:
                self.save_all()
        ev.accept()


def main():
    ap = argparse.ArgumentParser(description="Lightweight YOLO bbox annotator")
    ap.add_argument("--images", help="images folder")
    ap.add_argument("--labels", help="YOLO labels folder (default: guessed)")
    ap.add_argument("--classes", help="classes.txt (one name per line)")
    ap.add_argument("--model", help="YOLO weights for detection")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--weights-dir", help=f"where weights live (default {WEIGHTS_DIR})")
    ap.add_argument("--crops-out", help="skip the folder prompt on crop export")
    ap.add_argument("--crop-margin", type=float, default=0.25,
                    help="padding added on every side when cropping a person")
    ap.add_argument("--organize", choices=["ask", "always", "never"],
                    default="ask",
                    help="a folder of loose images becomes <folder>/images + "
                         "<folder>/labels (default: ask)")
    ap.add_argument("--no-download", action="store_true",
                    help="skip the first-run download of "
                         + ", ".join(DEFAULT_WEIGHTS))
    args = ap.parse_args()

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    w = MainWindow(args)
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
