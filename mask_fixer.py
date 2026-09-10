#!/usr/bin/env python3
"""Iteratively repair pipeline mattes with SAM 2.1 (or SAM 3).

BiRefNet drops thin, low-contrast parts — legs against a dark pitch, an arm
over a bright hoarding. This opens each crop next to its matte and lets you put
the missing part back with a prompt instead of by hand:

    press W (whole person)        SAM finds the whole body, Add unions it in
    click on the legs             a point prompt, Add unions just that part
    type "person" -> Segment      text prompts, SAM 3 only
    right-click on a bad blob     a negative point, Subtract removes it
    brush                         for the last few pixels

Nothing is written until you press Save, and saves go to a separate folder, so
the pipeline's own masks/ stays untouched.

    ./run_mask_fixer.sh [config.yaml]
"""
from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

import numpy as np
from PIL import Image

from PySide6.QtCore import Qt, QPoint, QRect, QSize
from PySide6.QtGui import (QAction, QBrush, QColor, QCursor, QImage, QKeySequence,
                           QPainter, QPen, QPixmap)
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDoubleSpinBox,
                               QFileDialog, QFrame, QHBoxLayout, QLabel,
                               QLineEdit, QListWidget, QListWidgetItem,
                               QMainWindow, QMessageBox, QPushButton, QSlider,
                               QSpinBox, QSplitter, QVBoxLayout, QWidget)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from sam_backend import SamError, combine, make_backend       # noqa: E402

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
UNDO_DEPTH = 24

DEFAULTS = {
    "images": "dataset_out/crops",
    "masks": "dataset_out/masks",
    "out": "",
    "model": "sam2",
    "sam2_repo": "facebook/sam2.1-hiera-large",
    "sam2_weights": "weights/sam2.1-hiera-large",
    "sam3_repo": "facebook/sam3",
    "sam3_weights": "weights/sam3",
    "device": "",
    "half": None,
    "text_prompt": "person",
    "det_threshold": 0.30,
    "mask_threshold": 0.50,
    "overlay_alpha": 0.55,
    "brush_size": 24,
    "save_binary": False,
}


# ------------------------------------------------------------------- config

def load_config(path: Path) -> dict:
    """YAML or JSON, whichever the file is. Unknown keys are left alone."""
    text = path.read_text()
    if path.suffix.lower() in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError:
            sys.exit("PyYAML is not installed — use a .json config, or "
                     "pip install pyyaml")
        cfg = yaml.safe_load(text) or {}
    else:
        cfg = json.loads(text)
    if not isinstance(cfg, dict):
        sys.exit(f"{path}: expected a mapping of settings")

    merged = dict(DEFAULTS)
    merged.update(cfg)
    base = path.resolve().parent
    for key in ("images", "masks", "out", "sam2_weights", "sam3_weights"):
        if merged.get(key):
            p = Path(str(merged[key])).expanduser()
            merged[key] = p if p.is_absolute() else (base / p)
    if not merged["out"]:
        merged["out"] = Path(str(merged["masks"])) .parent / (
            Path(str(merged["masks"])).name + "_fixed")
    return merged


def pair_files(images: Path, masks: Path) -> list[tuple[Path, Path | None]]:
    """Match image to mask by stem; a missing mask is fine (start from empty)."""
    if not images.is_dir():
        sys.exit(f"images folder does not exist: {images}")
    by_stem = {}
    if masks.is_dir():
        for m in masks.iterdir():
            if m.is_file() and m.suffix.lower() in IMG_EXT:
                by_stem[m.stem] = m
    out = []
    for p in sorted(images.iterdir()):
        if p.is_file() and p.suffix.lower() in IMG_EXT:
            out.append((p, by_stem.get(p.stem)))
    return out


# ------------------------------------------------------------------- canvas

class Canvas(QLabel):
    """Image + mask overlay. Left click = keep, right click = drop, drag = box."""

    def __init__(self, owner: "MaskFixer"):
        super().__init__()
        self.owner = owner
        self.setMinimumSize(QSize(480, 480))
        self.setAlignment(Qt.AlignCenter)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self._scale = 1.0
        self._off = QPoint(0, 0)
        self._drag_from: QPoint | None = None
        self._drag_to: QPoint | None = None
        self._painting = False

    # -- coordinate mapping: widget px <-> image px
    def to_image(self, pos: QPoint) -> tuple[int, int] | None:
        if self._scale <= 0 or self.owner.rgb is None:
            return None
        x = (pos.x() - self._off.x()) / self._scale
        y = (pos.y() - self._off.y()) / self._scale
        h, w = self.owner.rgb.shape[:2]
        if 0 <= x < w and 0 <= y < h:
            return int(x), int(y)
        return None

    def paintEvent(self, ev):                                   # noqa: N802
        super().paintEvent(ev)
        o = self.owner
        if o.rgb is None:
            return
        h, w = o.rgb.shape[:2]
        self._scale = min(self.width() / w, self.height() / h)
        dw, dh = int(w * self._scale), int(h * self._scale)
        self._off = QPoint((self.width() - dw) // 2, (self.height() - dh) // 2)

        p = QPainter(self)
        p.setRenderHint(QPainter.SmoothPixmapTransform)
        p.drawPixmap(QRect(self._off.x(), self._off.y(), dw, dh), o.composite())

        # prompts, drawn on top so they stay visible over the tint
        for (px, py, lab) in o.points:
            c = QColor(60, 230, 90) if lab == 1 else QColor(255, 70, 70)
            p.setPen(QPen(QColor(0, 0, 0), 2))
            p.setBrush(QBrush(c))
            sx = self._off.x() + px * self._scale
            sy = self._off.y() + py * self._scale
            p.drawEllipse(QPoint(int(sx), int(sy)), 6, 6)
        if self._drag_from and self._drag_to:
            p.setPen(QPen(QColor(80, 170, 255), 2, Qt.DashLine))
            p.setBrush(Qt.NoBrush)
            p.drawRect(QRect(self._drag_from, self._drag_to).normalized())
        elif o.box is not None:
            x1, y1, x2, y2 = o.box
            p.setPen(QPen(QColor(80, 170, 255), 2, Qt.DashLine))
            p.setBrush(Qt.NoBrush)
            p.drawRect(QRect(int(self._off.x() + x1 * self._scale),
                             int(self._off.y() + y1 * self._scale),
                             int((x2 - x1) * self._scale),
                             int((y2 - y1) * self._scale)))
        p.end()

    # -- mouse
    def mousePressEvent(self, ev):                              # noqa: N802
        o = self.owner
        if o.rgb is None:
            return
        pt = self.to_image(ev.position().toPoint())
        if pt is None:
            return
        if o.brush_on.isChecked():
            o.push_undo()
            self._painting = True
            o.paint_at(pt, erase=ev.button() == Qt.RightButton)
        elif ev.button() == Qt.MiddleButton or (
                ev.button() == Qt.LeftButton and ev.modifiers() & Qt.ShiftModifier):
            self._drag_from = ev.position().toPoint()
            self._drag_to = self._drag_from
        elif ev.button() == Qt.LeftButton:
            o.points.append((pt[0], pt[1], 1))
            o.run_points()
        elif ev.button() == Qt.RightButton:
            o.points.append((pt[0], pt[1], 0))
            o.run_points()
        self.update()

    def mouseMoveEvent(self, ev):                               # noqa: N802
        o = self.owner
        if self._painting:
            pt = self.to_image(ev.position().toPoint())
            if pt:
                o.paint_at(pt, erase=ev.buttons() & Qt.RightButton)
        elif self._drag_from is not None:
            self._drag_to = ev.position().toPoint()
        self.update()

    def mouseReleaseEvent(self, ev):                            # noqa: N802
        o = self.owner
        if self._painting:
            self._painting = False
            o.refresh()
            return
        if self._drag_from is not None and self._drag_to is not None:
            a = self.to_image(self._drag_from)
            b = self.to_image(self._drag_to)
            self._drag_from = self._drag_to = None
            if a and b and abs(a[0] - b[0]) > 4 and abs(a[1] - b[1]) > 4:
                o.box = [min(a[0], b[0]), min(a[1], b[1]),
                         max(a[0], b[0]), max(a[1], b[1])]
                o.run_points()
        self.update()

    def wheelEvent(self, ev):                                   # noqa: N802
        if self.owner.brush_on.isChecked():
            step = 2 if ev.angleDelta().y() > 0 else -2
            self.owner.brush_px.setValue(self.owner.brush_px.value() + step)
            ev.accept()


# ---------------------------------------------------------------- main window

class MaskFixer(QMainWindow):
    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        self.setWindowTitle("SAM mask fixer")
        self.resize(1360, 880)

        self.images_dir = Path(cfg["images"])
        self.masks_dir = Path(cfg["masks"])
        self.out_dir = Path(cfg["out"])
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.pairs = pair_files(self.images_dir, self.masks_dir)
        if not self.pairs:
            sys.exit(f"no images in {self.images_dir}")

        self.backend = make_backend(cfg)
        self.rgb: np.ndarray | None = None
        self.mask: np.ndarray | None = None
        self.pil: Image.Image | None = None
        self.last_pred: np.ndarray | None = None
        self.points: list[tuple[int, int, int]] = []
        self.box: list[int] | None = None
        self.undo: list[np.ndarray] = []
        self.idx = -1

        self._build_ui()
        self.load(0)

    # ------------------------------------------------------------------ ui
    def _build_ui(self):
        split = QSplitter()

        self.files = QListWidget()
        self.files.setMinimumWidth(240)
        for img, m in self.pairs:
            tag = "" if m else "  (no mask)"
            self.files.addItem(QListWidgetItem(img.name + tag))
        self.files.currentRowChanged.connect(self.load)
        split.addWidget(self.files)

        self.canvas = Canvas(self)
        split.addWidget(self.canvas)

        side = QWidget()
        side.setMinimumWidth(300)
        v = QVBoxLayout(side)

        def section(text):
            lab = QLabel(f"<b>{text}</b>")
            v.addWidget(lab)

        section(f"Prompt  ({self.backend.label})")
        b = QPushButton("Whole person  (W)")
        b.setToolTip("Points down the body's centre line with the corners as "
                     "negatives — the usual one-press fix for missing legs")
        b.clicked.connect(self.run_whole)
        v.addWidget(b)

        self.prompt = QLineEdit(str(self.cfg["text_prompt"]))
        self.prompt.returnPressed.connect(self.run_text)
        v.addWidget(self.prompt)
        row = QHBoxLayout()
        self.text_btn = QPushButton("Segment text  (T)")
        self.text_btn.clicked.connect(self.run_text)
        row.addWidget(self.text_btn)
        self.det_thr = QDoubleSpinBox()
        self.det_thr.setRange(0.01, 0.99)
        self.det_thr.setSingleStep(0.05)
        self.det_thr.setValue(float(self.cfg["det_threshold"]))
        self.det_thr.setPrefix("score ")
        row.addWidget(self.det_thr)
        v.addLayout(row)
        if not self.backend.supports_text:
            why = (f"{self.backend.label} has no text encoder. Use Whole "
                   f"person, clicks or a box — or set model: sam3 in the "
                   f"config once its licence is granted.")
            for wdg in (self.prompt, self.text_btn, self.det_thr):
                wdg.setEnabled(False)
                wdg.setToolTip(why)
            self.prompt.setText("(text needs SAM 3)")
        v.addWidget(QLabel("Left click = keep · right click = drop\n"
                           "Shift-drag or middle-drag = box"))

        section("Apply result as")
        self.op = QComboBox()
        self.op.addItems(["add", "subtract", "replace", "intersect"])
        v.addWidget(self.op)
        row = QHBoxLayout()
        for text, fn in (("Apply  (Enter)", self.apply_pred),
                         ("Clear prompts  (C)", self.clear_prompts)):
            b = QPushButton(text)
            b.clicked.connect(fn)
            row.addWidget(b)
        v.addLayout(row)
        self.auto_apply = QCheckBox("apply automatically after each prompt")
        self.auto_apply.setChecked(True)
        v.addWidget(self.auto_apply)

        section("Brush")
        row = QHBoxLayout()
        self.brush_on = QCheckBox("brush (B)")
        row.addWidget(self.brush_on)
        self.brush_px = QSpinBox()
        self.brush_px.setRange(2, 400)
        self.brush_px.setValue(int(self.cfg["brush_size"]))
        self.brush_px.setSuffix(" px")
        row.addWidget(self.brush_px)
        v.addLayout(row)
        v.addWidget(QLabel("left = paint · right = erase · wheel = size"))

        section("View")
        self.alpha = QSlider(Qt.Horizontal)
        self.alpha.setRange(0, 100)
        self.alpha.setValue(int(float(self.cfg["overlay_alpha"]) * 100))
        self.alpha.valueChanged.connect(self.canvas.update)
        v.addWidget(self.alpha)
        self.view = QComboBox()
        self.view.addItems(["overlay", "mask only", "image only", "cutout"])
        self.view.currentIndexChanged.connect(self.canvas.update)
        v.addWidget(self.view)

        section("Save")
        self.binary = QCheckBox("binarize on save")
        self.binary.setChecked(bool(self.cfg["save_binary"]))
        v.addWidget(self.binary)
        self.mask_thr = QDoubleSpinBox()
        self.mask_thr.setRange(0.01, 0.99)
        self.mask_thr.setSingleStep(0.05)
        self.mask_thr.setValue(float(self.cfg["mask_threshold"]))
        self.mask_thr.setPrefix("threshold ")
        v.addWidget(self.mask_thr)
        for text, fn in (("Save  (Ctrl+S)", self.save),
                         ("Save and next  (Ctrl+Enter)", self.save_next),
                         ("Undo  (Ctrl+Z)", self.undo_one),
                         ("Reset to original  (Ctrl+R)", self.reset)):
            b = QPushButton(text)
            b.clicked.connect(fn)
            v.addWidget(b)

        v.addStretch(1)
        self.info = QLabel("")
        self.info.setWordWrap(True)
        self.info.setFrameShape(QFrame.StyledPanel)
        v.addWidget(self.info)
        split.addWidget(side)

        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)
        split.setStretchFactor(2, 0)
        self.setCentralWidget(split)
        self.status = self.statusBar()

        for key, fn in (("T", self.run_text), ("W", self.run_whole),
                        ("C", self.clear_prompts),
                        ("B", lambda: self.brush_on.toggle()),
                        ("Return", self.apply_pred),
                        ("Ctrl+S", self.save), ("Ctrl+Return", self.save_next),
                        ("Ctrl+Z", self.undo_one), ("Ctrl+R", self.reset),
                        ("Right", lambda: self.step(1)),
                        ("Left", lambda: self.step(-1))):
            act = QAction(self)
            act.setShortcut(QKeySequence(key))
            act.triggered.connect(fn)
            self.addAction(act)

    # ------------------------------------------------------------ rendering
    def composite(self) -> QPixmap:
        """The view the canvas draws: image, tinted by the mask."""
        rgb = self.rgb
        m = self.mask
        mode = self.view.currentText()
        a = self.alpha.value() / 100.0

        if mode == "image only":
            out = rgb.copy()
        elif mode == "mask only":
            out = (m[..., None].repeat(3, 2) * 255).astype(np.uint8)
        elif mode == "cutout":
            out = (rgb.astype(np.float32) * m[..., None]).astype(np.uint8)
        else:
            tint = np.zeros_like(rgb, np.float32)
            tint[..., 1] = 255.0                       # green where kept
            w = (m * a)[..., None]
            out = (rgb.astype(np.float32) * (1 - w) + tint * w).astype(np.uint8)
            # a thin red wash marks what the mask drops, so gaps stand out
            drop = ((1 - m) * a * 0.30)[..., None]
            red = np.zeros_like(rgb, np.float32)
            red[..., 0] = 255.0
            out = (out.astype(np.float32) * (1 - drop) + red * drop).astype(np.uint8)

        h, w = out.shape[:2]
        img = QImage(np.ascontiguousarray(out).data, w, h, 3 * w,
                     QImage.Format_RGB888)
        return QPixmap.fromImage(img.copy())

    def refresh(self):
        self.canvas.update()
        n_on = float(self.mask.mean()) if self.mask is not None else 0
        img, mp = self.pairs[self.idx]
        saved = (self.out_dir / f"{img.stem}.png").exists()
        self.info.setText(
            f"<b>{img.name}</b><br>{self.rgb.shape[1]}x{self.rgb.shape[0]} px"
            f"<br>mask covers {n_on * 100:.1f}% of the crop"
            f"<br>source matte: {'yes' if mp else 'none — started empty'}"
            f"<br>saved copy: {'yes' if saved else 'not yet'}"
            f"<br>undo depth: {len(self.undo)}"
            f"<br>device: {self.backend.device}")

    # --------------------------------------------------------------- files
    def load(self, row: int):
        if row < 0 or row >= len(self.pairs):
            return
        self.idx = row
        img_p, mask_p = self.pairs[row]
        bgr = Image.open(img_p).convert("RGB")
        self.pil = bgr
        self.rgb = np.asarray(bgr, np.uint8)
        h, w = self.rgb.shape[:2]

        def read_mask(src) -> np.ndarray | None:
            if src is None or not Path(src).exists():
                return None
            m = Image.open(src).convert("L")
            if m.size != (w, h):
                m = m.resize((w, h), Image.BILINEAR)
            return np.asarray(m, np.float32) / 255.0

        # "original" always means the matte the pipeline produced, so Reset
        # still works after a save; the working copy resumes from the save.
        pipeline = read_mask(mask_p)
        self.original = (pipeline if pipeline is not None
                         else np.zeros((h, w), np.float32))
        saved = read_mask(self.out_dir / f"{img_p.stem}.png")
        self.mask = (saved if saved is not None else self.original).copy()

        self.undo.clear()
        self.clear_prompts()
        if self.files.currentRow() != row:
            self.files.setCurrentRow(row)
        self.refresh()

    def step(self, d: int):
        self.load(max(0, min(len(self.pairs) - 1, self.idx + d)))

    # ------------------------------------------------------------- editing
    def push_undo(self):
        self.undo.append(self.mask.copy())
        del self.undo[:-UNDO_DEPTH]

    def undo_one(self):
        if self.undo:
            self.mask = self.undo.pop()
            self.refresh()
        else:
            self.status.showMessage("nothing to undo", 2000)

    def reset(self):
        """Back to the pipeline's own matte, discarding this session's edits."""
        self.push_undo()
        self.mask = self.original.copy()
        self.clear_prompts()
        self.refresh()

    def clear_prompts(self):
        self.points.clear()
        self.box = None
        self.last_pred = None
        self.canvas.update()

    def paint_at(self, pt, erase: bool):
        x, y = pt
        r = self.brush_px.value() // 2
        h, w = self.mask.shape
        y0, y1 = max(0, y - r), min(h, y + r + 1)
        x0, x1 = max(0, x - r), min(w, x + r + 1)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        inside = (yy - y) ** 2 + (xx - x) ** 2 <= r * r
        patch = self.mask[y0:y1, x0:x1]
        patch[inside] = 0.0 if erase else 1.0
        self.canvas.update()

    # -------------------------------------------------------------- prompts
    def _guard(self, fn, *a, **kw):
        """Run a model call with a wait cursor and a readable failure."""
        QApplication.setOverrideCursor(QCursor(Qt.WaitCursor))
        self.status.showMessage(f"{self.backend.label} working …")
        QApplication.processEvents()
        try:
            return fn(*a, **kw)
        except SamError as e:
            QMessageBox.warning(self, "SAM", str(e))
        except Exception:                                       # noqa: BLE001
            QMessageBox.critical(self, f"{self.backend.label} failed",
                                 traceback.format_exc(limit=4))
        finally:
            QApplication.restoreOverrideCursor()
            self.status.clearMessage()
        return None

    def run_whole(self):
        if self.rgb is None:
            return
        pred = self._guard(self.backend.segment_whole, self.pil)
        self._took(pred, "whole person")

    def run_text(self):
        if self.rgb is None:
            return
        pred = self._guard(self.backend.segment_text, self.pil,
                           self.prompt.text().strip(),
                           float(self.det_thr.value()),
                           [self.box] if self.box else None)
        self._took(pred, f"text {self.prompt.text().strip()!r}")

    def run_points(self):
        if self.rgb is None or (not self.points and self.box is None):
            return
        pts = [(x, y) for x, y, _ in self.points]
        labs = [lab for _, _, lab in self.points]
        pred = self._guard(self.backend.segment_points, self.pil, pts, labs,
                           self.box)
        self._took(pred, f"{len(pts)} point(s)"
                         + (" + box" if self.box else ""))

    def _took(self, pred, what: str):
        if pred is None:
            return
        if pred.shape != self.mask.shape:
            QMessageBox.warning(self, "SAM",
                                f"prediction {pred.shape} does not match the "
                                f"mask {self.mask.shape}")
            return
        self.last_pred = pred
        cover = float(pred.mean()) * 100
        self.status.showMessage(f"{what}: covers {cover:.1f}% — "
                                f"{self.op.currentText()} it with Enter", 4000)
        if self.auto_apply.isChecked():
            self.apply_pred()
        else:
            self.canvas.update()

    def apply_pred(self):
        if self.last_pred is None:
            self.status.showMessage("no SAM 3 result to apply yet", 2000)
            return
        self.push_undo()
        self.mask = combine(self.mask, self.last_pred, self.op.currentText())
        self.last_pred = None
        self.points.clear()
        self.box = None
        self.refresh()

    # ---------------------------------------------------------------- save
    def save(self) -> bool:
        if self.mask is None:
            return False
        m = self.mask
        if self.binary.isChecked():
            m = (m >= float(self.mask_thr.value())).astype(np.float32)
        p = self.out_dir / f"{self.pairs[self.idx][0].stem}.png"
        Image.fromarray((m * 255).clip(0, 255).astype(np.uint8),
                        mode="L").save(p)
        self.status.showMessage(f"saved {p}", 4000)
        item = self.files.item(self.idx)
        if not item.text().startswith("✔"):
            item.setText("✔ " + item.text())
        self.refresh()
        return True

    def save_next(self):
        if self.save():
            self.step(1)


def main() -> int:
    cfg_path = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "mask_fixer.yaml"
    if not cfg_path.exists():
        app = QApplication(sys.argv)
        chosen, _ = QFileDialog.getOpenFileName(
            None, "Pick a mask_fixer config", str(HERE),
            "Config (*.yaml *.yml *.json)")
        if not chosen:
            sys.exit(f"no config file: {cfg_path}")
        cfg_path = Path(chosen)
    else:
        app = QApplication(sys.argv)

    win = MaskFixer(load_config(cfg_path))
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
