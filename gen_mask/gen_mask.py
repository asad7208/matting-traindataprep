#!/usr/bin/env python3
"""PySide6 mask generator for folders of images.

Pick a folder of images, pick a matting model (BiRefNet or RMBG-2.0), and
generate soft/binary masks either for the image on screen or for the whole
folder. The original is shown on the left (with an optional mask overlay) and
the mask itself on the right; both views share zoom and pan.

Masks are written to a folder next to the image folder, named "mask" by
default (--out relocates it), one PNG per image keeping the original stem:

    <images>/frame_000123.jpg  ->  <images>/../mask/frame_000123.png

    python gen_mask.py [--images IMGS] [--out MASKS] [--model birefnet]
"""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

import numpy as np
from PySide6.QtCore import QObject, Qt, QThread, Signal
from PySide6.QtGui import QAction, QColor, QImage, QKeySequence, QPixmap
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDockWidget,
                               QFileDialog, QGraphicsPixmapItem, QGraphicsScene,
                               QGraphicsView, QHBoxLayout, QLabel, QListWidget,
                               QListWidgetItem, QMainWindow, QMessageBox,
                               QProgressBar, QPushButton, QSizePolicy, QSlider,
                               QSpinBox, QSplitter, QVBoxLayout, QWidget)

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Every model here is loaded through transformers' AutoModelForImageSegmentation
# with trust_remote_code. `norm` is the input normalization; `minmax` rescales
# the prediction to [0,1] by its own extremes instead of taking the sigmoid,
# which is what U2Net-style checkpoints need — both models here take the sigmoid.
MODELS = {
    "birefnet": {"repo": "ZhengPeng7/BiRefNet", "label": "BiRefNet (general)",
                 "norm": (IMAGENET_MEAN, IMAGENET_STD), "minmax": False},
    "rmbg2": {"repo": "briaai/RMBG-2.0", "label": "RMBG-2.0",
              "norm": (IMAGENET_MEAN, IMAGENET_STD), "minmax": False},
}
DEFAULT_SIZE = 1024

GATED_HINT = (
    "This model is gated on Hugging Face. Accept the licence at\n"
    "    https://huggingface.co/{repo}\n"
    "then log in once in a terminal:\n"
    "    hf auth login\n"
    "(or export HF_TOKEN=...) and try again."
)


def list_images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir()
                  if p.is_file() and p.suffix.lower() in IMG_EXT)


def default_out_dir(images: Path) -> Path:
    """Sibling of the image folder, named "mask"."""
    return images.parent / "mask"


# --------------------------------------------------------------------------- #
# inference
# --------------------------------------------------------------------------- #
class Matter:
    """Lazily loaded matting model. Lives entirely inside the worker thread."""

    def __init__(self) -> None:
        self.key: str | None = None
        self.model = None
        self.device = "cpu"
        self.half = False

    def load(self, key: str) -> None:
        if self.key == key and self.model is not None:
            return
        import torch
        from transformers import AutoModelForImageSegmentation

        repo = MODELS[key]["repo"]
        torch.set_float32_matmul_precision("high")
        model = AutoModelForImageSegmentation.from_pretrained(
            repo, trust_remote_code=True)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.half = self.device == "cuda"
        model = model.to(self.device)
        if self.half:
            model = model.half()
        model.eval()
        self.model, self.key = model, key

    def infer(self, path: Path, size: int) -> np.ndarray:
        """Return the soft mask as float32 [0,1] at the image's own size."""
        import torch
        from PIL import Image

        with Image.open(path) as im:
            rgb = im.convert("RGB")
            w, h = rgb.size
            small = rgb.resize((size, size), Image.BILINEAR)

        mean, std = MODELS[self.key]["norm"]
        x = np.asarray(small, dtype=np.float32) / 255.0
        x = (x - np.array(mean, dtype=np.float32)) / np.array(std, np.float32)
        x = torch.from_numpy(x.transpose(2, 0, 1))[None]
        x = x.to(self.device)
        if self.half:
            x = x.half()

        with torch.no_grad():
            out = self.model(x)
        pred = out
        while isinstance(pred, (list, tuple)):        # heads, coarse -> fine
            pred = pred[-1] if not MODELS[self.key]["minmax"] else pred[0]
        pred = pred.float().cpu()[0, 0]
        if MODELS[self.key]["minmax"]:
            lo, hi = pred.min(), pred.max()
            pred = (pred - lo) / (hi - lo + 1e-8)
        else:
            pred = pred.sigmoid()

        mask = pred.clamp(0, 1).mul(255).byte().numpy()
        with Image.fromarray(mask, mode="L") as mim:
            mask = np.asarray(mim.resize((w, h), Image.BILINEAR))
        return mask.astype(np.float32) / 255.0


class Worker(QObject):
    """Processes a list of images on a background thread."""

    loading = Signal(str)                # model label being loaded
    done_one = Signal(str, object)       # path, float32 mask (H,W) in [0,1]
    failed = Signal(str, str)            # path, error text
    progress = Signal(int, int)          # done, total
    finished = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.matter = Matter()
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def run_batch(self, paths: list[str], key: str, size: int) -> None:
        self._cancel = False
        try:
            if self.matter.key != key or self.matter.model is None:
                self.loading.emit(MODELS[key]["label"])
            self.matter.load(key)
        except Exception as e:
            msg = f"Could not load {MODELS[key]['label']}.\n\n"
            if "gated repo" in str(e) or "401" in str(e):
                msg += GATED_HINT.format(repo=MODELS[key]["repo"])
            else:
                msg += traceback.format_exc(limit=3)
            self.failed.emit("", msg)
            self.finished.emit()
            return

        total = len(paths)
        for i, p in enumerate(paths, 1):
            if self._cancel:
                break
            try:
                self.done_one.emit(p, self.matter.infer(Path(p), size))
            except Exception:
                self.failed.emit(p, traceback.format_exc(limit=3))
            self.progress.emit(i, total)
        self.finished.emit()


# --------------------------------------------------------------------------- #
# views
# --------------------------------------------------------------------------- #
class ImageView(QGraphicsView):
    """Pannable, wheel-zoomable image view; zoom/pan mirrored onto a peer."""

    def __init__(self, title: str) -> None:
        super().__init__()
        self.setScene(QGraphicsScene(self))
        self.item = QGraphicsPixmapItem()
        self.scene().addItem(self.item)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setRenderHints(self.renderHints())
        self.setBackgroundBrush(QColor("#1e1e1e"))
        self.setMinimumSize(240, 240)
        self.title = title
        self.peer: ImageView | None = None
        self._syncing = False
        self._fitted = False

    def set_pixmap(self, pm: QPixmap | None, keep_view: bool) -> None:
        self.item.setPixmap(pm or QPixmap())
        if pm is None or pm.isNull():
            return
        self.setSceneRect(self.item.boundingRect())
        if not keep_view or not self._fitted:
            self.fit()

    def fit(self) -> None:
        if self.item.pixmap().isNull():
            return
        self.fitInView(self.item, Qt.KeepAspectRatio)
        self._fitted = True
        self._mirror()

    def wheelEvent(self, ev) -> None:
        if self.item.pixmap().isNull():
            return
        f = 1.25 if ev.angleDelta().y() > 0 else 0.8
        self.scale(f, f)
        self._mirror()

    def scrollContentsBy(self, dx: int, dy: int) -> None:
        super().scrollContentsBy(dx, dy)
        self._mirror()

    def _mirror(self) -> None:
        peer = self.peer
        if peer is None or self._syncing or peer.item.pixmap().isNull():
            return
        peer._syncing = True
        peer.setTransform(self.transform())
        peer.horizontalScrollBar().setValue(self.horizontalScrollBar().value())
        peer.verticalScrollBar().setValue(self.verticalScrollBar().value())
        peer._syncing = False


def to_qpixmap(arr: np.ndarray) -> QPixmap:
    """RGB uint8 (H,W,3) or gray uint8 (H,W) -> QPixmap (copies the buffer)."""
    arr = np.ascontiguousarray(arr)
    h, w = arr.shape[:2]
    if arr.ndim == 2:
        img = QImage(arr.data, w, h, w, QImage.Format_Grayscale8)
    else:
        img = QImage(arr.data, w, h, 3 * w, QImage.Format_RGB888)
    return QPixmap.fromImage(img.copy())


# --------------------------------------------------------------------------- #
# main window
# --------------------------------------------------------------------------- #
class MainWindow(QMainWindow):
    request_batch = Signal(list, str, int)

    def __init__(self, images: Path | None, out: Path | None,
                 model_key: str) -> None:
        super().__init__()
        self.setWindowTitle("gen_mask — BiRefNet / RMBG-2.0")
        self.resize(1500, 900)

        self.images_dir: Path | None = None
        self.out_dir: Path | None = out
        self.out_override = out is not None
        self.paths: list[Path] = []
        self.rgb: np.ndarray | None = None      # current image, uint8 RGB
        self.mask: np.ndarray | None = None     # current mask, float32 [0,1]
        self.masks: dict[str, np.ndarray] = {}  # path -> soft mask (session)

        self._build_ui(model_key)
        self._start_worker()

        if images:
            self.load_folder(images)

    # ---------------------------------------------------------------- ui ---
    def _build_ui(self, model_key: str) -> None:
        self.view_img = ImageView("original")
        self.view_mask = ImageView("mask")
        self.view_img.peer = self.view_mask
        self.view_mask.peer = self.view_img

        split = QSplitter(Qt.Horizontal)
        for v, cap in ((self.view_img, "Original / overlay"),
                       (self.view_mask, "Mask")):
            box = QWidget()
            lay = QVBoxLayout(box)
            lay.setContentsMargins(0, 0, 0, 0)
            lay.setSpacing(2)
            lab = QLabel(cap)
            lab.setAlignment(Qt.AlignCenter)
            lab.setStyleSheet("color:#bbb;padding:2px;")
            lay.addWidget(lab)
            lay.addWidget(v)
            split.addWidget(box)
        split.setSizes([750, 750])
        self.setCentralWidget(split)

        # --- top controls ---
        bar = self.addToolBar("main")
        bar.setMovable(False)

        act_open = QAction("Open folder…", self)
        act_open.setShortcut(QKeySequence.Open)
        act_open.triggered.connect(self.pick_folder)
        bar.addAction(act_open)

        act_out = QAction("Mask folder…", self)
        act_out.triggered.connect(self.pick_out)
        bar.addAction(act_out)
        bar.addSeparator()

        bar.addWidget(QLabel(" Model "))
        self.cmb_model = QComboBox()
        for key, spec in MODELS.items():
            self.cmb_model.addItem(spec["label"], key)
        idx = self.cmb_model.findData(model_key)
        self.cmb_model.setCurrentIndex(max(0, idx))
        bar.addWidget(self.cmb_model)

        bar.addWidget(QLabel("  Input size "))
        self.spin_size = QSpinBox()
        self.spin_size.setRange(256, 2048)
        self.spin_size.setSingleStep(128)
        self.spin_size.setValue(DEFAULT_SIZE)
        bar.addWidget(self.spin_size)
        bar.addSeparator()

        self.btn_one = QPushButton("Generate mask (this image)")
        self.btn_one.clicked.connect(self.generate_current)
        bar.addWidget(self.btn_one)

        self.btn_all = QPushButton("Generate all images")
        self.btn_all.clicked.connect(self.generate_all)
        bar.addWidget(self.btn_all)

        self.chk_skip = QCheckBox("skip existing")
        self.chk_skip.setChecked(True)
        self.chk_skip.setToolTip("In 'Generate all', skip images that already "
                                 "have a mask file on disk.")
        bar.addWidget(self.chk_skip)

        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self.stop_batch)
        bar.addWidget(self.btn_stop)

        # --- overlay controls ---
        bar2 = QWidget()
        row = QHBoxLayout(bar2)
        row.setContentsMargins(6, 2, 6, 2)
        self.chk_overlay = QCheckBox("Overlay mask")
        self.chk_overlay.setChecked(True)
        self.chk_overlay.toggled.connect(self.redraw)
        row.addWidget(self.chk_overlay)

        row.addWidget(QLabel(" mode "))
        self.cmb_overlay = QComboBox()
        self.cmb_overlay.addItems(["Tint subject", "Tint background",
                                   "Cutout on checker", "Alpha fade"])
        self.cmb_overlay.currentIndexChanged.connect(self.redraw)
        row.addWidget(self.cmb_overlay)

        row.addWidget(QLabel(" opacity "))
        self.sld_alpha = QSlider(Qt.Horizontal)
        self.sld_alpha.setRange(0, 100)
        self.sld_alpha.setValue(50)
        self.sld_alpha.setFixedWidth(120)
        self.sld_alpha.valueChanged.connect(self.redraw)
        row.addWidget(self.sld_alpha)

        self.chk_binary = QCheckBox("Binarize")
        self.chk_binary.toggled.connect(self.redraw)
        self.chk_binary.setToolTip("Threshold the mask (also affects what is "
                                   "saved).")
        row.addWidget(self.chk_binary)

        row.addWidget(QLabel(" threshold "))
        self.sld_thresh = QSlider(Qt.Horizontal)
        self.sld_thresh.setRange(1, 99)
        self.sld_thresh.setValue(50)
        self.sld_thresh.setFixedWidth(120)
        self.sld_thresh.valueChanged.connect(self.redraw)
        row.addWidget(self.sld_thresh)

        btn_fit = QPushButton("Fit")
        btn_fit.clicked.connect(lambda: (self.view_img.fit(),
                                         self.view_mask.fit()))
        row.addWidget(btn_fit)
        row.addStretch(1)

        self.progress = QProgressBar()
        self.progress.setFixedWidth(220)
        self.progress.setVisible(False)
        row.addWidget(self.progress)

        holder = QWidget()
        hl = QVBoxLayout(holder)
        hl.setContentsMargins(0, 0, 0, 0)
        hl.addWidget(bar2)
        dock2 = QDockWidget("", self)
        dock2.setTitleBarWidget(QWidget())
        dock2.setFeatures(QDockWidget.NoDockWidgetFeatures)
        dock2.setWidget(holder)
        self.addDockWidget(Qt.TopDockWidgetArea, dock2)

        # --- file list ---
        self.list = QListWidget()
        self.list.currentRowChanged.connect(self.on_row)
        self.list.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        dock = QDockWidget("Images", self)
        dock.setFeatures(QDockWidget.NoDockWidgetFeatures)
        dock.setWidget(self.list)
        dock.setMinimumWidth(280)
        self.addDockWidget(Qt.LeftDockWidgetArea, dock)

        self.statusBar().showMessage("Open a folder of images to start.")

        for key, slot in (("Right", lambda: self.step(1)),
                          ("Left", lambda: self.step(-1)),
                          ("Ctrl+G", self.generate_current),
                          ("Ctrl+Shift+G", self.generate_all)):
            a = QAction(self)
            a.setShortcut(QKeySequence(key))
            a.triggered.connect(slot)
            self.addAction(a)

    # ------------------------------------------------------------ worker ---
    def _start_worker(self) -> None:
        self.thread = QThread(self)
        self.worker = Worker()
        self.worker.moveToThread(self.thread)
        self.request_batch.connect(self.worker.run_batch)
        self.worker.loading.connect(
            lambda m: self.statusBar().showMessage(f"Loading {m} …"))
        self.worker.done_one.connect(self.on_mask)
        self.worker.failed.connect(self.on_failed)
        self.worker.progress.connect(self.on_progress)
        self.worker.finished.connect(self.on_finished)
        self.thread.start()

    # ------------------------------------------------------------ folder ---
    def pick_folder(self) -> None:
        start = str(self.images_dir or Path.cwd())
        d = QFileDialog.getExistingDirectory(self, "Folder of images", start)
        if d:
            self.load_folder(Path(d))

    def pick_out(self) -> None:
        start = str(self.out_dir or self.images_dir or Path.cwd())
        d = QFileDialog.getExistingDirectory(self, "Folder to save masks in",
                                             start)
        if d:
            self.out_dir = Path(d)
            self.out_override = True
            self.refresh_list(keep_row=True)
            self.statusBar().showMessage(f"Masks -> {self.out_dir}")

    def load_folder(self, folder: Path) -> None:
        paths = list_images(folder)
        if not paths:
            QMessageBox.warning(self, "gen_mask", f"No images in {folder}")
            return
        self.images_dir = folder
        self.paths = paths
        self.masks.clear()
        if not self.out_override:
            self.out_dir = default_out_dir(folder)
        self.refresh_list()
        self.list.setCurrentRow(0)
        self.statusBar().showMessage(
            f"{len(paths)} images in {folder}  |  masks -> {self.out_dir}")

    def mask_path(self, img: Path) -> Path:
        return (self.out_dir or default_out_dir(img.parent)) / f"{img.stem}.png"

    def refresh_list(self, keep_row: bool = False) -> None:
        row = self.list.currentRow()
        self.list.blockSignals(True)
        self.list.clear()
        for p in self.paths:
            done = str(p) in self.masks or self.mask_path(p).exists()
            it = QListWidgetItem(("✓  " if done else "     ") + p.name)
            if done:
                it.setForeground(QColor("#2ecc71"))
            self.list.addItem(it)
        self.list.blockSignals(False)
        if keep_row and 0 <= row < self.list.count():
            self.list.setCurrentRow(row)

    def mark_done(self, path: Path) -> None:
        try:
            i = self.paths.index(path)
        except ValueError:
            return
        it = self.list.item(i)
        if it:
            it.setText("✓  " + path.name)
            it.setForeground(QColor("#2ecc71"))

    # ------------------------------------------------------------- images ---
    @property
    def current(self) -> Path | None:
        i = self.list.currentRow()
        return self.paths[i] if 0 <= i < len(self.paths) else None

    def step(self, d: int) -> None:
        i = self.list.currentRow() + d
        if 0 <= i < self.list.count():
            self.list.setCurrentRow(i)

    def on_row(self, i: int) -> None:
        p = self.current
        if p is None:
            return
        from PIL import Image
        with Image.open(p) as im:
            self.rgb = np.asarray(im.convert("RGB"))

        self.mask = self.masks.get(str(p))
        if self.mask is None:
            mp = self.mask_path(p)
            if mp.exists():                      # show a mask made earlier
                try:
                    with Image.open(mp) as im:
                        m = np.asarray(im.convert("L"))
                    if m.shape == self.rgb.shape[:2]:
                        self.mask = m.astype(np.float32) / 255.0
                except Exception:
                    self.mask = None
        self.redraw(keep_view=False)
        self.statusBar().showMessage(
            f"[{i + 1}/{len(self.paths)}] {p.name}   "
            f"{self.rgb.shape[1]}x{self.rgb.shape[0]}"
            + ("" if self.mask is not None else "   (no mask yet)"))

    # ------------------------------------------------------------- render ---
    def hard(self, m: np.ndarray) -> np.ndarray:
        if self.chk_binary.isChecked():
            return (m >= self.sld_thresh.value() / 100.0).astype(np.float32)
        return m

    def redraw(self, *_, keep_view: bool = True) -> None:
        if self.rgb is None:
            return
        base = self.rgb
        m = self.hard(self.mask) if self.mask is not None else None

        if m is None or not self.chk_overlay.isChecked():
            left = base
        else:
            a = (self.sld_alpha.value() / 100.0) * m[..., None]
            mode = self.cmb_overlay.currentIndex()
            f = base.astype(np.float32)
            if mode == 0:                                    # tint subject
                col = np.array([46, 204, 113], dtype=np.float32)
                left = f * (1 - a) + col * a
            elif mode == 1:                                  # tint background
                col = np.array([255, 56, 56], dtype=np.float32)
                b = (self.sld_alpha.value() / 100.0) * (1 - m)[..., None]
                left = f * (1 - b) + col * b
            elif mode == 2:                                  # cutout, checker
                h, w = m.shape
                yy, xx = np.mgrid[0:h, 0:w]
                chk = (((yy // 16) + (xx // 16)) % 2).astype(np.float32)
                bg = (90.0 + 40.0 * chk)[..., None]
                left = f * m[..., None] + bg * (1 - m[..., None])
            else:                                            # alpha fade
                left = f * m[..., None]
            left = np.clip(left, 0, 255).astype(np.uint8)

        self.view_img.set_pixmap(to_qpixmap(left), keep_view)
        if m is None:
            self.view_mask.set_pixmap(None, keep_view)
        else:
            self.view_mask.set_pixmap(
                to_qpixmap((m * 255).round().astype(np.uint8)), keep_view)
        if not keep_view:
            self.view_img.fit()
            self.view_mask.fit()

    # ---------------------------------------------------------- generate ---
    def _busy(self, on: bool) -> None:
        for w in (self.btn_one, self.btn_all, self.cmb_model, self.spin_size):
            w.setEnabled(not on)
        self.btn_stop.setEnabled(on)
        self.progress.setVisible(on)

    def generate_current(self) -> None:
        p = self.current
        if p is None or self.btn_stop.isEnabled():
            return
        self._dispatch([p])

    def generate_all(self) -> None:
        if not self.paths or self.btn_stop.isEnabled():
            return
        todo = [p for p in self.paths
                if not (self.chk_skip.isChecked() and self.mask_path(p).exists())]
        if not todo:
            self.statusBar().showMessage(
                "Every image already has a mask (uncheck 'skip existing' to "
                "redo them).")
            return
        self._dispatch(todo)

    def _dispatch(self, paths: list[Path]) -> None:
        if self.out_dir is None:
            return
        try:
            self.out_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            QMessageBox.critical(self, "gen_mask",
                                 f"Cannot create {self.out_dir}\n{e}")
            return
        self._busy(True)
        self.progress.setRange(0, len(paths))
        self.progress.setValue(0)
        self.request_batch.emit([str(p) for p in paths],
                                self.cmb_model.currentData(),
                                self.spin_size.value())

    def on_mask(self, path: str, mask: np.ndarray) -> None:
        p = Path(path)
        self.masks[path] = mask
        out = self.mask_path(p)
        try:
            from PIL import Image
            arr = (self.hard(mask) * 255).round().astype(np.uint8)
            Image.fromarray(arr, mode="L").save(out)
        except Exception as e:
            self.statusBar().showMessage(f"Could not save {out}: {e}")
        self.mark_done(p)
        if p == self.current:
            self.mask = mask
            self.redraw()

    def on_failed(self, path: str, err: str) -> None:
        if not path:                                 # model failed to load
            self._busy(False)
            QMessageBox.critical(self, "gen_mask", err)
        else:
            self.statusBar().showMessage(f"Failed on {Path(path).name} — "
                                         f"{err.strip().splitlines()[-1]}")

    def on_progress(self, done: int, total: int) -> None:
        self.progress.setValue(done)
        self.statusBar().showMessage(f"Generating masks… {done}/{total}")

    def on_finished(self) -> None:
        self._busy(False)
        self.statusBar().showMessage(f"Done. Masks in {self.out_dir}")

    def stop_batch(self) -> None:
        self.worker.cancel()
        self.statusBar().showMessage("Stopping after the current image…")

    def closeEvent(self, ev) -> None:
        self.worker.cancel()
        self.thread.quit()
        self.thread.wait(5000)
        super().closeEvent(ev)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--images", type=Path, help="folder of images to open")
    ap.add_argument("--out", type=Path,
                    help="folder to write masks into "
                         "(default: <images>/../mask)")
    ap.add_argument("--model", choices=list(MODELS), default="birefnet")
    a = ap.parse_args()

    if a.images and not a.images.is_dir():
        ap.error(f"{a.images} is not a folder")

    app = QApplication(sys.argv)
    win = MainWindow(a.images, a.out, a.model)
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
