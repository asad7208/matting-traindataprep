#!/usr/bin/env python3
"""SAM wrapper for the mask fixer — SAM 2.1 or SAM 3, same interface.

SAM 2.1 (facebook/sam2.1-hiera-large) is ungated and is the default. It takes
click points and boxes. It has no text encoder, so "segment the person" is
expressed geometrically instead — see whole_person_prompt().

SAM 3 (facebook/sam3) additionally understands text ("person", "legs") and is
a drop-in once its licence is granted; one checkpoint holds both its heads.

Everything returns a float32 mask in [0, 1] at the image's own size, which is
what the pipeline's masks/*.png already are, so results union or subtract
straight into them.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

# Points down the body's centre line, plus the four corners as negatives. On a
# person crop the padding guarantees the corners are background, and the spine
# hits torso and legs alike — which is the whole point, since the legs are what
# the matte usually drops. Measured IoU 1.00 / 0.997 against a known mask.
SPINE_Y = (0.25, 0.50, 0.80)


class SamError(RuntimeError):
    """Anything the user can act on: no weights, no licence, no GPU."""


Sam3Error = SamError          # the old name, kept so existing imports still work


GATED_HINT = """\
{repo} weights are gated on Hugging Face. One-time setup:

  1. open https://huggingface.co/{repo} and accept the licence
  2. hf auth login          (or export HF_TOKEN=hf_...)
  3. ./setup_sam.sh sam3    (downloads into ./weights)

Or stay on SAM 2, which needs no licence: set `model: sam2` in the config.
"""


def whole_person_prompt(w: int, h: int) -> tuple[list[tuple[float, float]], list[int]]:
    """The geometric stand-in for typing "person": spine positives + corner negatives."""
    pts = [(w / 2, h * f) for f in SPINE_Y]
    labs = [1] * len(pts)
    inset = 2
    pts += [(inset, inset), (w - inset, inset),
            (inset, h - inset), (w - inset, h - inset)]
    labs += [0, 0, 0, 0]
    return pts, labs


# ------------------------------------------------------------------ backends

class _Base:
    supports_text = False
    label = "sam"

    def __init__(self, repo: str, device: str = "",
                 local_dir: str | Path | None = None, half: bool | None = None):
        # A local folder wins over the hub name, so a machine that already ran
        # setup_sam.sh never reaches for the network.
        self.source = str(local_dir) if local_dir and Path(local_dir).exists() else repo
        self.repo = repo
        self.device = device or self._auto_device()
        self.half = self.device.startswith("cuda") if half is None else half
        self._model = self._proc = None

    @staticmethod
    def _auto_device() -> str:
        try:
            import torch
            return "cuda:0" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    def _wrap_load(self, fn):
        try:
            return fn()
        except Exception as e:                                   # noqa: BLE001
            msg = str(e)
            if any(k in msg.lower() for k in ("gated", "restricted")) or "401" in msg:
                raise SamError(GATED_HINT.format(repo=self.repo)) from e
            if "not a local folder" in msg or "Repository Not Found" in msg:
                raise SamError(
                    f"cannot find {self.label} weights at {self.source!r}. "
                    f"Run ./setup_sam.sh to download them.") from e
            raise SamError(f"loading {self.label} from {self.source!r} failed: {msg}") from e

    def _load(self):
        raise NotImplementedError

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def _to_device(self, model):
        model = model.to(self.device)
        if self.half:
            model = model.half()
        return model.eval()

    def _cast(self, inputs):
        import torch
        out = {}
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                if self.half and v.dtype == torch.float32:
                    v = v.half()
                v = v.to(self.device)
            out[k] = v
        return out

    def _pick_best(self, out, masks):
        """SAM returns whole / part / subpart — trust its own quality head."""
        import torch
        m = torch.as_tensor(masks).float()
        while m.ndim > 3:
            m = m[0]
        if m.shape[0] > 1 and getattr(out, "iou_scores", None) is not None:
            scores = out.iou_scores.float().cpu().reshape(-1)
            m = m[int(scores[:m.shape[0]].argmax())]
        else:
            m = m[0]
        return torch.sigmoid(m).clamp(0, 1).numpy().astype(np.float32)

    # -- prompts every backend must answer
    def segment_points(self, image, points, labels, box=None) -> np.ndarray:
        raise NotImplementedError

    def segment_whole(self, image) -> np.ndarray:
        """Whole person, without a text prompt."""
        w, h = image.size
        pts, labs = whole_person_prompt(w, h)
        return self.segment_points(image, pts, labs)

    def segment_text(self, image, text, threshold=0.3, boxes=None) -> np.ndarray:
        raise SamError(
            f"{self.label} has no text encoder — use 'Whole person', a click, "
            f"or a box. Text prompts need model: sam3 in the config.")


class Sam2Backend(_Base):
    supports_text = False
    label = "SAM 2.1"

    def _load(self):
        if self._model is None:
            from transformers import Sam2Model, Sam2Processor
            self._proc = self._wrap_load(
                lambda: Sam2Processor.from_pretrained(self.source))
            # The published config declares sam2_video; loading the image model
            # from it is supported and warns once. Harmless.
            self._model = self._to_device(self._wrap_load(
                lambda: Sam2Model.from_pretrained(self.source)))
        return self._model, self._proc

    def segment_points(self, image, points, labels, box=None) -> np.ndarray:
        import torch

        w, h = image.size
        if not points and box is None:
            return np.zeros((h, w), np.float32)
        model, proc = self._load()

        kw = {}
        if points:
            kw["input_points"] = [[[[float(x), float(y)] for x, y in points]]]
            kw["input_labels"] = [[[int(v) for v in labels]]]
        if box is not None:
            kw["input_boxes"] = [[[float(v) for v in box]]]

        inputs = proc(images=image, return_tensors="pt", **kw)
        with torch.no_grad():
            out = model(**self._cast(inputs), multimask_output=True)
        masks = proc.post_process_masks(out.pred_masks.float().cpu(),
                                        inputs["original_sizes"], binarize=False)[0]
        return self._pick_best(out, masks)


class Sam3Backend(_Base):
    supports_text = True
    label = "SAM 3"

    def _load_detector(self):
        from transformers import Sam3Model, Sam3Processor
        if self._model is None:
            self._proc = self._wrap_load(
                lambda: Sam3Processor.from_pretrained(self.source))
            self._model = self._to_device(self._wrap_load(
                lambda: Sam3Model.from_pretrained(self.source)))
        return self._model, self._proc

    def _load_tracker(self):
        from transformers import Sam3TrackerModel, Sam3TrackerProcessor
        if getattr(self, "_trk", None) is None:
            self._trk_proc = self._wrap_load(
                lambda: Sam3TrackerProcessor.from_pretrained(self.source))
            self._trk = self._to_device(self._wrap_load(
                lambda: Sam3TrackerModel.from_pretrained(self.source)))
        return self._trk, self._trk_proc

    def segment_text(self, image, text, threshold=0.3, boxes=None) -> np.ndarray:
        import torch

        model, proc = self._load_detector()
        w, h = image.size
        kw = {}
        if boxes:
            kw["input_boxes"] = [[list(map(float, b)) for b in boxes]]
            kw["input_boxes_labels"] = [[1] * len(boxes)]
        inputs = proc(images=image, text=text or "person",
                      return_tensors="pt", **kw)
        with torch.no_grad():
            out = model(**self._cast(inputs))
        res = proc.post_process_instance_segmentation(
            out, threshold=threshold, target_sizes=[(h, w)])[0]
        masks = res.get("masks")
        if masks is None or len(masks) == 0:
            return np.zeros((h, w), np.float32)
        m = torch.as_tensor(masks).float()
        if m.ndim == 4:
            m = m[:, 0]
        return m.max(0).values.clamp(0, 1).cpu().numpy().astype(np.float32)

    def segment_points(self, image, points, labels, box=None) -> np.ndarray:
        import torch

        w, h = image.size
        if not points and box is None:
            return np.zeros((h, w), np.float32)
        model, proc = self._load_tracker()
        kw = {}
        if points:
            kw["input_points"] = [[[[float(x), float(y)] for x, y in points]]]
            kw["input_labels"] = [[[int(v) for v in labels]]]
        if box is not None:
            kw["input_boxes"] = [[[float(v) for v in box]]]
        inputs = proc(images=image, return_tensors="pt", **kw)
        with torch.no_grad():
            out = model(**self._cast(inputs), multimask_output=True)
        masks = proc.post_process_masks(out.pred_masks.float().cpu(),
                                        inputs["original_sizes"], binarize=False)[0]
        return self._pick_best(out, masks)


BACKENDS = {"sam2": Sam2Backend, "sam3": Sam3Backend}


def make_backend(cfg: dict):
    key = str(cfg.get("model", "sam2")).strip().lower()
    if key not in BACKENDS:
        raise SamError(f"unknown model {key!r} — pick one of {list(BACKENDS)}")
    return BACKENDS[key](str(cfg.get(f"{key}_repo", "")),
                         str(cfg.get("device", "")),
                         cfg.get(f"{key}_weights"),
                         cfg.get("half"))


# --------------------------------------------------------------- mask maths

def combine(base: np.ndarray, new: np.ndarray, op: str) -> np.ndarray:
    """Fold a SAM result into the mask being fixed. Both float [0,1]."""
    if base.shape != new.shape:
        raise ValueError(f"mask shapes differ: {base.shape} vs {new.shape}")
    if op == "add":
        return np.maximum(base, new)
    if op == "subtract":
        return np.clip(base - new, 0.0, 1.0)
    if op == "replace":
        return new.copy()
    if op == "intersect":
        return np.minimum(base, new)
    raise ValueError(f"unknown op: {op}")
