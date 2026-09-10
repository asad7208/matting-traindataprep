#!/usr/bin/env python3
"""End-to-end person-matting dataset pipeline, driven entirely from a terminal.

Same work as the GUI tools, no Qt:

    1. MINE   detect + ByteTrack people, pose (yolo11x-pose) for body
              configuration / full-body check, CLIP for appearance, dHash
              near-duplicate rejection, per-track cap, farthest-point sampling
              with a hard-example quota  ->  <out>/review/{images,labels}
              (this is mine_person_crops.scan/select/write_review_set verbatim)
    2. REVIEW optional approve/reject walk in the terminal (metadata only)
    3. CROPS  cut every approved box out with padding  ->  <out>/crops
    4. MASKS  BiRefNet (or RMBG-2.0) soft/binary mattes  ->  <out>/masks
    5. FINAL  RGBA cutouts / flat-background composites  ->  <out>/final

Run it with no arguments and it asks for everything; pass flags (or --yes) and
it asks for nothing.

    python pipeline_cli.py                                  # fully interactive
    python pipeline_cli.py --source clips/ --out ds --yes    # unattended
    python pipeline_cli.py --out ds --stages masks,final --yes
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
STAGES = ("mine", "review", "crops", "masks", "final")

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Same table as gen_mask/gen_mask.py: loaded through transformers'
# AutoModelForImageSegmentation with trust_remote_code.
MODELS = {
    "birefnet": {"repo": "ZhengPeng7/BiRefNet", "label": "BiRefNet (general)",
                 "norm": (IMAGENET_MEAN, IMAGENET_STD), "minmax": False},
    "rmbg2": {"repo": "briaai/RMBG-2.0", "label": "RMBG-2.0",
              "norm": (IMAGENET_MEAN, IMAGENET_STD), "minmax": False},
}

GATED_HINT = """
This model is gated on Hugging Face. Accept the licence at
    https://huggingface.co/{repo}
then log in once:  hf auth login   (or export HF_TOKEN=...)
"""


# --------------------------------------------------------------- terminal io

class Ask:
    """Terminal prompts. With --yes (or no tty) every answer is the default."""

    def __init__(self, interactive: bool):
        self.on = interactive and sys.stdin.isatty()

    def text(self, label: str, default: str) -> str:
        if not self.on:
            return default
        v = input(f"  {label} [{default}]: ").strip()
        return v or default

    def num(self, label, default, cast=int, lo=None, hi=None):
        while True:
            raw = self.text(label, str(default))
            try:
                v = cast(raw)
            except ValueError:
                print("    not a number")
                continue
            if (lo is not None and v < lo) or (hi is not None and v > hi):
                print(f"    must be between {lo} and {hi}")
                continue
            return v

    def yn(self, label: str, default: bool) -> bool:
        d = "Y/n" if default else "y/N"
        while True:
            raw = self.text(label + f" ({d})", "y" if default else "n").lower()
            if raw in ("y", "yes"):
                return True
            if raw in ("n", "no"):
                return False
            print("    answer y or n")

    def choice(self, label: str, options: list[str], default: str) -> str:
        if not self.on:
            return default
        print(f"  {label}")
        for i, o in enumerate(options, 1):
            print(f"    {i}) {o}{'   (default)' if o == default else ''}")
        while True:
            raw = input(f"  choose 1-{len(options)} [{default}]: ").strip()
            if not raw:
                return default
            if raw in options:
                return raw
            if raw.isdigit() and 1 <= int(raw) <= len(options):
                return options[int(raw) - 1]
            print("    no such option")


class Bar:
    """One-line progress bar. total <= 0 means "unknown length" (counts only)."""

    def __init__(self, total: int, label: str, width: int = 30):
        self.total = int(total)
        self.label = label
        self.width = width
        self.t0 = time.time()
        self.last = 0.0
        self.n = 0

    @staticmethod
    def _clock(sec: float) -> str:
        sec = max(0, int(sec))
        return f"{sec // 60:d}:{sec % 60:02d}"

    def update(self, n: int, suffix: str = "", force: bool = False) -> None:
        self.n = n
        now = time.time()
        if not force and now - self.last < 0.1:
            return
        self.last = now
        el = now - self.t0
        if self.total > 0:
            frac = min(1.0, max(0.0, n / self.total))
            fill = int(round(frac * self.width))
            bar = "#" * fill + "-" * (self.width - fill)
            eta = (el / frac - el) if frac > 0.02 else 0
            line = (f"  {self.label} [{bar}] {frac * 100:5.1f}%  "
                    f"{n}/{self.total}  {self._clock(el)}"
                    + (f"<{self._clock(eta)}" if eta else ""))
        else:
            spin = "|/-\\"[int(el * 8) % 4]
            line = f"  {self.label} {spin} {n}  {self._clock(el)} elapsed"
        if suffix:
            line += f"  {suffix}"
        print("\r" + line[:150].ljust(150), end="", flush=True)

    def close(self, suffix: str = "") -> None:
        # finish the bar even if the last few items hit a `continue`
        self.update(self.total if self.total > 0 else self.n, suffix, force=True)
        print()


def head(title: str) -> None:
    print("\n" + "=" * 68)
    print(f"  {title}")
    print("=" * 68)


def ask_stages(ask: "Ask", default: list[str]) -> list[str]:
    """Pick stages by name or by number. Never silently returns nothing."""
    print("\n  pipeline stages:")
    what = {"mine": "detect/track/pose/CLIP -> pick the frames worth labelling",
            "review": "approve or reject the mined frames (interactive)",
            "crops": "cut each approved box out with padding",
            "masks": "BiRefNet/RMBG matte for every crop",
            "final": "composite the cutouts (rgba / white / green / …)"}
    for i, st in enumerate(STAGES, 1):
        mark = "*" if st in default else " "
        print(f"   {mark} {i}) {st:<7} {what[st]}")
    print("   (* = will run)  enter names or numbers, comma separated, "
          "or 'all'; a range like 3-5 works too")

    while True:
        raw = ask.text("stages to run", ",".join(default)).strip().lower()
        if raw in ("all", "*"):
            return list(STAGES)
        picked, bad = [], []
        for tok in (t.strip() for t in raw.split(",") if t.strip()):
            span = tok.split("-")
            if len(span) == 2 and all(x.strip().isdigit() for x in span):
                lo, hi = (int(x) for x in span)
                if 1 <= lo <= hi <= len(STAGES):
                    picked += [STAGES[i - 1] for i in range(lo, hi + 1)]
                    continue
                bad.append(tok)
            elif tok.isdigit():
                if 1 <= int(tok) <= len(STAGES):
                    picked.append(STAGES[int(tok) - 1])
                else:
                    bad.append(tok)
            elif tok in STAGES:
                picked.append(tok)
            else:
                bad.append(tok)
        if bad:
            print(f"    don't know: {', '.join(bad)}  "
                  f"(names: {', '.join(STAGES)}  numbers: 1-{len(STAGES)})")
            continue
        if not picked:
            print("    pick at least one stage")
            continue
        # keep pipeline order, drop repeats
        return [st for st in STAGES if st in set(picked)]


def pick_device(requested: str) -> str:
    if requested:
        return requested
    try:
        import torch
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def list_images(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir()
                  if p.is_file() and p.suffix.lower() in IMG_EXT)


# ------------------------------------------------------------ stage 1: mine

def stage_mine(a, ask: Ask, out: Path) -> Path:
    """Run the miner (tracking + pose + CLIP + dedupe + FPS) into <out>/review."""
    head("STAGE 1/5  mine person frames  (track -> pose -> CLIP -> select)")
    import mine_person_crops as M

    # Every mining knob comes from the command line / the defaults in
    # build_parser(); the miner never stops to ask. Override any of them with
    # the matching --flag (see --help) or the variables in run_pipeline.sh.
    source = a.source or ""
    if not source:
        source = ask.text("source video / folder of videos / folder of frames",
                          "recordings")
    if not Path(source).exists():
        sys.exit(f"source does not exist: {source}")

    budget = a.budget
    per_track = a.per_track
    min_gap = a.min_gap
    stride = a.stride
    min_conf = a.min_conf
    min_height = a.min_height
    dhash_dist = a.dhash_dist
    use_pose = not a.no_pose
    use_clip = not a.no_clip
    full_only = a.full_body_only

    # Build the miner's own namespace so every default/behaviour stays identical.
    review = out / "review"
    argv = ["--source", str(source), "--out", str(review),
            "--budget", str(budget),
            "--auto-frac", str(a.auto_frac),
            "--auto-min", str(a.auto_min), "--auto-max", str(a.auto_max),
            "--per-track", str(per_track),
            "--min-gap", str(min_gap), "--stride", str(stride),
            "--min-conf", str(min_conf), "--min-height", str(min_height),
            "--dhash-dist", str(dhash_dist),
            "--det-model", a.det_model, "--imgsz", str(a.imgsz),
            "--device", a.device,
            "--pose-model", (a.pose_model if use_pose else "")]
    if not use_clip:
        argv.append("--no-clip")
    if not full_only:
        argv.append("--allow-partial")
    margs = M.build_parser().parse_args(argv)
    if margs.pose_model in ("", "none", "None"):
        margs.pose_model = None

    for w in filter(None, (margs.det_model, margs.pose_model)):
        if not Path(w).exists():
            print(f"  [warn] {w} not found locally; ultralytics will try to "
                  f"download it")

    print(f"\n  source      : {source}")
    print(f"  device      : {margs.device}")
    print(f"  detector    : {margs.det_model}")
    print(f"  pose        : {margs.pose_model or 'off'}")
    print(f"  clip        : {'off' if margs.no_clip else 'on'}")
    if margs.budget <= 0:
        print(f"  budget      : auto — kept until the picks stop adding variety "
              f"(frac {a.auto_frac}, {a.auto_min}-{a.auto_max})")
    else:
        print(f"  budget      : {margs.budget} frame(s)")
    print(f"  per person  : max {per_track} frame(s), >= {min_gap} frames apart")
    print(f"  gates       : conf >= {min_conf}, height >= {min_height}px, "
          f"dHash > {dhash_dist} bits, "
          f"{'full bodies only' if full_only else 'partial bodies allowed'}")
    print(f"  stride      : every {stride} frame(s)")
    print(f"  review set  : {review}\n")

    t0 = time.time()

    # Total frames the scan will read, so the bar has a denominator. Wrong or
    # missing totals only make the bar indeterminate; they never stop the scan.
    total_frames = 0
    try:
        vids, imgdirs = M.list_sources(Path(source))
        for v in vids:
            cap = cv2.VideoCapture(str(v))
            total_frames += int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) // max(1, margs.stride)
            cap.release()
        for d in imgdirs:
            total_frames += len(list_images(d))
    except Exception:                                           # noqa: BLE001
        total_frames = 0

    bar = Bar(total_frames, "scan  ")
    seen: dict[str, int] = {}

    def on_progress(name, frame_idx, n_cands):
        seen[name] = frame_idx // max(1, margs.stride) if margs.stride else frame_idx
        bar.update(sum(seen.values()), f"{n_cands} candidates  {name[:28]}")
        return True

    cands, meta = M.scan(margs, on_progress=on_progress)
    bar.close(f"{meta['n_candidates']} candidates")
    print(f"[scan] {meta['n_candidates']} candidates in {meta['seconds']}s "
          f"(pose={meta['pose']}, clip={meta['clip']})")
    if not cands:
        sys.exit("no person detections passed --min-height / --min-conf")

    print("  select: dedupe -> per-track cap -> farthest-point sampling …")
    picked, stats = M.select(cands, margs)
    if not picked:
        sys.exit("nothing survived the quality gate; loosen the filters")

    print(f"  writing {len(picked)} selected box(es) to {review} …")
    man = M.write_review_set(picked, review, margs, stats)
    print(f"\n[mine] {man['images']} frames / {man['crops']} boxes -> {review} "
          f"({time.time() - t0:.0f}s)")
    print("       " + " -> ".join(f"{k} {v}" for k, v in stats.items()))
    return review


# ---------------------------------------------------- stage 2: review in tty

def load_yolo_boxes(lbl: Path, w: int, h: int) -> list[tuple[int, float, float, float, float]]:
    out = []
    if not lbl.exists():
        return out
    for line in lbl.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cls = int(float(parts[0]))
        cx, cy, bw, bh = (float(v) for v in parts[1:5])
        x1 = (cx - bw / 2) * w; y1 = (cy - bh / 2) * h
        x2 = (cx + bw / 2) * w; y2 = (cy + bh / 2) * h
        out.append((cls, x1, y1, x2, y2))
    return out


def stage_review(a, ask: Ask, review: Path) -> set[str]:
    """Approve/reject frames from the terminal; returns approved image names."""
    head("STAGE 2/5  review the mined frames")
    images = list_images(review / "images")
    if not images:
        sys.exit(f"no images in {review / 'images'} — run the mine stage first")

    approved_file = review / "approved.txt"
    if not ask.on or a.no_review:
        names = {p.name for p in images}
        print(f"  auto-approving all {len(names)} frame(s) "
              f"(--no-review / non-interactive)")
        approved_file.write_text("\n".join(sorted(names)) + "\n")
        return names

    info: dict[str, dict] = {}
    man = review / "manifest.json"
    if man.exists():
        try:
            for e in json.loads(man.read_text()).get("entries", []):
                info[e["image"]] = e
        except Exception:                                       # noqa: BLE001
            pass

    print("  enter = approve, x = reject, a = approve all remaining, "
          "r = reject all remaining, q = stop here\n")
    approved: set[str] = set()
    auto = None
    for i, p in enumerate(images, 1):
        e = info.get(p.name, {})
        boxes = e.get("boxes", [])
        bits = []
        for b in boxes[:4]:
            bits.append("conf %.2f h %.2f sharp %.0f occl %.2f%s" % (
                b.get("conf") or 0, b.get("h_rel") or 0, b.get("sharp") or 0,
                b.get("occl") or 0, "  HARD" if b.get("hard") else ""))
        desc = " | ".join(bits) or "no metadata"
        line = f"  [{i}/{len(images)}] {p.name}  {len(boxes) or '?'} box(es)  {desc}"
        if auto is not None:
            if auto:
                approved.add(p.name)
            continue
        print(line)
        k = input("      keep? ").strip().lower()
        if k in ("", "y", "yes"):
            approved.add(p.name)
        elif k == "a":
            auto = True
            approved.add(p.name)
        elif k == "r":
            auto = False
        elif k == "q":
            break
    approved_file.write_text("\n".join(sorted(approved)) + "\n")
    print(f"\n  approved {len(approved)}/{len(images)} frame(s) "
          f"-> {approved_file}")
    return approved


# ----------------------------------------------------------- stage 3: crops

def stage_crops(a, ask: Ask, review: Path, out: Path,
                approved: set[str] | None) -> Path:
    """Cut approved boxes out with padding (bbox_annotator's export, headless)."""
    head("STAGE 3/5  export person crops")
    images = list_images(review / "images")
    if not images:
        sys.exit(f"no images in {review / 'images'}")

    if approved is None:
        af = review / "approved.txt"
        if af.exists():
            approved = {ln.strip() for ln in af.read_text().splitlines() if ln.strip()}
        else:
            approved = {p.name for p in images}

    margin = ask.num("crop padding as a fraction of the box", a.crop_margin,
                     float, 0.0, 2.0)
    min_side = ask.num("skip crops smaller than N px on a side", a.min_crop_px,
                       int, 8)
    crops_dir = out / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    rows, n = [], 0
    todo = [p for p in images if p.name in approved]
    bar = Bar(len(todo), "crops ")
    for i, p in enumerate(todo, 1):
        img = cv2.imread(str(p))
        if img is None:
            print(f"  [warn] unreadable: {p.name}")
            continue
        ih, iw = img.shape[:2]
        boxes = load_yolo_boxes(review / "labels" / f"{p.stem}.txt", iw, ih)
        for j, (cls, x1, y1, x2, y2) in enumerate(boxes):
            mx, my = (x2 - x1) * margin, (y2 - y1) * margin
            rx1 = int(round(max(0.0, x1 - mx))); ry1 = int(round(max(0.0, y1 - my)))
            rx2 = int(round(min(float(iw), x2 + mx)))
            ry2 = int(round(min(float(ih), y2 + my)))
            if rx2 - rx1 < min_side or ry2 - ry1 < min_side:
                continue
            crop = img[ry1:ry2, rx1:rx2]
            name = f"{p.stem}_person{j:02d}.jpg"
            if not cv2.imwrite(str(crops_dir / name), crop,
                               [cv2.IMWRITE_JPEG_QUALITY, a.jpeg_quality]):
                continue
            rows.append(f"{name},{p.name},person,{cls},{rx1},{ry1},{rx2},{ry2},"
                        f"{iw},{ih}")
            n += 1
        bar.update(i, f"{n} crops")
    bar.close(f"{n} crops")

    (crops_dir / "crops.csv").write_text(
        "crop,frame,class_name,class_id,x1,y1,x2,y2,frame_w,frame_h\n"
        + "\n".join(rows) + "\n")
    print(f"[crops] {n} crop(s) from {len(todo)} frame(s) -> {crops_dir}")
    if n == 0:
        sys.exit("no crops written; check the padding / min size")
    return crops_dir


# ----------------------------------------------------------- stage 4: masks

class Matter:
    """Lazily loaded matting model (same maths as gen_mask.Matter)."""

    def __init__(self) -> None:
        self.key: str | None = None
        self.model = None
        self.device = "cpu"
        self.half = False

    def load(self, key: str, device: str) -> None:
        if self.key == key and self.model is not None:
            return
        import torch
        from transformers import AutoModelForImageSegmentation

        repo = MODELS[key]["repo"]
        print(f"  loading {MODELS[key]['label']} ({repo}) …")
        torch.set_float32_matmul_precision("high")
        try:
            model = AutoModelForImageSegmentation.from_pretrained(
                repo, trust_remote_code=True)
        except Exception as e:                                  # noqa: BLE001
            if "gated repo" in str(e) or "401" in str(e):
                sys.exit(GATED_HINT.format(repo=repo))
            raise
        self.device = device if device.startswith("cuda") else "cpu"
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            self.device = "cpu"
        self.half = self.device.startswith("cuda")
        model = model.to(self.device)
        if self.half:
            model = model.half()
        model.eval()
        self.model, self.key = model, key
        print(f"  ready on {self.device}{' (fp16)' if self.half else ''}")

    def infer(self, path: Path, size: int) -> np.ndarray:
        """Soft mask as float32 [0,1] at the image's own size."""
        import torch
        from PIL import Image

        with Image.open(path) as im:
            rgb = im.convert("RGB")
            w, h = rgb.size
            small = rgb.resize((size, size), Image.BILINEAR)

        mean, std = MODELS[self.key]["norm"]
        x = np.asarray(small, dtype=np.float32) / 255.0
        x = (x - np.array(mean, dtype=np.float32)) / np.array(std, np.float32)
        x = torch.from_numpy(x.transpose(2, 0, 1))[None].to(self.device)
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


def stage_masks(a, ask: Ask, src_dir: Path, out: Path) -> Path:
    """Run the matting model over a folder of images -> <out>/masks/*.png."""
    head("STAGE 4/5  generate mattes")
    from PIL import Image

    imgs = list_images(src_dir)
    if not imgs:
        sys.exit(f"no images to mask in {src_dir}")

    key = ask.choice("matting model", list(MODELS), a.model)
    size = ask.num("model input size", a.mask_size, int, 128, 2048)
    binarize = ask.yn("binarize the mask", a.binarize)
    thresh = (ask.num("threshold 0-1", a.threshold, float, 0.0, 1.0)
              if binarize else a.threshold)
    skip = ask.yn("skip images that already have a mask", not a.overwrite)

    masks_dir = out / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)
    matter = Matter()
    matter.load(key, a.device)

    done = fail = skipped = 0
    bar = Bar(len(imgs), "masks ")
    for i, p in enumerate(imgs, 1):
        mp = masks_dir / f"{p.stem}.png"
        if skip and mp.exists():
            skipped += 1
            bar.update(i, f"{done} done, {skipped} skipped")
            continue
        try:
            m = matter.infer(p, size)
        except Exception:                                       # noqa: BLE001
            fail += 1
            print()
            print(f"  [fail] {p.name}\n{traceback.format_exc(limit=2)}")
            continue
        if binarize:
            m = (m >= thresh).astype(np.float32)
        Image.fromarray((m * 255).clip(0, 255).astype(np.uint8), mode="L").save(mp)
        done += 1
        bar.update(i, p.name[:30])
    bar.close(f"{done} written, {skipped} skipped, {fail} failed")

    print(f"[masks] {done} written, {fail} failed -> {masks_dir}")
    return masks_dir


# ----------------------------------------------------------- stage 5: final

def stage_final(a, ask: Ask, src_dir: Path, masks_dir: Path, out: Path) -> Path:
    """Composite image + matte into the deliverable images."""
    head("STAGE 5/5  final output images")
    from PIL import Image

    imgs = list_images(src_dir)
    if not imgs:
        sys.exit(f"no source images in {src_dir}")

    mode = ask.choice("final output", ["rgba", "white", "black", "green", "checker"],
                      a.final_mode)
    also_side = ask.yn("also write a side-by-side preview (image | matte | cutout)",
                       a.preview)

    final_dir = out / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    prev_dir = out / "preview"
    if also_side:
        prev_dir.mkdir(parents=True, exist_ok=True)

    bg_rgb = {"white": (255, 255, 255), "black": (0, 0, 0), "green": (0, 255, 0)}
    n = 0
    bar = Bar(len(imgs), "final ")
    for i, p in enumerate(imgs, 1):
        mp = masks_dir / f"{p.stem}.png"
        if not mp.exists():
            continue
        bgr = cv2.imread(str(p))
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        m = np.asarray(Image.open(mp).convert("L").resize((w, h), Image.BILINEAR))
        alpha = m.astype(np.float32) / 255.0

        if mode == "rgba":
            rgba = np.dstack([rgb, m.astype(np.uint8)])
            Image.fromarray(rgba, mode="RGBA").save(final_dir / f"{p.stem}.png")
            comp = (rgb.astype(np.float32) * alpha[..., None]).astype(np.uint8)
        else:
            if mode == "checker":
                tile = 16
                yy, xx = np.mgrid[0:h, 0:w]
                chk = (((yy // tile) + (xx // tile)) % 2).astype(np.float32)
                bg = (chk * 205 + (1 - chk) * 245)[..., None].repeat(3, 2)
            else:
                bg = np.zeros((h, w, 3), np.float32)
                bg[:] = bg_rgb[mode]
            comp = (rgb.astype(np.float32) * alpha[..., None]
                    + bg * (1 - alpha[..., None])).astype(np.uint8)
            Image.fromarray(comp).save(final_dir / f"{p.stem}.png")
        n += 1

        if also_side:
            strip = np.hstack([rgb, m[..., None].repeat(3, 2).astype(np.uint8), comp])
            Image.fromarray(strip).save(prev_dir / f"{p.stem}.jpg", quality=92)
        bar.update(i, p.name[:30])
    bar.close(f"{n} written")

    print(f"[final] {n} image(s) -> {final_dir}"
          + (f"\n[final] previews -> {prev_dir}" if also_side else ""))
    return final_dir


# ------------------------------------------------------------------- driver

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Terminal person-matting dataset pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--source", help="video / folder of videos / folder of frames")
    ap.add_argument("--out", help="pipeline output folder")
    ap.add_argument("--stages", default=",".join(STAGES),
                    help="comma list of " + ",".join(STAGES))
    ap.add_argument("--yes", "-y", action="store_true",
                    help="take every default, ask nothing")

    g = ap.add_argument_group("mining")
    g.add_argument("--budget", default="auto",
                   help="frames to keep, or 'auto' (default): the diversity of "
                        "the footage decides how many are worth keeping")
    g.add_argument("--auto-frac", type=float, default=0.35,
                   help="lower = smaller, more distinct set; higher = bigger, "
                        "more redundant")
    g.add_argument("--auto-min", type=int, default=12)
    g.add_argument("--auto-max", type=int, default=2000)
    g.add_argument("--per-track", type=int, default=6)
    g.add_argument("--min-gap", type=int, default=12)
    g.add_argument("--stride", type=int, default=3)
    g.add_argument("--min-conf", type=float, default=0.3)
    g.add_argument("--min-height", type=int, default=96)
    g.add_argument("--dhash-dist", type=int, default=6)
    g.add_argument("--det-model", default=str(HERE / "weights/yolo11x.pt"))
    g.add_argument("--pose-model", default=str(HERE / "weights/yolo11x-pose.pt"))
    g.add_argument("--imgsz", type=int, default=1280)
    g.add_argument("--no-pose", action="store_true")
    g.add_argument("--no-clip", action="store_true")
    g.add_argument("--full-body-only", action="store_true",
                   help="keep only fully visible bodies (head+knees+ankle, box "
                        "clear of the frame edge). Off by default: partial "
                        "bodies are kept.")
    g.add_argument("--allow-partial", action="store_true",
                   help=argparse.SUPPRESS)          # now the default; kept so
                                                    # old command lines still run
    g.add_argument("--device", default="")

    g = ap.add_argument_group("review / crops")
    g.add_argument("--no-review", action="store_true",
                   help="approve every mined frame without asking")
    g.add_argument("--crop-margin", type=float, default=0.08)
    g.add_argument("--min-crop-px", type=int, default=32)
    g.add_argument("--jpeg-quality", type=int, default=95)
    g.add_argument("--mask-whole-frames", action="store_true",
                   help="matte the mined frames instead of the person crops")

    g = ap.add_argument_group("matting")
    g.add_argument("--model", choices=list(MODELS), default="birefnet")
    g.add_argument("--mask-size", type=int, default=1024)
    g.add_argument("--binarize", action="store_true")
    g.add_argument("--threshold", type=float, default=0.5)
    g.add_argument("--overwrite", action="store_true",
                   help="redo masks that already exist")
    g.add_argument("--final-mode", default="rgba",
                   choices=["rgba", "white", "black", "green", "checker"])
    g.add_argument("--preview", action="store_true",
                   help="also write image|matte|cutout strips")
    return ap


def main() -> int:
    a = build_parser().parse_args()
    ask = Ask(not a.yes)
    a.device = pick_device(a.device)

    head("person matting dataset pipeline")
    print(f"  device: {a.device}   torch/ultralytics load on demand")

    stages = [s.strip() for s in a.stages.split(",") if s.strip()]
    bad = [s for s in stages if s not in STAGES]
    if bad:
        sys.exit(f"unknown stage(s): {', '.join(bad)} (pick from {', '.join(STAGES)})")
    if ask.on:
        stages = ask_stages(ask, stages)

    if a.out:
        out = Path(a.out).expanduser()
        print(f"  output folder: {out}")
    else:
        out = Path(ask.text("output folder", "dataset_out")).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    print(f"  stages       : {', '.join(stages)}")

    review = out / "review"
    approved: set[str] | None = None
    crops_dir = out / "crops"
    masks_dir = out / "masks"

    if "mine" in stages:
        review = stage_mine(a, ask, out)
    if "review" in stages:
        approved = stage_review(a, ask, review)
    if "crops" in stages:
        crops_dir = stage_crops(a, ask, review, out, approved)

    matting_src = review / "images" if a.mask_whole_frames else crops_dir
    if "masks" in stages:
        if not list_images(matting_src):
            sys.exit(f"nothing to matte in {matting_src}")
        masks_dir = stage_masks(a, ask, matting_src, out)
    if "final" in stages:
        stage_final(a, ask, matting_src, masks_dir, out)

    head("done")
    for label, path in (("review set", review), ("crops", crops_dir),
                        ("masks", masks_dir), ("final images", out / "final"),
                        ("previews", out / "preview")):
        if path.exists():
            print(f"  {label:<13} {path}  ({len(list(path.glob('*.*')))} files)")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\ninterrupted")
        sys.exit(130)
