#!/usr/bin/env python3
"""Mine a diverse set of person crops out of recordings.

Pipeline
    1. detect + TRACK people (ultralytics ByteTrack) so one person in one shot
       is a single identity, not 300 near-identical frames
    2. quality metrics per candidate: size, truncation, occlusion, sharpness,
       exposure, motion
    3. dHash near-duplicate rejection — identical/near-identical crops collapse
       to one, keeping the sharpest
    4. diversity descriptor: normalized pose keypoints (body configuration) +
       CLIP embedding (clothing, colour, lighting, background) + scalar context
       (scale, aspect, sharpness, exposure, motion, occlusion), each z-scored
    5. selection: cap per track (with a minimum frame gap), then greedy
       farthest-point sampling to the budget, with a reserved quota for hard
       examples (blurry / small / occluded / low confidence)
    6. re-decode only the selected frames and write them as a review set:
       <out>/images + <out>/labels (YOLO txt) + manifest.json

The review set opens directly in bbox_annotator.py, where boxes get fixed,
frames approved or rejected, and the approved crops exported.

    python mine_person_crops.py --source recordings/ --out review_set --budget 400
    python bbox_annotator.py --images review_set/images   # approve, then export

Needs weights/yolo11x.pt and weights/yolo11x-pose.pt (bbox_annotator.py
downloads the detector set on first run; pose is fetched the same way).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

VIDEO_EXT = {".mp4", ".mov", ".mkv", ".avi", ".m4v", ".mpg", ".mpeg", ".webm", ".ts"}
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
KPTS = 17                      # COCO pose


# --------------------------------------------------------------- descriptors

def normalize_pose(kxy: np.ndarray, kconf: np.ndarray) -> np.ndarray:
    """COCO keypoints -> 35-d translation/scale invariant descriptor.

    Centred on mid-hip, scaled by torso length, so distance in this space means
    'different body configuration' rather than different position or size.
    """
    out = np.zeros(KPTS * 2 + 1, np.float32)
    if kxy is None or len(kxy) < KPTS:
        return out
    vis = kconf > 0.3
    if vis.sum() < 5:
        return out
    l_hip, r_hip, l_sh, r_sh = 11, 12, 5, 6
    hips = [i for i in (l_hip, r_hip) if vis[i]]
    shs = [i for i in (l_sh, r_sh) if vis[i]]
    if not hips or not shs:
        centre = kxy[vis].mean(0)
        scale = max(kxy[vis].std(0).mean(), 1e-3)
    else:
        centre = kxy[hips].mean(0)
        scale = max(float(np.linalg.norm(kxy[shs].mean(0) - centre)), 1e-3)
    p = (kxy - centre) / scale
    p[~vis] = 0.0
    out[:KPTS * 2] = p.reshape(-1)
    out[-1] = float(vis.mean())
    return out


# COCO indices: head-ish points and the two ankles decide "whole person visible"
HEAD_KPTS = (0, 1, 2, 3, 4)
ANKLES = (15, 16)
KNEES = (13, 14)


def body_extent(kxy: np.ndarray, kconf: np.ndarray, thr=0.3
                ) -> tuple[list[float] | None, bool]:
    """Bounding box of the visible keypoints, and whether the body is complete.

    The detector box often clips feet or the top of the head; the keypoint hull
    is what tells us where the person actually ends.
    """
    if kxy is None or len(kxy) < KPTS:
        return None, False
    vis = kconf > thr
    if vis.sum() < 5:
        return None, False
    pts = kxy[vis]
    box = [float(pts[:, 0].min()), float(pts[:, 1].min()),
           float(pts[:, 0].max()), float(pts[:, 1].max())]
    head = any(vis[i] for i in HEAD_KPTS)
    feet = all(vis[i] for i in ANKLES) or any(vis[i] for i in ANKLES)
    knees = any(vis[i] for i in KNEES)
    return box, bool(head and feet and knees)


def dhash(crop: np.ndarray, size: int = 8) -> int:
    """64-bit difference hash: gradient direction between adjacent pixels.

    Two crops of the same person in the same posture and lighting collapse to
    the same (or a near-identical) hash, which is the cheapest possible way to
    throw away frames that add nothing.
    """
    if crop.size == 0:
        return 0
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, (size + 1, size), interpolation=cv2.INTER_AREA)
    bits = (g[:, 1:] > g[:, :-1]).reshape(-1)
    out = 0
    for b in bits:
        out = (out << 1) | int(b)
    return out


def hamming64(a: int, b: int) -> int:
    return int(popcount64(np.uint64(a) ^ np.uint64(b)))


def reset_tracker(model) -> None:
    """Forget every track, so ids never carry across unrelated images."""
    try:
        for t in getattr(model.predictor, "trackers", None) or []:
            t.reset()
    except Exception:                                           # noqa: BLE001
        pass


class ShotSplitter:
    """Splits a source into units of genuinely continuous footage.

    A folder can hold a frame sequence, unrelated stills, or both. Track ids
    only mean 'the same person' inside one continuous run, so whenever two
    consecutive frames look nothing alike we start a new unit and wipe the
    tracker — otherwise ByteTrack happily gives two different people the same
    id because they stand in similar places, and the per-track cap then throws
    away distinct people.
    """

    def __init__(self, max_dist: int, enabled: bool = True):
        self.max_dist = max_dist
        self.enabled = enabled
        self.unit = 0
        self.prev: int | None = None
        self.cuts = 0

    def feed(self, frame: np.ndarray) -> bool:
        """Returns True when this frame starts a new unit."""
        if not self.enabled:
            return False
        h = dhash(frame)
        cut = self.prev is not None and hamming64(self.prev, h) > self.max_dist
        self.prev = h
        if cut:
            self.unit += 1
            self.cuts += 1
        return cut


def popcount64(x: np.ndarray) -> np.ndarray:
    if hasattr(np, "bitwise_count"):
        return np.bitwise_count(x)
    c = np.zeros(x.shape, np.uint8)                # numpy < 2.0 fallback
    v = x.copy()
    while v.any():
        c += (v & np.uint64(1)).astype(np.uint8)
        v >>= np.uint64(1)
    return c


def dedupe_by_hash(cands: list[dict], idxs: list[int], max_dist: int,
                   ) -> tuple[list[int], int]:
    """Drop near-duplicates, keeping the sharpest crop of each duplicate group.

    Hamming distance on the dHash; max_dist 0 means exact duplicates only.
    """
    order = sorted(idxs, key=lambda i: -cands[i]["sharp"])   # sharpest first
    kept: list[int] = []
    kept_h = np.zeros(0, np.uint64)
    exact: set[int] = set()
    for i in order:
        h = int(cands[i].get("dhash", 0))
        if h in exact:
            continue
        if len(kept_h):
            d = popcount64(kept_h ^ np.uint64(h))
            if int(d.min()) <= max_dist:
                continue
        kept.append(i)
        exact.add(h)
        kept_h = np.append(kept_h, np.uint64(h))
    return sorted(kept), len(idxs) - len(kept)


def sharpness(crop: np.ndarray) -> float:
    """Laplacian variance at a fixed height, so it compares across scales."""
    if crop.size == 0:
        return 0.0
    h = 128
    c = cv2.resize(crop, (max(8, int(crop.shape[1] * h / max(crop.shape[0], 1))), h))
    g = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def iou_xyxy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU of box a against every box in b."""
    if len(b) == 0:
        return np.zeros(0, np.float32)
    x1 = np.maximum(a[0], b[:, 0]); y1 = np.maximum(a[1], b[:, 1])
    x2 = np.minimum(a[2], b[:, 2]); y2 = np.minimum(a[3], b[:, 3])
    iw = np.clip(x2 - x1, 0, None); ih = np.clip(y2 - y1, 0, None)
    inter = iw * ih
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a + area_b - inter, 1e-6)


def zblock(x: np.ndarray, w: float) -> np.ndarray:
    """Z-score a descriptor block so no block dominates the distance."""
    if x.size == 0:
        return x
    sd = x.std(0)
    sd[sd < 1e-6] = 1.0
    return (x - x.mean(0)) / sd * w


def farthest_point_sample(feat: np.ndarray, k: int,
                          seed_idx: int | None = None,
                          return_radii: bool = False):
    """Greedy max-min sampling: each pick is the point furthest from the set.

    Covers the tail of the distribution, which is exactly what k-means throws
    away — rare poses and unusual lighting are the point of the exercise.

    With return_radii=True also returns, after every pick, the distance from
    the picked set to the candidate still worst covered by it. That number only
    falls, and how fast it falls is what "how much variety is left in this
    footage" looks like numerically — auto_budget() reads the curve.
    """
    n = len(feat)
    if k <= 0 or n == 0:
        return ([], []) if return_radii else []
    if k >= n and not return_radii:
        return list(range(n))
    k = min(k, n)          # a caller asking for radii wants the whole curve
    start = int(np.argmax(np.linalg.norm(feat - feat.mean(0), axis=1))) \
        if seed_idx is None else seed_idx
    picked = [start]
    d = np.linalg.norm(feat - feat[start], axis=1)
    radii = [float(d.max())]
    for _ in range(k - 1):
        i = int(np.argmax(d))
        picked.append(i)
        d = np.minimum(d, np.linalg.norm(feat - feat[i], axis=1))
        radii.append(float(d.max()))
    return (picked, radii) if return_radii else picked


def auto_budget(feat: np.ndarray, args) -> tuple[int, str]:
    """How many frames this footage is actually worth — no fixed number.

    Keeps taking the most-different frame until every frame left is within
    --auto-frac of the initial spread of something already taken, i.e. until
    the next pick would be a near-repeat of what the set already covers.
    Ten minutes of one person in one pose saturates in a handful of frames;
    a busy multi-camera match keeps going.
    """
    n = len(feat)
    if n <= args.auto_min:
        return n, f"only {n} candidate(s) left, keeping all"
    k_max = min(n, args.auto_max)
    _, radii = farthest_point_sample(feat, k_max, return_radii=True)
    if not radii:
        return n, f"only {n} candidate(s) left, keeping all"

    r0 = radii[0]
    stop = args.auto_frac * r0
    k = k_max
    why = (f"coverage still {radii[-1] / r0:.2f} of the spread at the "
           f"--auto-max ceiling")
    for j, r in enumerate(radii):
        if r <= stop:
            k = j + 1
            why = (f"variety saturates here — every remaining frame is within "
                   f"{args.auto_frac:.2f} of the spread of one already picked")
            break
    k = int(min(k_max, max(args.auto_min, k)))
    return k, why


# ------------------------------------------------------------------- sources

def list_sources(src: Path) -> tuple[list[Path], list[Path]]:
    """Return (video files, image-sequence dirs)."""
    if not src.exists():
        raise ValueError(f"source does not exist: {src}")
    if src.is_file():
        return ([src], []) if src.suffix.lower() in VIDEO_EXT else ([], [src.parent])
    vids = sorted(p for p in src.rglob("*") if p.suffix.lower() in VIDEO_EXT)
    imgdirs = []
    if any(p.suffix.lower() in IMG_EXT for p in src.iterdir() if p.is_file()):
        imgdirs.append(src)
    for d in sorted(p for p in src.rglob("*") if p.is_dir()):
        if any(q.suffix.lower() in IMG_EXT for q in d.iterdir() if q.is_file()):
            imgdirs.append(d)
    return vids, imgdirs


class ClipEncoder:
    """Optional CLIP appearance embedding (clothing / lighting / background)."""

    def __init__(self, device: str):
        self.ok = False
        try:
            import clip
            import torch
            self.torch = torch
            self.model, self.pre = clip.load("ViT-B/32", device=device)
            self.model.eval()
            self.device = device
            self.ok = True
        except Exception as e:                                  # noqa: BLE001
            print(f"[clip] disabled ({e}); using pose + scalars only")

    def encode(self, crops: list[np.ndarray]) -> np.ndarray:
        from PIL import Image
        if not self.ok or not crops:
            return np.zeros((len(crops), 0), np.float32)
        with self.torch.no_grad():
            batch = self.torch.stack([
                self.pre(Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)))
                for c in crops]).to(self.device)
            f = self.model.encode_image(batch).float()
            f = f / f.norm(dim=-1, keepdim=True)
        return f.cpu().numpy().astype(np.float32)


# ------------------------------------------------------------------ pass one

def scan(args, on_progress=None) -> tuple[list[dict], dict]:
    """on_progress(source_name, frame_index, n_candidates) -> bool.

    Returning False from the callback stops the scan (used by the GUI's Stop).
    """
    from ultralytics import YOLO

    det = YOLO(args.det_model)
    pose = YOLO(args.pose_model) if args.pose_model else None
    clip_enc = ClipEncoder(args.device) if not args.no_clip else None

    vids, imgdirs = list_sources(Path(args.source))
    if not vids and not imgdirs:
        raise ValueError(f"no videos or images under {args.source}")
    print(f"[scan] {len(vids)} video(s), {len(imgdirs)} image folder(s)")

    cands: list[dict] = []
    last_centre: dict[tuple[str, int], tuple[int, np.ndarray]] = {}
    t0 = time.time()
    n_units = 0

    for src in vids + imgdirs:
        key = str(src)
        is_video = src in vids
        split = ShotSplitter(args.cut_dhash, not args.no_cut_detect)
        if is_video:
            stream = det.track(source=str(src), classes=[0], conf=args.min_conf,
                               iou=0.5, stream=True, verbose=False,
                               vid_stride=args.stride, tracker=args.tracker,
                               device=args.device, imgsz=args.imgsz)
        else:
            # ultralytics resets the tracker for every new file path, so an
            # image sequence has to be fed frame by frame with persist=True or
            # every frame invents brand-new track ids. Unrelated stills get the
            # tracker wiped first, so no id is ever shared across them.
            def _seq(folder: Path, sp: ShotSplitter):
                files = sorted(p for p in folder.iterdir()
                               if p.is_file() and p.suffix.lower() in IMG_EXT)
                for p in files:
                    img = cv2.imread(str(p))
                    if img is None:
                        continue
                    if sp.feed(img):
                        reset_tracker(det)
                    r = det.track(img, classes=[0], conf=args.min_conf, iou=0.5,
                                  persist=True, verbose=False,
                                  tracker=args.tracker, device=args.device,
                                  imgsz=args.imgsz)[0]
                    r.path = str(p)
                    yield r
            stream = _seq(src, split)
        n_frames = 0
        for i, res in enumerate(stream):
            frame = res.orig_img
            if frame is None:
                continue
            n_frames += 1
            if is_video and split.feed(frame):
                reset_tracker(det)     # a cut: ids must not cross it
            unit = f"{key}#{split.unit}"
            fh, fw = frame.shape[:2]
            b = res.boxes
            if b is None or len(b) == 0:
                continue
            xyxy = b.xyxy.cpu().numpy()
            confs = b.conf.cpu().numpy()
            ids = (b.id.cpu().numpy().astype(int) if b.id is not None
                   else np.arange(len(xyxy)) + i * 1000)
            frame_idx = i * (args.stride if is_video else 1)
            fpath = key if is_video else str(res.path)

            rows, crops = [], []
            for j, (bx, cf, tid) in enumerate(zip(xyxy, confs, ids)):
                x1, y1, x2, y2 = bx
                bh, bw = y2 - y1, x2 - x1
                if bh < args.min_height or bw < 8:
                    continue
                pad_x, pad_y = bw * args.margin, bh * args.margin
                cx1 = int(max(0, x1 - pad_x)); cy1 = int(max(0, y1 - pad_y))
                cx2 = int(min(fw, x2 + pad_x)); cy2 = int(min(fh, y2 + pad_y))
                crop = frame[cy1:cy2, cx1:cx2]
                if crop.size == 0:
                    continue
                others = np.delete(xyxy, j, axis=0)
                occl = float(iou_xyxy(bx, others).max()) if len(others) else 0.0
                edge = min(x1, y1, fw - x2, fh - y2)
                hsv = cv2.cvtColor(cv2.resize(crop, (32, 64)), cv2.COLOR_BGR2HSV)
                ckey = (unit, int(tid))
                prev = last_centre.get(ckey)
                centre = np.array([(x1 + x2) / 2, (y1 + y2) / 2], np.float32)
                motion = (float(np.linalg.norm(centre - prev[1]) /
                                max(frame_idx - prev[0], 1) / max(bh, 1))
                          if prev else 0.0)
                last_centre[ckey] = (frame_idx, centre)
                rows.append(dict(
                    source=fpath, unit=unit, is_video=is_video, frame=frame_idx,
                    frame_w=fw, frame_h=fh, crop_origin=(cx1, cy1),
                    full_body=None,
                    track=int(tid), box=[float(v) for v in bx],
                    conf=float(cf), h_rel=float(bh / fh),
                    aspect=float(bw / max(bh, 1e-6)),
                    sharp=sharpness(crop), dhash=dhash(crop),
                    bright=float(hsv[..., 2].mean() / 255),
                    sat=float(hsv[..., 1].mean() / 255),
                    occl=occl, edge_px=float(edge), motion=motion))
                crops.append(crop)

            if not rows:
                continue

            if pose is not None:
                pres = pose.predict(crops, verbose=False, device=args.device,
                                    imgsz=args.pose_imgsz)
                for r, pr in zip(rows, pres):
                    kp = pr.keypoints
                    if kp is None or kp.xy is None or len(kp.xy) == 0:
                        r["pose"] = np.zeros(KPTS * 2 + 1, np.float32).tolist()
                        continue
                    areas = [float((k[:, 0].max() - k[:, 0].min()) *
                                   (k[:, 1].max() - k[:, 1].min()))
                             if len(k) else 0.0 for k in kp.xy.cpu().numpy()]
                    m = int(np.argmax(areas))          # the main person in the crop
                    kxy = kp.xy.cpu().numpy()[m]
                    kcf = (kp.conf.cpu().numpy()[m] if kp.conf is not None
                           else np.ones(len(kxy), np.float32))
                    r["pose"] = normalize_pose(kxy, kcf).tolist()
                    # keypoints are in crop space -> back to frame space
                    kbox, full = body_extent(kxy, kcf)
                    r["full_body"] = full
                    if kbox is not None:
                        ox, oy = r["crop_origin"]
                        kb = [kbox[0] + ox, kbox[1] + oy, kbox[2] + ox, kbox[3] + oy]
                        r["kpt_box"] = kb
                        # union: never let a clipped detector box cut off feet
                        x1, y1, x2, y2 = r["box"]
                        r["box"] = [max(0.0, min(x1, kb[0])),
                                    max(0.0, min(y1, kb[1])),
                                    min(float(r["frame_w"]), max(x2, kb[2])),
                                    min(float(r["frame_h"]), max(y2, kb[3]))]
            else:
                for r in rows:
                    r["pose"] = np.zeros(KPTS * 2 + 1, np.float32).tolist()
                    r["full_body"] = None      # unknown without a pose model

            if clip_enc is not None and clip_enc.ok:
                emb = clip_enc.encode(crops)
                for r, e in zip(rows, emb):
                    r["clip"] = e.tolist()

            cands.extend(rows)
            if len(cands) % 500 < len(rows):
                print(f"\r[scan] {Path(key).name}  frame {frame_idx}  "
                      f"{len(cands)} candidates  {time.time() - t0:.0f}s",
                      end="", flush=True)
            if on_progress is not None and n_frames % 5 == 0:
                if on_progress(Path(key).name, frame_idx, len(cands)) is False:
                    print("\n[scan] stopped by caller")
                    return cands, dict(n_candidates=len(cands),
                                       seconds=round(time.time() - t0, 1),
                                       clip=bool(clip_enc and clip_enc.ok),
                                       pose=bool(pose), stopped=True)
        n_units += split.unit + 1
        print(f"\r[scan] {Path(key).name}: {n_frames} frames read, "
              f"{split.cuts} break(s) -> {split.unit + 1} continuous unit(s), "
              f"{len(cands)} candidates so far{' ' * 10}")

    meta = dict(n_candidates=len(cands), seconds=round(time.time() - t0, 1),
                clip=bool(clip_enc and clip_enc.ok), pose=bool(pose),
                units=n_units)
    return cands, meta


# ------------------------------------------------------------------ pass two

def build_features(cands: list[dict], args) -> np.ndarray:
    pose = np.array([c["pose"] for c in cands], np.float32)
    scal = np.array([[c["h_rel"], c["aspect"], np.log1p(c["sharp"]),
                      c["bright"], c["sat"], c["motion"], c["occl"]]
                     for c in cands], np.float32)
    blocks = [zblock(pose, args.w_pose), zblock(scal, args.w_scalar)]
    if "clip" in cands[0]:
        clip = np.array([c["clip"] for c in cands], np.float32)
        blocks.append(zblock(clip, args.w_clip))
    return np.concatenate(blocks, 1)


def is_hard(c: dict, sharp_lo: float, args) -> bool:
    return (c["sharp"] < sharp_lo or c["conf"] < args.easy_conf
            or c["occl"] > args.max_occl or c["h_rel"] < args.easy_h_rel
            or c["edge_px"] < args.edge_px)


def select(cands: list[dict], args) -> tuple[list[dict], dict]:
    """Returns (picked, stats) — stats says what each signal removed."""
    feat = build_features(cands, args)
    sharp_lo = float(np.percentile([c["sharp"] for c in cands], args.blur_pct))
    stats = {"candidates": len(cands)}

    # hard gate: things no amount of diversity makes worth training on
    keep = [i for i, c in enumerate(cands)
            if c["h_rel"] >= args.min_h_rel and c["occl"] <= args.hard_max_occl
            and c["sharp"] >= sharp_lo * args.blur_floor]
    stats["quality_gate"] = len(keep)
    if not args.allow_partial:
        # whole person only: head + knees + at least one ankle visible, and the
        # box not jammed against a frame edge (which means a truncated body)
        before = len(keep)
        keep = [i for i in keep
                if cands[i].get("full_body") is not False
                and cands[i]["edge_px"] >= args.body_edge_px]
        print(f"[select] full-body filter: {len(keep)}/{before} kept "
              f"(--allow-partial to disable)")
    stats["full_body"] = len(keep)
    print(f"[select] {len(keep)}/{len(cands)} pass the quality gate")
    if not keep:
        return [], stats

    if args.dhash_dist >= 0 and not args.no_dhash:
        keep, dropped = dedupe_by_hash(cands, keep, args.dhash_dist)
        print(f"[select] dHash: dropped {dropped} near-duplicate crop(s) "
              f"(<= {args.dhash_dist} bits apart), {len(keep)} left")
    stats["after_dhash"] = len(keep)

    # per-track cap, with a minimum frame gap so picks are not adjacent frames
    by_track: dict[tuple, list[int]] = defaultdict(list)
    for i in keep:
        by_track[(cands[i].get("unit", cands[i]["source"]),
                  cands[i]["track"])].append(i)
    capped: list[int] = []
    for idxs in by_track.values():
        idxs.sort(key=lambda i: cands[i]["frame"])
        order = farthest_point_sample(feat[idxs], min(args.per_track * 3, len(idxs)))
        chosen: list[int] = []
        for o in order:
            i = idxs[o]
            if all(abs(cands[i]["frame"] - cands[j]["frame"]) >= args.min_gap
                   for j in chosen):
                chosen.append(i)
            if len(chosen) >= args.per_track:
                break
        capped.extend(chosen)
    print(f"[select] {len(capped)} after per-track cap "
          f"({len(by_track)} tracks, max {args.per_track} each)")
    stats["units"] = len({k[0] for k in by_track})
    stats["tracks"] = len(by_track)
    stats["after_track_cap"] = len(capped)

    # how many to keep: either the number asked for, or the number this
    # footage actually supports before the picks start repeating themselves
    budget = args.budget
    if budget <= 0:
        budget, why = auto_budget(feat[np.array(capped)], args)
        print(f"[select] auto budget: {budget} of {len(capped)} — {why}")
    stats["budget"] = budget

    # split easy / hard so the hard examples get a guaranteed share
    hard = [i for i in capped if is_hard(cands[i], sharp_lo, args)]
    easy = [i for i in capped if i not in set(hard)]
    n_hard = min(len(hard), int(round(budget * args.hard_quota)))
    n_easy = min(len(easy), budget - n_hard)
    n_hard = min(len(hard), budget - n_easy)

    out: list[int] = []
    for pool, n in ((easy, n_easy), (hard, n_hard)):
        if pool and n > 0:
            sub = np.array(pool)
            out.extend(sub[farthest_point_sample(feat[sub], n)].tolist())
    print(f"[select] final {len(out)} = {n_easy} easy + {n_hard} hard "
          f"(budget {budget}{' auto' if args.budget <= 0 else ''})")

    picked = []
    for rank, i in enumerate(out):
        c = dict(cands[i])
        for k in ("clip", "crop_origin"):
            c.pop(k, None)
        c["dhash"] = f'{c.get("dhash", 0):016x}'      # readable, still unique
        c["rank"] = rank
        c["hard"] = i in set(hard)
        picked.append(c)
    stats.update(easy=n_easy, hard=n_hard, selected=len(picked))
    return picked, stats


def write_review_set(picked: list[dict], out: Path, args,
                     stats: dict | None = None) -> dict:
    """Re-decode only the selected frames; write images + YOLO labels."""
    img_dir, lbl_dir = out / "images", out / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    by_frame: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for c in picked:
        by_frame[(c["source"], c["frame"])].append(c)

    entries, written = [], 0
    for src in sorted({k[0] for k in by_frame}):
        frames = sorted(f for (s, f) in by_frame if s == src)
        is_vid = Path(src).suffix.lower() in VIDEO_EXT
        cap = cv2.VideoCapture(src) if is_vid else None
        for fi in frames:
            if is_vid:
                cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
                ok, frame = cap.read()
                if not ok:
                    print(f"[write] cannot seek {Path(src).name} frame {fi}")
                    continue
            else:
                frame = cv2.imread(src)
                if frame is None:
                    continue
            fh, fw = frame.shape[:2]
            name = f"{Path(src).stem}_f{fi:07d}.jpg"
            cv2.imwrite(str(img_dir / name),
                        frame, [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality])
            lines, boxes = [], []
            for c in by_frame[(src, fi)]:
                x1, y1, x2, y2 = c["box"]
                lines.append(f"0 {(x1 + x2) / 2 / fw:.6f} {(y1 + y2) / 2 / fh:.6f} "
                             f"{(x2 - x1) / fw:.6f} {(y2 - y1) / fh:.6f}")
                boxes.append({k: c.get(k) for k in
                              ("track", "conf", "h_rel", "sharp", "occl",
                               "bright", "motion", "hard", "rank",
                               "full_body", "dhash")})
            (lbl_dir / f"{Path(name).stem}.txt").write_text("\n".join(lines) + "\n")
            entries.append(dict(image=name, source=src, frame=fi,
                                width=fw, height=fh, boxes=boxes))
            written += 1
        if cap is not None:
            cap.release()

    (out / "classes.txt").write_text("person\n")
    manifest = dict(created=time.strftime("%Y-%m-%d %H:%M:%S"),
                    images=written, crops=len(picked), stats=stats or {},
                    args={k: (str(v) if isinstance(v, Path) else v)
                          for k, v in vars(args).items()},
                    entries=entries)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest


def budget_arg(v: str) -> int:
    """'auto' (or 0 / '') -> 0, which select() reads as 'decide for me'."""
    v = str(v).strip().lower()
    if v in ("auto", "", "0", "none"):
        return 0
    try:
        n = int(v)
    except ValueError:
        raise argparse.ArgumentTypeError(f"budget must be a number or 'auto', got {v!r}")
    if n < 0:
        raise argparse.ArgumentTypeError("budget cannot be negative")
    return n


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Mine diverse person crops from recordings",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--source", required=True,
                    help="video file, folder of videos, or folder of frames")
    ap.add_argument("--out", required=True, help="review set output folder")
    ap.add_argument("--budget", type=budget_arg, default=0,
                    help="frames to keep, or 'auto' (the default) to let the "
                         "diversity of the footage decide")
    ap.add_argument("--auto-frac", type=float, default=0.35,
                    help="'auto' stops once the worst-covered frame is within "
                         "this fraction of the initial spread (lower = smaller, "
                         "more distinct set; higher = bigger, more redundant)")
    ap.add_argument("--auto-min", type=int, default=12,
                    help="'auto' never returns fewer than this")
    ap.add_argument("--auto-max", type=int, default=2000,
                    help="'auto' never returns more than this")
    ap.add_argument("--per-track", type=int, default=6,
                    help="max crops from one tracked person")
    ap.add_argument("--min-gap", type=int, default=12,
                    help="minimum frames between two picks of one track")

    g = ap.add_argument_group("models")
    g.add_argument("--det-model", default="weights/yolo11x.pt")
    g.add_argument("--pose-model", default="weights/yolo11x-pose.pt",
                   help="'' to skip pose")
    g.add_argument("--no-clip", action="store_true", help="skip CLIP embedding")
    g.add_argument("--device", default="")
    g.add_argument("--tracker", default="bytetrack.yaml")
    g.add_argument("--imgsz", type=int, default=1280)
    g.add_argument("--pose-imgsz", type=int, default=256)
    g.add_argument("--stride", type=int, default=3, help="read every Nth frame")

    g = ap.add_argument_group("quality gate")
    g.add_argument("--min-conf", type=float, default=0.3)
    g.add_argument("--min-height", type=int, default=96, help="box px")
    g.add_argument("--min-h-rel", type=float, default=0.05, help="box h / frame h")
    g.add_argument("--hard-max-occl", type=float, default=0.65)
    g.add_argument("--blur-pct", type=float, default=15.0,
                   help="sharpness percentile treated as 'blurry'")
    g.add_argument("--blur-floor", type=float, default=0.4,
                   help="reject below blur-pct * this")
    g.add_argument("--margin", type=float, default=0.25,
                   help="context padding used for pose/CLIP crops")
    g.add_argument("--allow-partial", action="store_true",
                   help="keep people whose full body is not visible")
    g.add_argument("--body-edge-px", type=float, default=2.0,
                   help="min distance from frame edge for a full-body crop")
    g.add_argument("--dhash-dist", type=int, default=6,
                   help="drop a crop within this Hamming distance (of 64) of a "
                        "kept one; 0 = exact duplicates only")
    g.add_argument("--no-dhash", action="store_true",
                   help="keep near-duplicate crops")
    g.add_argument("--cut-dhash", type=int, default=22,
                   help="frame-to-frame dHash distance above which footage is "
                        "treated as unrelated (new unit, tracker wiped)")
    g.add_argument("--no-cut-detect", action="store_true",
                   help="treat every source as one continuous sequence")

    g = ap.add_argument_group("hard-example quota")
    g.add_argument("--hard-quota", type=float, default=0.15)
    g.add_argument("--easy-conf", type=float, default=0.6)
    g.add_argument("--easy-h-rel", type=float, default=0.12)
    g.add_argument("--max-occl", type=float, default=0.15)
    g.add_argument("--edge-px", type=float, default=3.0)

    g = ap.add_argument_group("descriptor weights")
    g.add_argument("--w-pose", type=float, default=1.0)
    g.add_argument("--w-clip", type=float, default=1.0)
    g.add_argument("--w-scalar", type=float, default=0.5)
    ap.add_argument("--jpeg-quality", type=int, default=95)
    ap.add_argument("--cache", help="write raw candidate metadata here (json)")
    return ap


def main():
    args = build_parser().parse_args()

    if not args.device:
        try:
            import torch
            args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        except ImportError:
            args.device = "cpu"
    if args.pose_model in ("", "none", "None"):
        args.pose_model = None

    try:
        cands, meta = scan(args)
    except ValueError as e:
        sys.exit(str(e))
    print(f"[scan] {meta['n_candidates']} candidates in {meta['seconds']}s "
          f"(pose={meta['pose']}, clip={meta['clip']})")
    if not cands:
        sys.exit("no person detections passed --min-height / --min-conf")
    if args.cache:
        Path(args.cache).write_text(json.dumps(cands))

    picked, stats = select(cands, args)
    if not picked:
        sys.exit("nothing survived the quality gate; loosen --min-h-rel/--blur-floor")

    out = Path(args.out)
    man = write_review_set(picked, out, args, stats)
    print(f"\n[done] {man['images']} frames / {man['crops']} person boxes "
          f"-> {out}")
    print("       " + " -> ".join(f"{k} {v}" for k, v in stats.items()))
    print(f"       review them with:\n"
          f"       python bbox_annotator.py --images {out / 'images'}")


if __name__ == "__main__":
    main()
