"""
EgoLoc 3‑D Demo – *lite* edition
Assumed repo layout (relative to this file):
EgoLoc/
│
├── ./                    ← this script lives here (anywhere inside EgoLoc)
├── HaMeR/                ← git clone https://github.com/geopavlakos/hamer.git
└── Video-Depth-Anything/ ← git clone https://github.com/DepthAnything/Video-Depth-Anything.git
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import argparse
import base64
import json
import math
import os
import subprocess
import time
import warnings
from pathlib import Path

import cv2
import dotenv  # read .env creds for GPT-4o
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage  # connected‑component helper
from scipy.ndimage import gaussian_filter1d
from typing import List, Dict, Tuple, Optional, Any

try:
    import torch
except ImportError:  # Allow import on machines without torch
    torch = None

# ---------------------------------------------------------------------------
# External repos that *must* exist inside EgoLoc root
# ---------------------------------------------------------------------------
try:
    import hamer  # type: ignore
except ImportError as e:
    raise ImportError(
        "HaMeR repo not found.  Make sure you cloned it inside the EgoLoc root."
    ) from e

try:
    from hamer.vitpose_model import ViTPoseModel  # type: ignore
except ImportError as e:
    raise ImportError(
        "vitpose_model.py not found.  It ships with HaMeR; "
        "confirm your PYTHONPATH contains that folder."
    ) from e

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
plt.switch_backend("Agg")  # headless plotting

# ---------------------------------------------------------------------------
# Helper – unpack Video‑Depth‑Anything *_depths.npz → per‑frame .npy
# ---------------------------------------------------------------------------


def _unpack_depth_npz(depth_dir: Path) -> None:
    """Convert VDA's combined depth archive into individual per‑frame tensors."""
    npz_files = list(depth_dir.glob("*_depths.npz"))
    if not npz_files:
        return  # nothing to unpack

    arr = np.load(npz_files[0])["depths"]  # (N, H, W)
    for i, depth in enumerate(arr):
        np.save(depth_dir / f"pred_depth_{i:06d}.npy", depth.astype(np.float32))
    print(f"[VDA] Unpacked {len(arr)} frames to .npy tensors")


# ---------------------------------------------------------------------------
# Paths & repo‑root helpers
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = next(
    p
    for p in [SCRIPT_DIR] + list(SCRIPT_DIR.parents)
    if (p / "Video-Depth-Anything").exists()
)

VDA_DIR = REPO_ROOT / "Video-Depth-Anything"
# VDA_CHECKPOINT_PATH = Path(
#     os.getenv(
#         "VDA_CHECKPOINT_PATH", VDA_DIR / "checkpoints" / "video_depth_anything_vits.pth"
#     )
# ).resolve()

# if not VDA_CHECKPOINT_PATH.exists():
#     raise FileNotFoundError(
#         f"Video-Depth-Anything checkpoint not found at {VDA_CHECKPOINT_PATH}"
#     )

DEPTH_SCALE_M = 3.0  # pixel value 255 ↔ 3 m (linear scaling)
MAX_FEEDBACKS = 20


# ====== basic helpers (speed JSON, frame resize, etc.) =====================
def get_json_path(video_name: str, base_dir: str):
    return os.path.join(base_dir, f"{video_name}_speed.json")


# --------------------------------------------------------------------------
#                           IMAGE HELPERS
# --------------------------------------------------------------------------


def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    """Resize *image* while preserving aspect ratio."""
    if width is None and height is None:
        return image
    (h, w) = image.shape[:2]
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    return cv2.resize(image, dim, interpolation=inter)


def image_resize_for_vlm(frame, inter=cv2.INTER_AREA):
    """Resize an image so the short side ≤ 768 px and long side ≤ 2000 px."""
    h, w = frame.shape[:2]
    aspect = w / h
    max_short, max_long = 768, 2000
    if aspect > 1:
        new_w = min(w, max_long)
        new_h = int(new_w / aspect)
        if new_h > max_short:
            new_h = max_short
            new_w = int(new_h * aspect)
    else:
        new_h = min(h, max_long)
        new_w = int(new_h * aspect)
        if new_w > max_short:
            new_w = max_short
            new_h = int(new_w / aspect)
    return cv2.resize(frame, (new_w, new_h), interpolation=inter)


# --------------------------------------------------------------------------
#                     VISION–LANGUAGE MODEL WRAPPER
# --------------------------------------------------------------------------


def extract_json_part(text_output: str) -> Optional[str]:
    """Extract the JSON fragment {"points":[...]} from GPT text output."""
    text = text_output.strip().replace(" ", "").replace("\n", "")
    try:
        idx = text.index('{"points":')
        frag = text[idx:]
        end = frag.index("}") + 1
        return frag[:end]
    except ValueError:
        print("Text received:", text_output)
        return None

def scene_understanding(credentials: Dict[str, Any], frame: np.ndarray, prompt: str, *, flag: Optional[str] = None, raw: bool = False):
    """
    Vision-language helper.

    ── Return modes ───────────────────────────────────────────────
    • default ( flag is None  and  raw is False ):
        → (first_point, full_text)
    • feedback / “state” checks ( flag is not None  or  raw is True ):
        → full_text  (str)
    """

    frame = image_resize_for_vlm(frame)
    _, buf = cv2.imencode(".jpg", frame)
    b64 = base64.b64encode(buf).decode("utf-8")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{b64}",
                        "detail": "high",
                    },
                },
            ],
        }
    ]

    import openai

    if credentials.get("AZURE_OPENAI_API_KEY"):
        from openai import AzureOpenAI

        client = AzureOpenAI(
            api_version="2024-02-01",
            azure_endpoint=credentials["AZURE_OPENAI_ENDPOINT"],
            api_key=credentials["AZURE_OPENAI_API_KEY"],
        )
        params = {"model": credentials["AZURE_OPENAI_DEPLOYMENT_NAME"]}
    else:
        from openai import OpenAI

        client = OpenAI(
            api_key=credentials["OPENAI_API_KEY"],
            base_url="https://api.chatanywhere.tech/v1",
        )
        params = {"model": "gpt-4o"}

    params.update(dict(messages=messages, max_tokens=200, temperature=0.1, top_p=0.5))

    retries = 0
    while retries < 5:
        try:
            result = client.chat.completions.create(**params)
            break
        except (openai.RateLimitError, openai.APIStatusError):
            time.sleep(2)
            retries += 1
    else:
        raise RuntimeError("Failed to obtain GPT-4o response.")

    content = result.choices[0].message.content

    # ── raw-text mode ────────────────────────────────────────────
    if raw or flag is not None:
        return content

    # ── JSON-parsing mode ───────────────────────────────────────
    frag = extract_json_part(content)
    if frag is None:
        return -1, content
    try:
        pts = json.loads(frag)["points"]
        return (pts[0] if pts else -1), content
    except Exception:  # malformed JSON or no "points"
        return -1, content


# --------------------------------------------------------------------------
#        GRID BUILDERS & FRAME-SELECTION UTILITIES  (verbatim from 2-D)
# --------------------------------------------------------------------------


def create_frame_grid_with_keyframe(video_path: str, frame_indices: List[int], grid_size: int) -> np.ndarray:
    """Return a numbered frame grid (BGR uint8)."""
    spacer = 0
    cap = cv2.VideoCapture(video_path)
    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frm = cap.read()
        if not ok:
            frm = (
                np.zeros_like(frames[0])
                if frames
                else np.zeros((200, 200, 3), np.uint8)
            )
        frm = image_resize(frm, width=200)
        frames.append(frm)
    cap.release()

    while len(frames) < grid_size**2:
        frames.append(np.zeros_like(frames[0]))

    fh, fw = frames[0].shape[:2]
    grid_h = grid_size * fh + (grid_size - 1) * spacer
    grid_w = grid_size * fw + (grid_size - 1) * spacer
    grid = np.ones((grid_h, grid_w, 3), np.uint8) * 255

    for i in range(grid_size):
        for j in range(grid_size):
            k = i * grid_size + j
            frm = frames[k]
            max_d = int(min(frm.shape[:2]) * 0.5)
            cc = (frm.shape[1] - max_d // 2, max_d // 2)
            overlay = frm.copy()
            cv2.circle(overlay, cc, max_d // 2, (255, 255, 255), -1)
            frm = cv2.addWeighted(overlay, 0.3, frm, 0.7, 0)
            cv2.circle(frm, cc, max_d // 2, (255, 255, 255), 2)
            fs = max_d / 50
            txtsz = cv2.getTextSize(str(k + 1), cv2.FONT_HERSHEY_SIMPLEX, fs, 2)[0]
            tx = frm.shape[1] - txtsz[0] // 2 - max_d // 2
            ty = txtsz[1] // 2 + max_d // 2
            cv2.putText(
                frm, str(k + 1), (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), 2
            )
            y1, y2 = i * (fh + spacer), (i + 1) * fh + i * spacer
            x1, x2 = j * (fw + spacer), (j + 1) * fw + j * spacer
            grid[y1:y2, x1:x2] = frm
    return grid


def select_top_n_frames_from_json(json_path: str, n: int, frame_index: Optional[int] = None, flag: Optional[str] = None, receive_flag: Optional[str] = None):
    """Return indices with lowest speed; behaviour altered by *flag*."""
    with open(json_path, "r") as f:
        data = json.load(f)
    items = list(data.items())
    if frame_index is None:
        valid = [(int(i), s) for i, s in items if s != 0 and not math.isnan(s)]
    else:
        if flag == "feedback":
            valid = [
                (int(i), s)
                for i, s in items
                if s != 0 and not math.isnan(s) and int(i) > frame_index
            ]
            inval = [
                int(i)
                for i, s in items
                if s != 0 and not math.isnan(s) and int(i) <= frame_index
            ]
        elif flag == "speed":
            valid = [
                (int(i), s)
                for i, s in items
                if s != 0 and not math.isnan(s) and s < frame_index
            ]
            inval = [
                int(i)
                for i, s in items
                if s != 0 and not math.isnan(s) and s >= frame_index
            ]
        else:
            valid = [
                (int(i), s)
                for i, s in items
                if s != 0 and not math.isnan(s) and int(i) != frame_index
            ]
            inval = [
                int(i)
                for i, s in items
                if s != 0 and not math.isnan(s) and int(i) == frame_index
            ]
    top = [idx for idx, _ in sorted(valid, key=lambda x: x[1])[:n]]
    if receive_flag is None:
        return top
    return inval, top


def select_frames_near_average(indices: List[int], grid_size: int, total_frames: int, invalid: List[int], min_index: Optional[int] = None):
    """Return *grid_size²* unique frames centred on average of *indices*."""
    avg = round(np.mean(indices))
    used = []
    if avg not in invalid and (min_index is None or avg > min_index):
        used.append(avg)
    left, right = avg - 1, avg + 1
    while len(used) < grid_size**2:
        if left >= 0:
            if left not in invalid and (min_index is None or left > min_index):
                used.insert(0, left)
            left -= 1
        if len(used) >= grid_size**2:
            break
        if right < total_frames:
            if right not in invalid and (min_index is None or right > min_index):
                used.append(right)
            right += 1
    return used[: grid_size**2]


def select_and_filter_keyframes_with_anchor(sel: List[int], total_idx: List[int], grid_size: int, anchor: str, video_path: str):
    """Keep frames in first/second half depending on *anchor* ('start'/'end')."""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if anchor == "start":
        filtered = [i for i in sel if i < total_frames // 2]
        if len(filtered) < grid_size:
            extra = [
                i for i in total_idx if i not in filtered and i < total_frames // 2
            ]
            filtered.extend(extra[: grid_size - len(filtered)])
    elif anchor == "end":
        filtered = [i for i in sel if i >= total_frames // 2]
        if len(filtered) < grid_size:
            extra = [
                i for i in total_idx if i not in filtered and i >= total_frames // 2
            ]
            filtered.extend(extra[: grid_size - len(filtered)])
    else:
        raise ValueError("anchor must be 'start' or 'end'")
    return sorted(filtered)


# --------------------------------------------------------------------------
#             FEEDBACK + TASK-SELECTION (verbatim logic)
# --------------------------------------------------------------------------

def determine_by_state(credentials, video_path, action, grid_size, total_frames, frame_index, anchor, speed_folder):
    """Ask GPT-4o if *frame_index* is valid contact/separation."""
    prompt = (
        "I will show an image of hand-object interaction. "
        "You need to help me determine whether the hand and the object "
        "in the current image are in obvious contact "
        if anchor == "start"
        else "I will show an image of hand-object interaction. "
        "You need to help me determine whether the hand and the object "
        "are clearly separate. "
    ) + "If yes, answer 1; if no, answer 0"

    grid = create_frame_grid_with_keyframe(video_path, [frame_index], 1)
    res = scene_understanding(credentials, grid, prompt, raw=True)
    if res == "1" or frame_index > total_frames - 5:
        return frame_index
    return process_task(
        credentials,
        video_path,
        action,
        grid_size,
        total_frames,
        anchor,
        speed_folder,
        frame_index,
        flag="feedback",
    )


def determine_by_speed(credentials, video_path, action, grid_size, total_frames, frame_index, anchor, speed_folder):
    """Reject frames whose speed exceeds 30-percentile threshold."""
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    json_path = get_json_path(video_name, speed_folder)
    with open(json_path, "r") as f:
        data = json.load(f)
    valid = [(int(i), s) for i, s in data.items() if s != 0 and not math.isnan(s)]
    if not valid:
        return process_task(
            credentials,
            video_path,
            action,
            grid_size,
            total_frames,
            anchor,
            speed_folder,
            frame_index,
            flag="speed",
        )
    valid.sort(key=lambda x: x[1])
    threshold = valid[int(0.3 * len(valid))][1]
    cur_speed = data.get(str(frame_index), 0)
    if cur_speed <= threshold:
        return frame_index
    return process_task(
        credentials,
        video_path,
        action,
        grid_size,
        total_frames,
        anchor,
        speed_folder,
        frame_index,
        flag="speed",
    )


def feedback_contact(credentials, video_path, action, grid_size, total_frames, frame_start, max_fb, anchor, speed_folder):
    cnt = 0
    cur = frame_start
    while cnt < max_fb:
        new = determine_by_state(
            credentials,
            video_path,
            action,
            grid_size,
            total_frames,
            cur,
            anchor,
            speed_folder,
        )
        if new is None:
            return None
        if new == cur:
            new_sp = determine_by_speed(
                credentials,
                video_path,
                action,
                grid_size,
                total_frames,
                cur,
                anchor,
                speed_folder,
            )
            if new_sp == cur:
                return cur
            cur = new_sp
        else:
            cur = new
        cnt += 1
    return cur


def feedback_separation(credentials, video_path, action, grid_size, total_frames, frame_end, max_fb, anchor, speed_folder):
    return feedback_contact(
        credentials,
        video_path,
        action,
        grid_size,
        total_frames,
        frame_end,
        max_fb,
        anchor,
        speed_folder,
    )


def process_task(credentials, video_path, action, grid_size, total_frames, anchor, speed_folder, frame_index=None, flag=None):
    """Core routine that builds frame grid, asks GPT-4o for best frame."""
    prompt_start = (
        "I will show an image sequence of human cooking. "
        "Choose the number that is closest to the moment when the "
        f"({action}) has started. "
        "Give one-sentence analysis (<50 words). "
        "If the action is absent choose -1. "
        'Return JSON: {"points": []}'
    )
    prompt_end = (
        "I will show an image sequence of human cooking. "
        "Choose the number that is closest to the moment when the "
        f"({action}) has ended. "
        "Give one-sentence analysis (<50 words). "
        "If the action has not ended choose -1. "
        'Return JSON: {"points": []}'
    )
    prompt = prompt_start if anchor == "start" else prompt_end

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    json_path = get_json_path(video_name, speed_folder)

    if frame_index is None:
        top = select_top_n_frames_from_json(json_path, 4)
        total_idx = select_top_n_frames_from_json(json_path, total_frames)
        filt = select_and_filter_keyframes_with_anchor(
            top, total_idx, 4, anchor, video_path
        ) or sorted(top)
        used = select_frames_near_average(filt, grid_size, total_frames, [])
    else:
        if flag == "feedback":
            invalid, top = select_top_n_frames_from_json(
                json_path, 4, frame_index, flag, receive_flag="right"
            )
            total_idx = select_top_n_frames_from_json(
                json_path, total_frames, frame_index, flag
            )
            filt = select_and_filter_keyframes_with_anchor(
                top, total_idx, 4, anchor, video_path
            ) or sorted(top)
            used = select_frames_near_average(
                filt, grid_size, total_frames, invalid, min_index=frame_index
            )
        elif flag == "speed":
            invalid, top = select_top_n_frames_from_json(
                json_path, 4, frame_index, flag, receive_flag="right"
            )
            total_idx = select_top_n_frames_from_json(
                json_path, total_frames, frame_index, flag
            )
            filt = select_and_filter_keyframes_with_anchor(
                top, total_idx, 4, anchor, video_path
            ) or sorted(top)
            used = select_frames_near_average(filt, grid_size, total_frames, invalid)
        else:
            invalid, top = select_top_n_frames_from_json(
                json_path, 4, frame_index, receive_flag="right"
            )
            total_idx = select_top_n_frames_from_json(
                json_path, total_frames, frame_index
            )
            filt = select_and_filter_keyframes_with_anchor(
                top, total_idx, 4, anchor, video_path
            ) or sorted(top)
            used = select_frames_near_average(filt, grid_size, total_frames, invalid)

    grid = create_frame_grid_with_keyframe(video_path, used, grid_size)
    result = scene_understanding(credentials, grid, prompt)
    if isinstance(result, tuple):
        choice, _ = result
    else:
        choice = result
    if choice == -1:
        return None
    idx = max(0, min(int(choice) - 1, len(used) - 1))
    return used[idx]


# ---------------------------------------------------------------------------
# Depth video generation via Video‑Depth‑Anything
# ---------------------------------------------------------------------------

def generate_depth_video_vda(video_path: str, depth_out_path: str, *, device: str = "cuda", encoder: str = "vits") -> Path:
    """Run Video‑Depth‑Anything once and save the raw depth video."""
    video_path = Path(video_path).resolve()
    depth_out_path = Path(depth_out_path).resolve()
    depth_out_path.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python",
        "run.py",
        "--input_video",
        str(video_path),
        "--output_dir",
        str(depth_out_path),
        "--encoder",
        encoder,  # or "vitl" if you want the large model
        "--save_npz",  # save per‑frame tensors
    ]
    if device == "cpu":
        cmd.append("--fp32")  # avoids half‑precision on CPU

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{VDA_DIR}:{env.get('PYTHONPATH', '')}"

    print("[VDA] Running:", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=VDA_DIR, env=env)

    # Unpack & sanity‑check
    _unpack_depth_npz(depth_out_path)
    if not any(depth_out_path.glob("pred_depth_*.npy")):
        raise RuntimeError(
            "[VDA] No pred_depth_*.npy tensors were produced – check the VDA output above."
        )

    return depth_out_path


# -------------------------------------------------------------------------
# Depth loading helper
# ---------------------------------------------------------------------------


def _load_depth(depth_dir: Path, idx: int, scale_m: float = DEPTH_SCALE_M) -> Optional[np.ndarray]:
    """Read VDA’s inverse‑depth tensor for frame *idx* and convert to metres."""
    f = depth_dir / f"pred_depth_{idx:06d}.npy"
    if not f.exists():
        return None
    inv = np.load(f)  # (H, W) float32
    inv_norm = (inv - inv.min()) / (inv.max() - inv.min() + 1e-8)  # 0‑1
    depth = (1.0 - inv_norm) * scale_m
    return depth.astype(np.float32)


# ---------------------------------------------------------------------------
# Simple camera projection helper
# ---------------------------------------------------------------------------


def _pixel_to_camera(u: float, v: float, z: float, W: int, H: int):
    fx = fy = max(W, H)
    cx, cy = W / 2.0, H / 2.0
    X = (u - cx) * z / fx
    Y = (v - cy) * z / fy
    return X, Y, z


# ---------------------------------------------------------------------------
# HaMeR / ViTPose – created once, reused
# ---------------------------------------------------------------------------

_HAMER_CACHE: Dict[str, ViTPoseModel] = {}


def _get_vitpose_model(device: str = "cuda") -> ViTPoseModel:
    """Return a cached `ViTPoseModel` (no Detectron2 dependency)."""
    if "cpm" in _HAMER_CACHE:
        return _HAMER_CACHE["cpm"]

    import hamer.vitpose_model as _vpm  # local import avoids side‑effects if unused

    _hamer_root = Path(hamer.__file__).resolve().parent.parent
    _vpm.ROOT_DIR = str(_hamer_root)
    _vpm.VIT_DIR = str(_hamer_root / "third-party" / "ViTPose")

    cfg_rel = Path(
        "configs/wholebody/2d_kpt_sview_rgb_img/topdown_heatmap/coco-wholebody/"
        "ViTPose_huge_wholebody_256x192.py"
    )
    ckpt_rel = Path("_DATA/vitpose_ckpts/vitpose+_huge/wholebody.pth")

    for _name, _dic in _vpm.ViTPoseModel.MODEL_DICT.items():
        _dic["config"] = str(Path(_vpm.VIT_DIR) / cfg_rel)
        _dic["model"] = str(Path(_vpm.ROOT_DIR) / ckpt_rel)

    _HAMER_CACHE["cpm"] = ViTPoseModel(device)
    return _HAMER_CACHE["cpm"]


# ---------------------------------------------------------------------------
# Wrist detection helper (depth‑guided + ViTPose)
# ---------------------------------------------------------------------------


def _wrist_from_frame(frame_bgr: np.ndarray, gray_depth: np.ndarray, cpm: ViTPoseModel):
    # 1) Depth‑based hand ROI – nearest object in view
    nearest = gray_depth < np.percentile(gray_depth, 7)  # closest 7 %
    labels, n_lbl = ndimage.label(nearest)
    if n_lbl == 0:
        return None

    # largest blob → hand / forearm
    sizes = ndimage.sum(nearest, labels, range(1, n_lbl + 1))
    hand_lbl = 1 + int(np.argmax(sizes))
    mask = labels == hand_lbl
    ys, xs = np.where(mask)
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()

    roi_bgr = frame_bgr[y0 : y1 + 1, x0 : x1 + 1]

    # 2) ViTPose inside ROI
    bbox = np.array(
        [[0, 0, roi_bgr.shape[1] - 1, roi_bgr.shape[0] - 1, 1.0]], dtype=np.float32
    )
    pose = cpm.predict_pose(roi_bgr[:, :, ::-1], [bbox])[0]

    hand_kpts = pose["keypoints"][-21:]  # right‑hand keypoints
    valid = hand_kpts[:, 2] > 0.35
    if valid.sum() <= 3:
        return None

    wrist_u = x0 + hand_kpts[0, 0]
    wrist_v = y0 + hand_kpts[0, 1]
    return float(wrist_u), float(wrist_v)


# ---------------------------------------------------------------------------
# Contact / separation detection from speed curve
# ---------------------------------------------------------------------------


def contact_separation_from_speed(
    speed_dict: Dict[int, float], *, min_speed_ratio=0.2, sigma=3
):
    """Return (first_contact_idx, last_separation_idx) based on smoothed speed."""
    speeds = gaussian_filter1d(np.array(list(speed_dict.values())), sigma)
    baseline = np.median(speeds)
    peak = speeds.max()
    thresh = baseline + (peak - baseline) * min_speed_ratio

    active = speeds > thresh
    if not active.any():
        return -1, -1

    contact = int(np.argmax(active))
    separation = int(len(active) - np.argmax(active[::-1]))
    return contact, separation


def refine_with_vlm(current_idx: int, creds: Dict[str, Any], video_path: str, prompt: str) -> int:
    """Ask GPT‑4(o) to confirm or shift the key frame.  Fall back to `current_idx`."""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, current_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return current_idx

    new_idx = scene_understanding(creds, frame, prompt)
    return current_idx if new_idx in (-1, None) else int(new_idx)


# ---------------------------------------------------------------------------
# 3‑D hand‑speed extraction + visualisation
# ---------------------------------------------------------------------------


def extract_3d_speed_and_visualize(video_path: str, output_dir: str, *, device: str = "cuda", encoder: str = "vits"):
    cpm = _get_vitpose_model(device)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    video_name = Path(video_path).stem

    depth_dir = output_dir / "depth"
    depth_vis_path = depth_dir / "depth_vis.mp4"

    if (
        not depth_vis_path.exists()
        or not (depth_dir / "pred_depth_000000.npy").exists()
    ):
        print("[3D‑Pipeline] Generating depth (tensors + video) …")
        depth_dir.mkdir(parents=True, exist_ok=True)
        generate_depth_video_vda(video_path, depth_dir, device=device, encoder=encoder)
    else:
        print("[3D‑Pipeline] Reusing cached depth outputs in", depth_dir)

    cap_rgb = cv2.VideoCapture(video_path)
    if not cap_rgb.isOpened():
        raise RuntimeError("Could not open RGB video.")

    total_frames = int(cap_rgb.get(cv2.CAP_PROP_FRAME_COUNT))
    speed_dict: Dict[int, float] = {}
    prev_xyz = None

    for idx in range(total_frames):
        ok_rgb, frame_bgr = cap_rgb.read()
        if not ok_rgb:
            speed_dict[idx] = 0.0
            prev_xyz = None
            continue

        gray_depth = _load_depth(depth_dir, idx)
        if gray_depth is None:
            speed_dict[idx] = 0.0
            prev_xyz = None
            continue

        H, W = gray_depth.shape
        wrist = _wrist_from_frame(frame_bgr, gray_depth, cpm)
        if wrist is None:
            speed_dict[idx] = 0.0
            prev_xyz = None
            continue

        u, v = wrist
        z = float(gray_depth[min(int(v), H - 1), min(int(u), W - 1)])
        X, Y, Z = _pixel_to_camera(u, v, z, W, H)

        if prev_xyz is None:
            speed = 0.0
        else:
            dX, dY, dZ = X - prev_xyz[0], Y - prev_xyz[1], Z - prev_xyz[2]
            speed = math.sqrt(dX * dX + dY * dY + dZ * dZ)
        prev_xyz = (X, Y, Z)
        speed_dict[idx] = speed

        if (idx + 1) % 100 == 0 or idx == total_frames - 1:
            print(f"[3D‑Pipeline] Processed {idx + 1}/{total_frames} frames")

    cap_rgb.release()

    speed_json_path = output_dir / f"{video_name}_speed.json"
    with open(speed_json_path, "w") as f:
        json.dump(speed_dict, f, indent=2)

    plt.figure(figsize=(12, 4))
    plt.plot(list(speed_dict.keys()), list(speed_dict.values()), label="3‑D Hand Speed")
    plt.xlabel("Frame")
    plt.ylabel("Speed (relative)")
    plt.tight_layout()
    speed_vis_path = output_dir / f"{video_name}_speed_vis.png"
    plt.savefig(speed_vis_path)
    plt.close()

    return speed_dict, str(speed_json_path), str(speed_vis_path), str(depth_vis_path)


# ---------------------------------------------------------------------------
# Video Convert
# ---------------------------------------------------------------------------

def convert_video(video_path: str, action: str, credentials: Dict[str, Any], grid_size: int, speed_folder: str, max_feedbacks: int = MAX_FEEDBACKS, repeat_times: int = 3):
    """Driver wrapper (identical logic to 2-D)."""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    contact_list, separate_list = [], []
    for _ in range(repeat_times):
        frame_start = process_task(
            credentials,
            video_path,
            action,
            grid_size,
            total_frames,
            "start",
            speed_folder,
        )
        frame_end = process_task(
            credentials,
            video_path,
            action,
            grid_size,
            total_frames,
            "end",
            speed_folder,
        )
        frame_contact = feedback_contact(
            credentials,
            video_path,
            action,
            grid_size,
            total_frames,
            frame_start,
            max_feedbacks,
            "start",
            speed_folder,
        )
        frame_separate = feedback_separation(
            credentials,
            video_path,
            action,
            grid_size,
            total_frames,
            frame_end,
            max_feedbacks,
            "end",
            speed_folder,
        )
        contact_list.append(frame_contact)
        separate_list.append(frame_separate)

    contact_list = [x for x in contact_list if x not in (None, -1)]
    separate_list = [x for x in separate_list if x not in (None, -1)]
    final_contact = -1 if not contact_list else int(round(np.mean(contact_list)))
    final_separate = -1 if not separate_list else int(round(np.mean(separate_list)))
    return final_contact, final_separate


# ---------------------------------------------------------------------------
# Tiny visual helper
# ---------------------------------------------------------------------------

def visualize_frame(video_path: str, idx: int, out_path: str, label: Optional[str] = None):
    if idx < 0:
        return
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        print(f"[WARN] Could not seek to frame {idx} in {video_path}")
        return
    if label:
        cv2.putText(frame, label, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
    cv2.imwrite(out_path, frame)
    print(f"Visualized frame {idx} to {out_path}")


# ---------------------------------------------------------------------------
# CLI entry point (with feedback loop)
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser("EgoLoc 3-D Demo (lite edition)")
    parser.add_argument("--video_path", required=True, help="Input video")
    parser.add_argument(
        "--action", default="Grasping the object", help="Action label shown to GPT-4o"
    )
    parser.add_argument(
        "--grid_size", type=int, default=3, help="Grid size for numbered frame grids"
    )
    parser.add_argument("--output_dir", default="output", help="Output folder")
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu", "auto"],
        help="Computation device",
    )
    parser.add_argument(
        "--credentials", required=True, help="Path to .env with OpenAI / Azure keys"
    )
    parser.add_argument(
        "--encoder",
        default="vits",
        choices=["vits", "vitl"],
        help="Video-Depth-Anything backbone: 'vits' (small) or 'vitl' (large)",
    )
    args = parser.parse_args()

    # ------------------- device selection -------------------
    if args.device == "auto":
        device = "cuda" if torch and torch.cuda.is_available() else "cpu"
    elif args.device == "cuda" and (not torch or not torch.cuda.is_available()):
        print("[WARN] CUDA requested but unavailable – falling back to CPU")
        device = "cpu"
    else:
        device = args.device

    # ------------------- step 1: 3-D speed -------------------
    print("\n [1/3] Extracting 3-D hand speed and visualizing ...")
    speed_dict, speed_json, speed_vis, depth_vid = extract_3d_speed_and_visualize(
        args.video_path,
        args.output_dir,
        device=device,
        encoder=args.encoder,
    )

    # ------------- step 2: temporal localisation -------------
    print("\n [2/3] Locating contact/separation frames and visualizing ...")
    credentials = dotenv.dotenv_values(args.credentials)

    contact_idx, separation_idx = convert_video(
        args.video_path,
        args.action,
        credentials,
        args.grid_size,
        args.output_dir,  # *_speed.json lives here
        max_feedbacks=1,  # GPT-4o stops early once verified
        repeat_times=1,
    )
    print(f"Resolved ➜ contact: {contact_idx}, separation: {separation_idx}")

    # ------------- visualise keyframes -----------------------
    video_name = Path(args.video_path).stem
    contact_vis = Path(args.output_dir) / f"{video_name}_contact_frame.png"
    separation_vis = Path(args.output_dir) / f"{video_name}_separation_frame.png"
    visualize_frame(args.video_path, contact_idx, str(contact_vis), "Contact")
    visualize_frame(args.video_path, separation_idx, str(separation_vis), "Separation")

    # ---------------- step 3: save results -------------------
    print("\n [3/3] Saving results ...")
    result = {
        "contact_frame": contact_idx,
        "separation_frame": separation_idx,
    }
    result_path = Path(args.output_dir) / f"{video_name}_result.json"
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)

    # ------------------- final console log -------------------
    print("EgoLoc output\n", result)
    print(f"Result json      : {result_path}")
    print(f"3-D speed json   : {speed_json}")
    print(f"3-D speed vis    : {speed_vis}")
    print(f"Depth video      : {depth_vid}")
    print(f"Contact frame vis: {contact_vis}")
    print(f"Separation vis   : {separation_vis}")


if __name__ == "__main__":
    main()
