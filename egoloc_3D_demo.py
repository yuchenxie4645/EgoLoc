"""
EgoLoc 3‑D Demo – *lite, compact* edition
=======================================

• Stripped of unused dependencies / dead code.
• Adds a **simple feedback loop**: the Vision‑Language Model (VLM)
  can iteratively refine the contact / separation frames up to
  `--max_feedbacks` times until the suggestion stabilises.

Assumed repo layout (relative to this file):
EgoLoc/
│
├── ./                    ← this script lives here (anywhere inside EgoLoc)
├── HaMeR/                ← git clone https://github.com/geopavlakos/hamer.git
└── Video-Depth-Anything/ ← git clone https://github.com/DepthAnything/Video-Depth-Anything.git
"""

# ---------------------------------------------------------------------------
# Imports – keep it minimal
# ---------------------------------------------------------------------------
import argparse, base64, json, math, os, subprocess, time, warnings
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage                    # connected‑component helper
from scipy.ndimage import gaussian_filter1d

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
    from vitpose_model import ViTPoseModel  # type: ignore
except ImportError as e:
    raise ImportError(
        "vitpose_model.py not found.  It ships with HaMeR; "
        "confirm your PYTHONPATH contains that folder."
    ) from e

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
plt.switch_backend("Agg")                    # headless plotting

# ---------------------------------------------------------------------------
# Helper – unpack Video‑Depth‑Anything *_depths.npz → per‑frame .npy
# ---------------------------------------------------------------------------

def _unpack_depth_npz(depth_dir: Path) -> None:
    """Convert VDA's combined depth archive into individual per‑frame tensors."""
    npz_files = list(depth_dir.glob("*_depths.npz"))
    if not npz_files:
        return                              # nothing to unpack

    arr = np.load(npz_files[0])["depths"]  # (N, H, W)
    for i, depth in enumerate(arr):
        np.save(depth_dir / f"pred_depth_{i:06d}.npy", depth.astype(np.float32))
    print(f"[VDA] Unpacked {len(arr)} frames to .npy tensors")

# ---------------------------------------------------------------------------
# Paths & repo‑root helpers
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = next(
    p for p in [SCRIPT_DIR] + list(SCRIPT_DIR.parents) if (p / "Video-Depth-Anything").exists()
)

VDA_DIR = REPO_ROOT / "Video-Depth-Anything"
VDA_CHECKPOINT_PATH = Path(
    os.getenv("VDA_CHECKPOINT_PATH", VDA_DIR / "checkpoints" / "video_depth_anything_vits.pth")
).resolve()

if not VDA_CHECKPOINT_PATH.exists():
    raise FileNotFoundError(f"Video-Depth-Anything checkpoint not found at {VDA_CHECKPOINT_PATH}")

DEPTH_SCALE_M = 3.0  # pixel value 255 ↔ 3 m (linear scaling)

# ---------------------------------------------------------------------------
# Depth video generation via Video‑Depth‑Anything
# ---------------------------------------------------------------------------

def generate_depth_video_vda(video_path: str, depth_out_path: str, device: str = "cuda") -> Path:
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
        "vits",  # or "vitl" if you want the large model
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
        raise RuntimeError("[VDA] No pred_depth_*.npy tensors were produced – check the VDA output above.")

    return depth_out_path

# ---------------------------------------------------------------------------
# Depth loading helper
# ---------------------------------------------------------------------------

def _load_depth(depth_dir: Path, idx: int, scale_m: float = DEPTH_SCALE_M) -> np.ndarray | None:
    """Read VDA’s inverse‑depth tensor for frame *idx* and convert to metres."""
    f = depth_dir / f"pred_depth_{idx:06d}.npy"
    if not f.exists():
        return None
    inv = np.load(f)                       # (H, W) float32
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

_HAMER_CACHE: dict[str, ViTPoseModel] = {}


def _get_vitpose_model(device: str = "cuda") -> ViTPoseModel:
    """Return a cached `ViTPoseModel` (no Detectron2 dependency)."""
    if "cpm" in _HAMER_CACHE:
        return _HAMER_CACHE["cpm"]

    import vitpose_model as _vpm  # local import avoids side‑effects if unused

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
    nearest = gray_depth < np.percentile(gray_depth, 7)      # closest 7 %
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
    bbox = np.array([[0, 0, roi_bgr.shape[1] - 1, roi_bgr.shape[0] - 1, 1.0]], dtype=np.float32)
    pose = cpm.predict_pose(roi_bgr[:, :, ::-1], [bbox])[0]

    hand_kpts = pose["keypoints"][-21:]            # right‑hand keypoints
    valid = hand_kpts[:, 2] > 0.35
    if valid.sum() <= 3:
        return None

    wrist_u = x0 + hand_kpts[0, 0]
    wrist_v = y0 + hand_kpts[0, 1]
    return float(wrist_u), float(wrist_v)

# ---------------------------------------------------------------------------
# Contact / separation detection from speed curve
# ---------------------------------------------------------------------------

def contact_separation_from_speed(speed_dict: dict[int, float], *, min_speed_ratio=0.2, sigma=3):
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

# ---------------------------------------------------------------------------
# VLM helper – robust exception handling
# ---------------------------------------------------------------------------

def scene_understanding(credentials: dict, frame: np.ndarray, prompt: str, *, raw: bool = False):
    """Query GPT‑4(o) or Azure OpenAI with an image and prompt."""
    import openai  # imported lazily

    frame = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))
    _, buf = cv2.imencode(".jpg", frame)
    b64 = base64.b64encode(buf).decode("utf-8")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "high"},
                },
            ],
        }
    ]

    # Choose client
    if credentials.get("AZURE_OPENAI_API_KEY"):
        from openai import AzureOpenAI  # type: ignore

        client = AzureOpenAI(
            api_version="2024-02-01",
            azure_endpoint=credentials["AZURE_OPENAI_ENDPOINT"],
            api_key=credentials["AZURE_OPENAI_API_KEY"],
        )
        params = {"model": credentials["AZURE_OPENAI_DEPLOYMENT_NAME"]}
    else:
        from openai import OpenAI  # type: ignore

        client = OpenAI(
            api_key=credentials["OPENAI_API_KEY"],
            base_url="https://api.chatanywhere.tech/v1",
        )
        params = {"model": "gpt-4o"}

    params.update(
        {
            "messages": messages,
            "max_tokens": 200,
            "temperature": 0.1,
            "top_p": 0.5,
        }
    )

    # Retry (transient errors are common)
    for _ in range(5):
        try:
            result = client.chat.completions.create(**params)
            break
        except Exception as exc:  # noqa: BLE001
            print("[VLM] transient error:", exc)
            time.sleep(2)
    else:
        raise RuntimeError("Failed to get response from OpenAI after retries.")

    content = result.choices[0].message.content
    if raw:
        return content

    # Extract minimal JSON: {"points": [idx]}
    text = content.strip().replace(" ", "").replace("\n", "")
    try:
        start = text.index('{"points":')
        json_part = text[start : text.index("}") + 1]
        pts = json.loads(json_part)["points"]
        return pts[0] if pts else -1
    except ValueError:
        return -1
    except Exception:
        return -1


def refine_with_vlm(current_idx: int, creds: dict, video_path: str, prompt: str) -> int:
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

def extract_3d_speed_and_visualize(video_path: str, output_dir: str, device: str = "cuda"):
    cpm = _get_vitpose_model(device)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    video_name = Path(video_path).stem

    depth_dir = output_dir / "depth"
    depth_vis_path = depth_dir / "depth_vis.mp4"

    if not depth_vis_path.exists() or not (depth_dir / "pred_depth_000000.npy").exists():
        print("[3D‑Pipeline] Generating depth (tensors + video) …")
        depth_dir.mkdir(parents=True, exist_ok=True)
        generate_depth_video_vda(video_path, depth_dir, device)
    else:
        print("[3D‑Pipeline] Reusing cached depth outputs in", depth_dir)

    cap_rgb = cv2.VideoCapture(video_path)
    if not cap_rgb.isOpened():
        raise RuntimeError("Could not open RGB video.")

    total_frames = int(cap_rgb.get(cv2.CAP_PROP_FRAME_COUNT))
    speed_dict: dict[int, float] = {}
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
        z = float(gray_depth[int(round(v)) % H, int(round(u)) % W])
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
# Tiny visual helper
# ---------------------------------------------------------------------------

def visualize_frame(video_path: str, idx: int, out_path: str, label: str | None = None):
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

# ---------------------------------------------------------------------------
# CLI entry point (with feedback loop)
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser("EgoLoc 3‑D Demo (lite edition)")
    parser.add_argument("--video_path", required=True)
    parser.add_argument("--output_dir", default="output")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu", "auto"])
    parser.add_argument("--credentials", required=True, help="Path to .env with OpenAI/Azure keys")
    parser.add_argument("--max_feedbacks", type=int, default=1, help="Iterations of VLM refinement")
    args = parser.parse_args()

    # Pick device safely
    if args.device == "auto":
        device = "cuda" if torch and torch.cuda.is_available() else "cpu"
    elif args.device == "cuda" and (not torch or not torch.cuda.is_available()):
        print("[WARN] CUDA requested but unavailable – falling back to CPU")
        device = "cpu"
    else:
        device = args.device

    # 1. Speed extraction --------------------------------------------------
    print("[1/3] Extracting 3‑D hand speed …")
    speed_dict, speed_json, speed_vis, depth_vid = extract_3d_speed_and_visualize(
        args.video_path, args.output_dir, device
    )

    # 2. Localisation ------------------------------------------------------
    from dotenv import dotenv_values

    creds = dotenv_values(args.credentials)  # type: ignore
    print("[2/3] Localising contact / separation …")

    contact_idx, separation_idx = contact_separation_from_speed(speed_dict)
    print(f"[2/3] Initial estimate – contact: {contact_idx}, separation: {separation_idx}")

    contact_prompt = "Which frame best shows the hand FIRST TOUCHING the object?"
    separation_prompt = "Which frame best shows the hand COMPLETELY LEAVING the object?"

    # -------------------- Simple feedback loop ---------------------------
    for i in range(args.max_feedbacks):
        new_contact = refine_with_vlm(contact_idx, creds, args.video_path, contact_prompt)
        new_separation = refine_with_vlm(separation_idx, creds, args.video_path, separation_prompt)

        if (new_contact, new_separation) == (contact_idx, separation_idx):
            break  # converged
        contact_idx, separation_idx = new_contact, new_separation
        print(f"[2/3] Feedback pass {i+1}: contact → {contact_idx}, separation → {separation_idx}")

    # 3. Visualise keyframes ---------------------------------------------
    video_name = Path(args.video_path).stem
    contact_vis = Path(args.output_dir) / f"{video_name}_contact_frame.png"
    separation_vis = Path(args.output_dir) / f"{video_name}_separation_frame.png"
    visualize_frame(args.video_path, contact_idx, str(contact_vis), "Contact")
    visualize_frame(args.video_path, separation_idx, str(separation_vis), "Separation")

    result = {"contact_frame": contact_idx, "separation_frame": separation_idx}
    result_path = Path(args.output_dir) / f"{video_name}_result.json"
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)

    print(
        "[3/3] Done. Outputs:\n",
        "- 3D speed       :", speed_json, "\n",
        "- 3D speed plot  :", speed_vis, "\n",
        "- Depth video    :", depth_vid, "\n",
        "- Contact frame  :", contact_vis, "\n",
        "- Separation frame:", separation_vis, "\n",
        "- Result JSON    :", result_path,
    )


if __name__ == "__main__":
    main()
