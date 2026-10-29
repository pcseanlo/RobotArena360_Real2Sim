"""Compare a new pose fit with the old repo's ``aligned_pose.npy`` (GPU).

Renders both poses with the new gsplat renderer, reports the fit loss of each
(lower is better), the translation / yaw difference, and writes a side-by-side
review image next to the new result.

    python tools/legacy/align_pose_check.py --scene droid2
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from r2s.layout import SceneLayout  # noqa: E402
from old_repo import OLD  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True)
    parser.add_argument("--old-pose", type=Path, default=None)
    args = parser.parse_args()

    import torch
    import torch.nn.functional as F
    from PIL import Image

    from r2s.align.pose_fit import DINOv2Extractor, cosine_loss, rgb_l2_loss
    from r2s.align.render import Camera, SplatModel, axis_quat_t
    from r2s.common.quat import quat_mul_np
    from r2s.common.transforms import SCALE_OPT_ADJUSTMENT, SCALE_OPT_BASE_ROT

    layout = SceneLayout.create(args.scene)
    old_path = args.old_pose or OLD / "optimization_results" / args.scene / "aligned_pose.npy"
    old = np.load(old_path).astype(np.float64).reshape(-1)
    new = np.load(layout.aligned_pose).astype(np.float64).reshape(-1)

    device = torch.device("cuda")
    cam = Camera.from_droid(layout.inputs.intrinsics, layout.inputs.extrinsics, device)
    splat = SplatModel(layout.gaussian_ply, quat_mul_np(SCALE_OPT_ADJUSTMENT, SCALE_OPT_BASE_ROT), device)
    real = torch.from_numpy(np.array(Image.open(layout.inputs.background_png).convert("RGB"))).to(device).float() / 255
    real = F.interpolate(real.permute(2, 0, 1)[None], size=(cam.height, cam.width), mode="bilinear",
                         align_corners=False)[0].permute(1, 2, 0)
    dino = DINOv2Extractor(device)
    real_dino = dino.extract(real)

    def render(pose):
        q = axis_quat_t(torch.tensor([pose[-1]], device=device, dtype=torch.float32), 2)
        return splat.render(cam, q, torch.tensor(pose[:3], device=device, dtype=torch.float32), bg=0.5)

    out = {"scene": args.scene, "old_pose": old.tolist(), "new_pose": new.tolist()}
    images = []
    with torch.no_grad():
        for name, pose in (("old", old), ("new", new)):
            rgb = render(pose)
            out[f"loss_{name}"] = float(10.0 * rgb_l2_loss(rgb, real) + cosine_loss(dino.extract(rgb), real_dino))
            images.append((rgb.clamp(0, 1) * 255).byte().cpu().numpy())
    out["dt_m"] = float(np.linalg.norm(new[:3] - old[:3]))
    out["dyaw_deg"] = float(math.degrees((new[-1] - old[-1] + math.pi) % (2 * math.pi) - math.pi))
    tgt = (real.clamp(0, 1) * 255).byte().cpu().numpy()
    Image.fromarray(np.concatenate([tgt, images[0], images[1]], axis=1)).save(layout.align_dir / "check_target_old_new.png")
    print(json.dumps(out, indent=2))
    (layout.align_dir / "check_vs_old.json").write_text(json.dumps(out, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
