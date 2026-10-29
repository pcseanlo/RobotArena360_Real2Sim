"""Worker for ``inputs.meshes`` (env ``sam3d-objects``): one SAM 3D Objects mesh per object mask.

Port of scene_generation ``src/pipeline/sam3d.py`` (``main`` without the
commented-out Gemini/CoACD parts). Runs ``Inference(pipeline.yaml)(image, mask, seed)``
for every mask (mesh post-processing + texture baking, as configured inside
``Inference.__call__``), then writes per object ``<name>``
(``image_<k>_mask`` -> ``image_<k>_segmented``):

    <name>_mask.png     the mask actually used (blue channel > 0 of the RGB-masked image)
    <name>.ply          raw SAM 3D mesh with vertex colours
    <name>.glb          textured GLB from the pipeline
    <name>_scaled.glb   colourless copy in Hunyuan3D view space (Y up, centred, max extent 1.8)

The orchestrating step chooses the masks and writes ``asset_info.json``.
Imports only what the env provides plus ``<sam3d_objects>/notebook/inference.py``.
"""
import argparse
import os
import random
import sys
from pathlib import Path

# The r2s repo env exports these for its own gsplat build; the sam3d-objects env
# was always run without them (prebuilt nvdiffrast), so restore that.
for _var in ("TORCH_EXTENSIONS_DIR", "TORCH_CUDA_ARCH_LIST"):
    os.environ.pop(_var, None)


def convert_sam_to_hunyuan_space(mesh, rotation_matrix=None):
    """
    Transforms a SAM-3D canonical mesh into Hunyuan3D view-space
    and STRIPS vertex colors by creating a fresh mesh.
    """
    import numpy as np
    import trimesh

    # ---------------------------------------------------------
    # STEP 0: Force Strip Colors (The "Nuclear" Option)
    # ---------------------------------------------------------
    # We explicitly instantiate a new mesh with ONLY verts and faces.
    # We do NOT pass 'visuals' or 'process=True'.
    vertices=mesh.vertices.copy()
    # Default Z-up to Y-up fix
    vertices = vertices @ np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    out_mesh = trimesh.Trimesh(vertices=vertices, faces=mesh.faces.copy(), process=False)

    # Double check: explicitly set visual to None just in case
    out_mesh.visual = trimesh.visual.ColorVisuals(out_mesh) # Resets to empty default if mesh has no colors
    out_mesh.visual = None # Or force None

    # ---------------------------------------------------------
    # STEP 2: Center and Normalize Scale
    # ---------------------------------------------------------
    out_mesh.apply_translation(-out_mesh.bounds.mean(axis=0))

    max_extent = np.max(out_mesh.extents)
    if max_extent > 0:
        scale_factor = 1.8 / max_extent
        out_mesh.apply_scale(scale_factor)

    return out_mesh


def load_mask(path, load_image):
    print(path)
    mask = load_image(path)
    mask = mask > 0
    if mask.ndim == 3:
        mask = mask[..., -1]
    mask = mask > 0
    return mask


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sam3d-root", required=True, help="SAM 3D Objects checkout (has notebook/inference.py)")
    ap.add_argument("--config", required=True, help="checkpoints/hf/pipeline.yaml")
    ap.add_argument("--image", required=True, help="DROID/<s>/images0/1.jpg")
    ap.add_argument("--masks", nargs="+", required=True, help="masks/image_<k>_mask.png, in output order")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--seed", type=int, default=42, help="seed passed to every Inference call (old: 42)")
    ap.add_argument("--resize-h", type=int, default=-1)
    ap.add_argument("--resize-w", type=int, default=-1)
    args = ap.parse_args()

    # Old script: sys.path.insert(0, "<sam3d>/notebook"); from inference import Inference, load_image
    sys.path.insert(0, str(Path(args.sam3d_root) / "notebook"))
    import cv2
    import numpy as np
    import trimesh
    from inference import Inference, load_image

    # The pipeline seeds torch per call; numpy/random were unseeded in the old run.
    random.seed(args.seed)
    np.random.seed(args.seed)

    image = load_image(args.image)
    masks = [load_mask(p, load_image) for p in args.masks]
    names = [Path(p).stem for p in args.masks]
    inference = Inference(args.config, compile=False)
    outputs = [inference(image, mask, seed=args.seed) for mask in masks]

    meshes = [output["mesh"][0] for output in outputs]
    scales = [output["scale"] for output in outputs]
    glbs = [output["glb"] for output in outputs]

    os.makedirs(args.output_dir, exist_ok=True)
    for i in range(len(outputs)):
        obj_mask = masks[i]
        mesh = meshes[i]
        scale = scales[i]
        object_name = names[i].replace("mask", "segmented")
        object_save_dir = args.output_dir

        # resize mask and save
        if args.resize_h > 0 and args.resize_w > 0:
            resized_obj_mask = cv2.resize(obj_mask.astype(np.uint8), (args.resize_w, args.resize_h), interpolation=cv2.INTER_NEAREST)
            resized_obj_mask = (resized_obj_mask > 0).astype(np.uint8) * 255
        else:
            resized_obj_mask = obj_mask.astype(np.uint8) * 255
        print(resized_obj_mask.shape)
        resized_obj_mask_path = os.path.join(object_save_dir, f"{object_name}_mask.png")
        cv2.imwrite(resized_obj_mask_path, resized_obj_mask)

        # save asset mesh
        mesh_trimesh = trimesh.Trimesh(
            vertices=mesh.vertices.cpu().numpy(),
            faces=mesh.faces.cpu().numpy(),
            vertex_colors=(mesh.vertex_attrs.cpu().numpy()[..., :3] * 255).astype(np.uint8)
        )
        mesh_filename = os.path.join(object_save_dir, f"{object_name}.ply")
        mesh_trimesh.export(mesh_filename)

        glbs[i].export(os.path.join(object_save_dir, f"{object_name}.glb"))

        rescaled_glb_path = os.path.join(object_save_dir, f"{object_name}_scaled.glb")
        rescaled_glb = convert_sam_to_hunyuan_space(mesh_trimesh)
        rescaled_glb.export(rescaled_glb_path)
        print(f"[sam3d] {object_name}: {len(mesh_trimesh.vertices)} verts, {len(mesh_trimesh.faces)} faces, "
              f"scale {scale.cpu().numpy().mean():.5f}")


if __name__ == "__main__":
    main()
