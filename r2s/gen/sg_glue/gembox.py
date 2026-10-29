"""Gemini-box + SAM3 segmentation for step 0c-1, from SG ``src/pipeline/segment_object_gembox_updated.py``.

Moved verbatim: the three prompts, ``_draw_boxes_debug``, ``_extract_json_array``,
``_pad_to_square``, ``_denorm_box``, ``_expand_xyxy_box``, ``_prefixed_label``,
``_box_mask_from_xyxy``, ``_inpaint_mask_or_box_fallback``, ``_sam3_masks_batched``.

Changes:
- ``_gemini_boxes`` takes the API key and model name as arguments (SG read the key
  from ``configs/config_DROID.yaml``; r2s passes ``$GEMINI_API_KEY``) and also
  returns the raw response text so it can be saved for replay.
- ``_load_sam3`` takes the model id and device as arguments (SG: ``$SAM3_MODEL_ID``
  or ``facebook/sam3``; ``cuda`` when available).
- SG's ``run()`` interleaved the Gemini calls with the SAM3 calls. Here it is split
  into :func:`gemini_prompts` (the three Gemini queries -> box prompts) and
  :func:`segment_from_prompts` (SAM3 + persistence), so the SAM3 half can be re-run
  from saved Gemini answers. SAM3 sees the same batches (robot, human, objects) in
  the same order, and the persisted masks are built in the same order.
"""

import json
import os
import re

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .gemini_utils import SegmentationMask, plot_segmentation_masks

_DEBUG_COLORS = [
    "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
    "#911eb4", "#46f0f0", "#f032e6", "#bcf60c", "#fabebe",
    "#008080", "#e6beff", "#9a6324", "#800000", "#aaffc3",
]


def _draw_boxes_debug(image, prompts, out_path):
    """Overlay (label, xyxy) boxes on the image and save to out_path."""
    if not prompts:
        return
    canvas = image.copy().convert("RGB")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf", 18,
        )
    except (OSError, IOError):
        font = ImageFont.load_default()

    for i, (label, (x0, y0, x1, y1)) in enumerate(prompts):
        color = _DEBUG_COLORS[i % len(_DEBUG_COLORS)]
        draw.rectangle([x0, y0, x1, y1], outline=color, width=4)
        text = f"{i}: {label}"
        try:
            tx0, ty0, tx1, ty1 = draw.textbbox((x0, max(0, y0 - 22)), text, font=font)
            draw.rectangle([tx0 - 2, ty0 - 2, tx1 + 2, ty1 + 2], fill=color)
        except AttributeError:
            pass  # older PIL without textbbox; skip background
        draw.text((x0, max(0, y0 - 22)), text, fill="white", font=font)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    canvas.save(out_path)
    print(f"[gembox] wrote debug overlay -> {out_path}")


def _pick_image(video_dir):
    super_res = os.path.join(video_dir, "super.png")
    if os.path.exists(super_res):
        return super_res
    return os.path.join(video_dir, "1.jpg")


# Modified version of the original `gemini_mask` prompt -- keeps everything the
# downstream pipeline cares about (box + label + physics) and explicitly tells
# Gemini not to bother with mask / polygon since SAM3 will produce the mask.
OBJECT_PROMPT = """\
Give bounding boxes for all important complete foreground objects on the plane
which the robot interacts with. Ignore background objects, irrelevant surfaces,
the robot itself, and anything occluded or severely blocked.
Return a JSON list ONLY (no commentary). Each entry must contain:
- 'box_2d': [ymin, xmin, ymax, xmax], each value normalized to 0-1000.
- 'mass': estimated object mass in kilograms (float).
- 'friction': estimated friction coefficient (float).
- 'surface_type': one of [Glass, Water, Emission, Plastic, Rough, Smooth,
  Reflective, Metal, Iron, Aluminium, Copper, Gold].
- 'label': a short descriptive text label.
Do NOT include 'mask' or 'polygon' fields.
"""

ROBOT_PROMPT = """\
Give bounding boxes for every visible part of the robot in this scene, if any.
Include the full visible robot body: base, pedestal, mount, support column,
links, arm, wrist, gripper, fingers, end-effector, and any robot-attached
cables or hardware. Do not stop at only the arm or gripper.
Return a JSON list ONLY (no commentary). Each entry must contain:
- 'box_2d': [ymin, xmin, ymax, xmax], normalized 0-1000.
- 'label': 'robot', 'robot base', 'robot arm', 'gripper', or a short robot-part label.
Do NOT include 'mask' or 'polygon'.
"""

HUMAN_PROMPT = """\
Give bounding boxes for every visible human or human body part in this scene,
if any. Include the full visible person: head, hair, face, torso, shoulders,
arms, elbows, hands, fingers, waist, legs, knees, feet, and shoes. For seated
or partially occluded people, include all visible lower-body parts too.
Return a JSON list ONLY (no commentary). Each entry must contain:
- 'box_2d': [ymin, xmin, ymax, xmax], normalized 0-1000.
- 'label': 'human', 'person', 'hand', 'arm', 'leg', 'foot', or a short human-part label.
Do NOT include 'mask' or 'polygon'.
"""


def _extract_json_array(text):
    if not text:
        return []
    m = re.search(r"\[.*\]", text, flags=re.S)
    if not m:
        return []
    raw = m.group(0)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        cleaned = re.sub(r"```(?:json)?", "", raw).strip("` \n")
        return json.loads(cleaned)


def _pad_to_square(img):
    """Center-pad an image to a square (black fill).
    Returns (square_image, side, pad_x, pad_y) so callers can map coordinates
    on the square back to the original.
    """
    w, h = img.size
    side = max(w, h)
    pad_x = (side - w) // 2
    pad_y = (side - h) // 2
    canvas = Image.new("RGB", (side, side), (0, 0, 0))
    canvas.paste(img, (pad_x, pad_y))
    return canvas, side, pad_x, pad_y


def _gemini_boxes(image_path, api_key, model_name, prompt, pad_square=True):
    """Query Gemini for boxes. Returns (items, side, pad_x, pad_y, raw_text).

    When `pad_square=True`, the image is center-padded to a square before being
    sent, so Gemini's letterboxing for non-square inputs doesn't skew coords.
    Returned `side`/`pad_x`/`pad_y` describe the canvas Gemini saw.
    """
    import google.generativeai as genai

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name=model_name)

    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    if pad_square and w != h:
        sent, side, pad_x, pad_y = _pad_to_square(img)
    else:
        sent, side, pad_x, pad_y = img, max(w, h), 0, 0

    resp = model.generate_content([prompt, sent])
    text = resp.text or ""
    return _extract_json_array(text), side, pad_x, pad_y, text


def _denorm_box(box, side, pad_x, pad_y, w, h):
    """Denormalize a Gemini [ymin, xmin, ymax, xmax] (0-1000) against the
    square canvas it saw, then translate back to original image coords."""
    y0 = int(np.clip(box[0] / 1000.0 * side - pad_y, 0, h))
    x0 = int(np.clip(box[1] / 1000.0 * side - pad_x, 0, w))
    y1 = int(np.clip(box[2] / 1000.0 * side - pad_y, 0, h))
    x1 = int(np.clip(box[3] / 1000.0 * side - pad_x, 0, w))
    return y0, x0, y1, x1


def _expand_xyxy_box(box, w, h, frac=0.08):
    x0, y0, x1, y1 = box
    pad_x = int((x1 - x0) * frac)
    pad_y = int((y1 - y0) * frac)
    return (
        max(0, x0 - pad_x),
        max(0, y0 - pad_y),
        min(w, x1 + pad_x),
        min(h, y1 + pad_y),
    )


def _prefixed_label(label, prefix):
    label = label.strip() or prefix
    if prefix in label.lower().split():
        return label
    return f"{prefix} {label}"


def _box_mask_from_xyxy(box, h, w):
    x0, y0, x1, y1 = box
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[y0:y1, x0:x1] = 255
    return mask, (y0, x0, y1, x1)


def _inpaint_mask_or_box_fallback(label, box, res, h, w, min_mask_box_coverage=0.03, max_box_area_ratio=0.30):
    x0, y0, x1, y1 = box
    box_area = max(1, (x1 - x0) * (y1 - y0))
    image_area = max(1, h * w)

    use_fallback = res is None
    if res is not None:
        mask_u8, _ = res
        coverage = float(np.count_nonzero(mask_u8[y0:y1, x0:x1])) / box_area
        use_fallback = coverage < min_mask_box_coverage

    if not use_fallback:
        return res

    if box_area / image_area > max_box_area_ratio:
        print(
            f"[gembox] skipping bbox fallback for {label!r}: "
            f"box covers {box_area / image_area:.2%} of image"
        )
        return res

    print(f"[gembox] using bbox fallback mask for inpaint-only {label!r}")
    return _box_mask_from_xyxy(box, h, w)


def _load_sam3(model_id="facebook/sam3", device=None):
    try:
        from transformers import Sam3Model, Sam3Processor
    except ImportError as e:
        raise ImportError(
            "SAM3 requires transformers with Sam3 support (the `gemini` env has it)."
        ) from e
    import torch

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = Sam3Processor.from_pretrained(model_id)
    model = Sam3Model.from_pretrained(model_id).to(device).eval()
    return processor, model, device, torch


def _sam3_masks_batched(processor, model, device, torch, image, prompts, score_threshold=0.3, mask_threshold=0.5):
    """Run SAM3 once on a batch of (label, bbox) prompts over the same image.

    Replicates the image once per prompt -- each batch entry pairs a single text
    label with a single bbox, matching the recipe in the SAM3 docs where each
    image carries its own prompt(s).
    Returns one (mask_u8, (y0,x0,y1,x1)) or None per prompt in input order.
    """
    n = len(prompts)
    if n == 0:
        return []
    images = [image] * n
    texts = [label for label, _ in prompts]
    input_boxes = [[[float(b[0]), float(b[1]), float(b[2]), float(b[3])]]
                   for _, b in prompts]
    input_boxes_labels = [[1]] * n  # one positive box per image

    inputs = processor(
        images=images,
        text=texts,
        input_boxes=input_boxes,
        input_boxes_labels=input_boxes_labels,
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    original_sizes = inputs.get("original_sizes")
    target_sizes = (original_sizes.tolist()
                    if original_sizes is not None else [(image.height, image.width)] * n)

    results_list = processor.post_process_instance_segmentation(
        outputs,
        threshold=score_threshold,
        mask_threshold=mask_threshold,
        target_sizes=target_sizes,
    )

    out = []
    for (label, (x0, y0, x1, y1)), results in zip(prompts, results_list):
        masks = results.get("masks")
        scores = results.get("scores")
        if masks is None or len(masks) == 0:
            out.append(None)
            continue

        masks_t = masks.detach().cpu() if hasattr(masks, "detach") else masks
        masks_np = np.asarray(masks_t)
        if masks_np.ndim == 4:
            masks_np = masks_np[:, 0]

        if scores is not None and len(scores) > 0:
            scores_np = np.asarray(
                scores.detach().cpu() if hasattr(scores, "detach") else scores
            )
            idx = int(scores_np.argmax())
        else:
            idx = int(np.array([m.sum() for m in masks_np]).argmax())

        mask = (masks_np[idx] > 0).astype(np.uint8) * 255
        keep = np.zeros_like(mask)
        keep[y0:y1, x0:x1] = mask[y0:y1, x0:x1]
        if keep.sum() == 0:
            out.append(None)
            continue
        ys, xs = np.where(keep > 0)
        out.append((keep, (int(ys.min()), int(xs.min()),
                           int(ys.max()) + 1, int(xs.max()) + 1)))
    return out


# --------------------------------------------------------------------------- #
# SG run(), split in two halves (see module docstring).
# --------------------------------------------------------------------------- #

def gemini_prompts(image_path, api_key, model_name):
    """Phase 1 of SG ``run()``: the three Gemini queries -> box prompts.

    Returns ``(prompts, answers)``. ``prompts`` is what :func:`segment_from_prompts`
    consumes; ``answers`` holds the raw Gemini replies (saved for replay).
    """
    pil_img = Image.open(image_path).convert("RGB")
    h, w = pil_img.height, pil_img.width
    answers = {"model_name": model_name, "image_size_wh": [w, h]}

    # ---- robot / gripper first ----
    robot_items, r_side, r_pad_x, r_pad_y, r_text = _gemini_boxes(image_path, api_key, model_name, ROBOT_PROMPT)
    answers["robot"] = {"canvas_side": r_side, "canvas_pad_xy": [r_pad_x, r_pad_y], "items": robot_items, "text": r_text}
    # ---- humans / visible body parts (inpaint-only, not reconstruction) ----
    human_items, h_side, h_pad_x, h_pad_y, h_text = _gemini_boxes(image_path, api_key, model_name, HUMAN_PROMPT)
    answers["human"] = {"canvas_side": h_side, "canvas_pad_xy": [h_pad_x, h_pad_y], "items": human_items, "text": h_text}
    # ---- foreground objects ----
    obj_items, o_side, o_pad_x, o_pad_y, o_text = _gemini_boxes(image_path, api_key, model_name, OBJECT_PROMPT)
    answers["objects"] = {"canvas_side": o_side, "canvas_pad_xy": [o_pad_x, o_pad_y], "items": obj_items, "text": o_text}
    return prompts_from_answers(answers), answers


def prompts_from_answers(answers):
    """Turn raw Gemini answers into SAM3 box prompts, exactly as SG ``run()`` did."""
    w, h = answers["image_size_wh"]

    rob = answers["robot"]
    r_side, (r_pad_x, r_pad_y) = rob["canvas_side"], rob["canvas_pad_xy"]
    print(f"[gembox] robot canvas: side={r_side} pad=({r_pad_x},{r_pad_y})")
    robot_prompts = []
    for item in rob["items"]:
        if "box_2d" not in item:
            continue
        print(f"[gembox] robot raw box_2d={item['box_2d']} label={item.get('label')}")
        y0, x0, y1, x1 = _denorm_box(item["box_2d"], r_side, r_pad_x, r_pad_y, w, h)
        if y1 <= y0 or x1 <= x0:
            continue
        label = _prefixed_label(str(item.get("label", "robot")), "robot")
        box = _expand_xyxy_box((x0, y0, x1, y1), w, h, frac=0.10)
        robot_prompts.append((label, box))

    hum = answers["human"]
    h_side, (h_pad_x, h_pad_y) = hum["canvas_side"], hum["canvas_pad_xy"]
    print(f"[gembox] human canvas: side={h_side} pad=({h_pad_x},{h_pad_y})")
    human_prompts = []
    for item in hum["items"]:
        if "box_2d" not in item:
            continue
        print(f"[gembox] human raw box_2d={item['box_2d']} label={item.get('label')}")
        y0, x0, y1, x1 = _denorm_box(item["box_2d"], h_side, h_pad_x, h_pad_y, w, h)
        if y1 <= y0 or x1 <= x0:
            continue
        label = _prefixed_label(str(item.get("label", "human")), "human")
        box = _expand_xyxy_box((x0, y0, x1, y1), w, h, frac=0.12)
        human_prompts.append((label, box))

    obj = answers["objects"]
    o_side, (o_pad_x, o_pad_y) = obj["canvas_side"], obj["canvas_pad_xy"]
    print(f"[gembox] obj canvas: side={o_side} pad=({o_pad_x},{o_pad_y}); orig (w,h)=({w},{h})")
    print(f"[gembox] gemini returned {len(obj['items'])} object candidates")
    kept_items = []
    obj_prompts = []
    raw_dump = []
    for item in obj["items"]:
        if "box_2d" not in item:
            print(f"[gembox] skipping (missing box_2d): {item}")
            continue
        raw_dump.append({"label": item.get("label"), "box_2d_raw": item["box_2d"]})
        print(f"[gembox] raw box_2d={item['box_2d']} label={item.get('label')}")
        y0, x0, y1, x1 = _denorm_box(item["box_2d"], o_side, o_pad_x, o_pad_y, w, h)
        if y1 <= y0 or x1 <= x0:
            print(f"[gembox] skipping invalid box: {item['box_2d']}")
            continue
        label = str(item.get("label", "object"))
        obj_prompts.append((label, (x0, y0, x1, y1)))
        kept_items.append(item)

    return {
        "image_size_wh": [w, h],
        "canvas_side": o_side,
        "canvas_pad_xy": [o_pad_x, o_pad_y],
        "raw_items": raw_dump,
        "robot": robot_prompts,
        "human": human_prompts,
        "objects": obj_prompts,
        "object_items": kept_items,
    }


def gemini_boxes_raw(prompts):
    """Content of ``masks/gemini_boxes_raw.json`` (SG schema)."""
    return {
        "image_size_wh": list(prompts["image_size_wh"]),
        "canvas_side": prompts["canvas_side"],
        "canvas_pad_xy": list(prompts["canvas_pad_xy"]),
        "items": prompts["raw_items"],
        "denormalized_xyxy": [{"label": lbl, "xyxy": list(box)} for lbl, box in prompts["objects"]],
        "inpaint_only_denormalized_xyxy": [
            {"label": lbl, "xyxy": list(box)} for lbl, box in prompts["robot"] + prompts["human"]
        ],
    }


def segment_from_prompts(image_path, prompts, mask_dir, save_dict, input_mask_dir, render_mask_dir,
                         sam3_model_id="facebook/sam3", device=None):
    """Phase 2 of SG ``run()``: debug overlay, SAM3 per batch, persistence.

    Returns a small summary dict (labels, which prompts SAM3 dropped, ...).
    """
    pil_img = Image.open(image_path).convert("RGB")
    h, w = pil_img.height, pil_img.width
    if [w, h] != list(prompts["image_size_wh"]):
        raise ValueError(f"anchor is {w}x{h} but the Gemini answers were for {prompts['image_size_wh']}")

    robot_prompts = [(lbl, tuple(box)) for lbl, box in prompts["robot"]]
    human_prompts = [(lbl, tuple(box)) for lbl, box in prompts["human"]]
    obj_prompts = [(lbl, tuple(box)) for lbl, box in prompts["objects"]]
    kept_items = prompts["object_items"]

    processor, model, device, torch = _load_sam3(sam3_model_id, device)
    all_masks = []
    summary = {"sam3_device": str(device), "inpaint_only": [], "objects": []}

    robot_results = _sam3_masks_batched(processor, model, device, torch, pil_img, robot_prompts)
    for (label, box), res in zip(robot_prompts, robot_results):
        res0 = res
        res = _inpaint_mask_or_box_fallback(label, box, res, h, w)
        summary["inpaint_only"].append({"label": label, "sam3": res0 is not None, "kept": res is not None,
                                        "box_fallback": res is not None and res is not res0})
        if res is None:
            continue
        mask_u8, (my0, mx0, my1, mx1) = res
        all_masks.append(SegmentationMask(y0=my0, x0=mx0, y1=my1, x1=mx1, mask=mask_u8, label=label))

    human_results = _sam3_masks_batched(processor, model, device, torch, pil_img, human_prompts)
    for (label, box), res in zip(human_prompts, human_results):
        res0 = res
        res = _inpaint_mask_or_box_fallback(label, box, res, h, w)
        summary["inpaint_only"].append({"label": label, "sam3": res0 is not None, "kept": res is not None,
                                        "box_fallback": res is not None and res is not res0})
        if res is None:
            continue
        mask_u8, (my0, mx0, my1, mx1) = res
        all_masks.append(SegmentationMask(y0=my0, x0=mx0, y1=my1, x1=mx1, mask=mask_u8, label=label))

    # Visualize Gemini-returned boxes + labels for debugging.
    debug_prompts = robot_prompts + human_prompts + obj_prompts
    _draw_boxes_debug(pil_img, debug_prompts, os.path.join(mask_dir, "gemini_boxes_debug.png"))
    with open(os.path.join(mask_dir, "gemini_boxes_raw.json"), "w") as _f:
        json.dump(gemini_boxes_raw(prompts), _f, indent=2)

    obj_results = _sam3_masks_batched(processor, model, device, torch, pil_img, obj_prompts)
    for item, (label, _), res in zip(kept_items, obj_prompts, obj_results):
        summary["objects"].append({"label": label, "sam3": res is not None})
        if res is None:
            print(f"[gembox] SAM3 returned no mask for {label!r}")
            continue
        mask_u8, (my0, mx0, my1, mx1) = res

        try:
            mass = float(item.get("mass", 0.0))
        except (TypeError, ValueError):
            mass = 0.0
        try:
            friction = float(item.get("friction", 0.0))
        except (TypeError, ValueError):
            friction = 0.0
        surface = str(item.get("surface_type", "unknown"))

        all_masks.append(SegmentationMask(
            y0=my0, x0=mx0, y1=my1, x1=mx1,
            mask=mask_u8, label=label,
            mass=mass, friction=friction, surface=surface,
        ))

    if not all_masks:
        raise RuntimeError("Gemini+SAM3 produced no usable masks.")

    _, object_dicts = plot_segmentation_masks(pil_img, all_masks, mask_dir, input_mask_dir, render_mask_dir)

    os.makedirs(save_dict, exist_ok=True)
    with open(os.path.join(save_dict, "frame_00001_bounding_boxes_and_masks.json"), "w") as f:
        json.dump(object_dicts, f, indent=4)
    summary["persisted"] = [d["object_name"] for d in object_dicts]
    return summary
