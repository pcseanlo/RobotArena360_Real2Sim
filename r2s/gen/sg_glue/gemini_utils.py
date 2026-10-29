"""Mask persistence for step 0c-1, from SG ``src/utils/gemini_utils.py``.

Moved verbatim: ``INPAINT_ONLY_LABEL_KEYWORDS``, ``is_inpaint_only_label``,
``dilate_single_mask``, ``SegmentationMask``, ``overlay_mask_on_img``,
``plot_segmentation_masks``.

Changes in ``plot_segmentation_masks`` (paths only, no numerics):
- the binary inpaint masks + anchor copies go to an explicit ``input_mask_dir``
  (SG wrote ``<scene>/input_mask`` and the driver later moved it to
  ``data/input_mask/<scene>``);
- the optional robot render mask is read from an explicit ``render_mask_dir``
  (SG derived it with ``output_folder.replace("masks", "masks_render")``).
The Gemini SAM2/base64 parsers of the SG file are not used by the gembox variant
and are not moved.
"""

import dataclasses
import json
import os
import re

import cv2
import numpy as np
from PIL import Image, ImageColor, ImageDraw, ImageFont

additional_colors = [colorname for (colorname, colorcode) in ImageColor.colormap.items()]
INPAINT_ONLY_LABEL_KEYWORDS = {
    "robot",
    "gripper",
    "human",
    "person",
    "hand",
    "arm",
    "finger",
    "body",
    "torso",
    "head",
    "face",
    "hair",
    "shoulder",
    "elbow",
    "waist",
    "leg",
    "knee",
    "foot",
    "feet",
    "shoe",
}


def is_inpaint_only_label(label: str) -> bool:
    label_terms = set(re.sub(r"[^a-z0-9]+", " ", label.lower()).split())
    return bool(label_terms & INPAINT_ONLY_LABEL_KEYWORDS)


def dilate_single_mask(mask_img, min_area=1000, dilation_kernel_size=9, dilation_iterations=4):
    """Dilate a single mask after dropping contours smaller than ``min_area``.

    Returns a binary uint8 mask (0/255).
    """
    # Convert RGB to grayscale if needed
    if len(mask_img.shape) == 3:
        gray = cv2.cvtColor(mask_img, cv2.COLOR_RGB2GRAY)
    else:
        gray = mask_img

    # Threshold the grayscale mask to binary
    _, thresh = cv2.threshold(gray, 127, 255, 0)

    # Optionally apply a small dilation to connect close regions before contour extraction
    thresh = cv2.dilate(thresh, np.ones((3, 3), np.uint8), iterations=1)

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Filter out contours with area smaller than the minimum
    contours = [cnt for cnt in contours if cv2.contourArea(cnt) >= min_area]

    if not contours:
        print("No valid contours found.")
        return np.zeros_like(gray)

    # Create a binary mask from the filtered contours
    filtered_mask = np.zeros_like(gray)
    cv2.drawContours(filtered_mask, contours, -1, 255, -1)

    # Apply dilation to the filtered binary mask
    dilation_kernel = np.ones((dilation_kernel_size, dilation_kernel_size), np.uint8)
    dilated_mask = cv2.dilate(filtered_mask, dilation_kernel, iterations=dilation_iterations)

    return dilated_mask


@dataclasses.dataclass(frozen=True)
class SegmentationMask:
    # bounding box pixel coordinates (not normalized)
    y0: int  # in [0..height - 1]
    x0: int  # in [0..width - 1]
    y1: int  # in [0..height - 1]
    x1: int  # in [0..width - 1]
    mask: np.ndarray  # [img_height, img_width] with values 0..255
    label: str
    mass: float = 0.0
    friction: float = 0.0
    surface: str = "unknown"


def overlay_mask_on_img(img, mask, color, alpha=0.7):
    """Overlay one mask (values > 127) on a PIL image with a named color."""
    if not (0.0 <= alpha <= 1.0):
        raise ValueError("Alpha must be between 0.0 and 1.0")

    try:
        color_rgb = ImageColor.getrgb(color)
    except ValueError as e:
        raise ValueError(f"Invalid color name '{color}'. Supported names are typically HTML/CSS color names. Error: {e}")

    img_rgba = img.convert("RGBA")
    width, height = img_rgba.size

    alpha_int = int(alpha * 255)
    overlay_color_rgba = color_rgb + (alpha_int,)

    colored_mask_layer_np = np.zeros((height, width, 4), dtype=np.uint8)

    # Mask has values between 0 and 255, threshold at 127 to get binary mask.
    mask_np_logical = mask > 127

    colored_mask_layer_np[mask_np_logical] = overlay_color_rgba

    colored_mask_layer_pil = Image.fromarray(colored_mask_layer_np, "RGBA")

    result_img = Image.alpha_composite(img_rgba, colored_mask_layer_pil)

    return result_img


def plot_segmentation_masks(img, segmentation_masks, output_folder, input_mask_dir, render_mask_dir=None):
    """Persist the masks of one anchor image (SG's on-disk layout).

    Writes into ``output_folder`` (``<scene>/masks``): ``image_k_mask.png`` (anchor RGB
    with everything outside the object black), ``cropped/image_k_segmented.png`` (that
    image cropped to the box + 40 px) and ``result.json`` (name/count/mass/friction/
    surface per object). Writes into ``input_mask_dir``: ``image_k_mask.png`` (binary,
    contour-filtered and dilated) + ``image_k.png`` (the anchor) per object, and one
    more pair for the union of all inpaint-only (robot/human) masks.

    Returns ``(overlay_image, object_dicts)``; ``object_dicts`` is the content of
    ``bounding_boxes_and_masks/frame_00001_bounding_boxes_and_masks.json``.
    """
    colors = [
        "red",
        "green",
        "blue",
        "yellow",
        "orange",
        "pink",
        "purple",
        "brown",
        "gray",
        "beige",
        "turquoise",
        "cyan",
        "magenta",
        "lime",
        "navy",
        "maroon",
        "teal",
        "olive",
        "coral",
        "lavender",
        "violet",
        "gold",
        "silver",
    ] + additional_colors
    font = ImageFont.load_default()
    ori_image = img.copy()
    # Do this in 3 passes to make sure the boxes and text are always visible.
    # Overlay the mask
    for i, mask in enumerate(segmentation_masks):
        color = colors[i % len(colors)]
        img = overlay_mask_on_img(img, mask.mask, color)

    draw = ImageDraw.Draw(img)

    # Draw the bounding boxes
    for i, mask in enumerate(segmentation_masks):
        color = colors[i % len(colors)]
        draw.rectangle(((mask.x0, mask.y0), (mask.x1, mask.y1)), outline=color, width=4)

    # Draw the text labels
    for i, mask in enumerate(segmentation_masks):
        color = colors[i % len(colors)]
        if mask.label != "":
            draw.text((mask.x0 + 8, mask.y0 - 20), mask.label, fill=color, font=font)
    # Save cropped masks excluding robot/gripper
    object_dicts = []
    robot_mask = []
    physic_dicts = []
    path2 = str(input_mask_dir)
    print("Saving masks to ", path2)
    os.makedirs(path2, exist_ok=True)
    if render_mask_dir is not None and os.path.exists(render_mask_dir):
        rendered_mask = os.path.join(render_mask_dir, "1.png")
        robot_mask.append(cv2.imread(rendered_mask, cv2.IMREAD_GRAYSCALE))
    if output_folder:
        count = 0
        for segmentation_mask in segmentation_masks:
            if is_inpaint_only_label(segmentation_mask.label):
                robot_mask.append(segmentation_mask.mask)
                continue
            # 1.save the original image with mask and black background
            saved_mask = segmentation_mask.mask.copy().astype(bool)
            object_dicts.append(
                {
                    "object_name": segmentation_mask.label,
                    "mask": saved_mask.tolist(),
                    "count": count,
                }
            )
            physic_dicts.append(
                {
                    "object_name": segmentation_mask.label,
                    "count": [count],
                    "mass": segmentation_mask.mass,
                    "friction": segmentation_mask.friction,
                    "surface": segmentation_mask.surface,
                }
            )
            original_img = ori_image.copy()
            original_img = np.array(original_img)
            original_img[segmentation_mask.mask == 0] = 0
            bnw = np.zeros_like(original_img, dtype=np.uint8)
            bnw[segmentation_mask.mask != 0] = 255
            # if bbox small adjust min_area
            bnw = dilate_single_mask(bnw, min_area=300)
            cv2.imwrite(os.path.join(path2, f"image_{count:04d}_mask.png"), bnw)
            ori_image.save(os.path.join(path2, f"image_{count:04d}.png"), "PNG")

            original_img = Image.fromarray(original_img)
            original_img.save(os.path.join(output_folder, f"image_{count:04d}_mask.png"), "PNG")

            # Get bounding box
            padding = 40
            img_width, img_height = original_img.size
            x0 = max(int(segmentation_mask.x0) - padding, 0)
            y0 = max(int(segmentation_mask.y0) - padding, 0)
            x1 = min(int(segmentation_mask.x1) + padding, img_width)
            y1 = min(int(segmentation_mask.y1) + padding, img_height)
            cropped_img = original_img.crop((x0, y0, x1, y1))
            path_dir = os.path.join(output_folder, "cropped")
            os.makedirs(path_dir, exist_ok=True)
            cropped_img.save(os.path.join(path_dir, f"image_{count:04d}_segmented.png"), "PNG")
            count += 1

        if robot_mask:
            combined_inpaint_mask = np.zeros((ori_image.height, ori_image.width), dtype=np.uint8)
            for segmentation_mask in robot_mask:
                if segmentation_mask is None:
                    continue
                if len(segmentation_mask.shape) == 3:
                    segmentation_mask = cv2.cvtColor(segmentation_mask, cv2.COLOR_RGB2GRAY)
                if segmentation_mask.shape != combined_inpaint_mask.shape:
                    segmentation_mask = cv2.resize(
                        segmentation_mask,
                        (ori_image.width, ori_image.height),
                        interpolation=cv2.INTER_NEAREST,
                    )
                combined_inpaint_mask = np.maximum(
                    combined_inpaint_mask,
                    (segmentation_mask > 0).astype(np.uint8) * 255,
                )

            if np.count_nonzero(combined_inpaint_mask) > 0:
                inpaint_mask = (combined_inpaint_mask > 0).astype(np.uint8) * 255
                inpaint_mask = cv2.dilate(inpaint_mask, np.ones((5, 5), np.uint8), iterations=1)
                cv2.imwrite(os.path.join(path2, f"image_{count:04d}_mask.png"), inpaint_mask)
                ori_image.save(os.path.join(path2, f"image_{count:04d}.png"), "PNG")
            else:
                print("No non-empty inpaint-only mask found.")
        # save results to output folder
        with open(os.path.join(output_folder, "result.json"), "w") as f:
            json.dump(physic_dicts, f, indent=4)
    return img, object_dicts
