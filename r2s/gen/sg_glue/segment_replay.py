"""Saved Gemini answers -> SAM3 box prompts, for replaying step 0c-1 without Gemini.

Two inputs are accepted:

``gemini_answers.json`` (written by r2s live runs)
    The three raw Gemini replies (robot, human, objects) with their canvases. The
    prompts are rebuilt exactly as in a live run (:func:`gembox.prompts_from_answers`).

``gemini_boxes_raw.json`` + ``result.json`` (what SG runs left on disk)
    ``gemini_boxes_raw.json`` has the object boxes (raw 0-1000 and denormalised) and,
    since mid-May 2026, the robot/human boxes as SG fed them to SAM3 (already
    prefixed and expanded). Files written before that have no robot/human key
    (droid2): then there were no inpaint-only prompts. SG never saved the
    per-object mass/friction/surface Gemini returned except in ``result.json``, so
    they are taken from there (matched in order by name; objects SAM3 dropped in
    the old run are not in ``result.json`` and get SG's defaults 0/0/"unknown").
    Trailing ``result.json`` entries that match no prompt (later steps append the
    table, e.g. droid2) are ignored and reported.

New in r2s (not an SG file). Imports only the standard library and its sibling SG
modules, so it runs in the ``gemini`` env (``segment_gembox.py``) and in the repo env
(``inputs.segment`` validates a replay before clearing the folders it lives in).
"""

import json
import os

from .gembox import _denorm_box, prompts_from_answers
from .gemini_utils import is_inpaint_only_label


def is_answers_file(data):
    return isinstance(data, dict) and all(k in data for k in ("robot", "human", "objects"))


def split_inpaint_only(entries):
    """Split SG's ``robot_prompts + human_prompts`` list back into the two batches.

    SG prefixes every robot label so it contains the word ``robot`` and every human
    label so it contains ``human``; robots come first.
    """
    robot, human = [], []
    for label, box in entries:
        words = label.lower().split()
        if not human and "robot" in words:
            robot.append((label, box))
        elif "human" in words:
            human.append((label, box))
        else:
            raise ValueError(f"cannot tell whether inpaint-only prompt {label!r} was a robot or a human box")
    return robot, human


def prompts_from_boxes_raw(raw, result_entries):
    """Rebuild SG's prompts from ``gemini_boxes_raw.json`` + ``result.json``. Returns (prompts, notes)."""
    notes = []
    w, h = raw["image_size_wh"]
    side = raw["canvas_side"]
    pad_x, pad_y = raw["canvas_pad_xy"]

    recomputed = []
    for item in raw["items"]:
        y0, x0, y1, x1 = _denorm_box(item["box_2d_raw"], side, pad_x, pad_y, w, h)
        if y1 <= y0 or x1 <= x0:
            continue
        label = "object" if item.get("label") is None else str(item["label"])
        recomputed.append((label, [x0, y0, x1, y1]))
    stored = [(d["label"], list(d["xyxy"])) for d in raw["denormalized_xyxy"]]
    if recomputed != stored:
        raise ValueError(f"denormalized_xyxy does not follow from the raw boxes:\n  stored     {stored}\n  recomputed {recomputed}")
    objects = [(lbl, tuple(box)) for lbl, box in stored]

    if "inpaint_only_denormalized_xyxy" in raw:
        inpaint = [(d["label"], tuple(d["xyxy"])) for d in raw["inpaint_only_denormalized_xyxy"]]
    else:
        inpaint = []
        notes.append("gemini_boxes_raw.json predates the robot/human prompts: no inpaint-only boxes")
    robot, human = split_inpaint_only(inpaint)

    entries = list(result_entries)
    k = 0
    object_items = []
    for label, _ in objects:
        if is_inpaint_only_label(label):
            object_items.append({})  # persisted as inpaint-only; physics never written
            continue
        if k < len(entries) and str(entries[k]["object_name"]) == label:
            e = entries[k]
            k += 1
            object_items.append({"mass": e["mass"], "friction": e["friction"], "surface_type": e["surface"]})
        else:
            notes.append(f"{label!r} is not in result.json (old SAM3 dropped it); physics would default to 0/0/unknown")
            object_items.append({})
    for e in entries[k:]:
        notes.append(f"result.json entry {e.get('object_name')!r} (count {e.get('count')}) matches no Gemini box; "
                     "assumed appended by a later step and ignored")

    prompts = {
        "image_size_wh": [w, h],
        "canvas_side": side,
        "canvas_pad_xy": [pad_x, pad_y],
        "raw_items": raw["items"],
        "robot": robot,
        "human": human,
        "objects": objects,
        "object_items": object_items,
    }
    return prompts, notes


def load_replay(gemini_path, result_path=None):
    """Prompts for a replay run from a saved answers file. Returns (prompts, notes, kind)."""
    with open(gemini_path) as f:
        data = json.load(f)
    if is_answers_file(data):
        return prompts_from_answers(data), [], "gemini_answers"
    if result_path is None:
        result_path = os.path.join(os.path.dirname(gemini_path), "result.json")
    with open(result_path) as f:
        result_entries = json.load(f)
    prompts, notes = prompts_from_boxes_raw(data, result_entries)
    return prompts, notes, "gemini_boxes_raw"
