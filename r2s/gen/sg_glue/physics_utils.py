"""Image-grounded physics query for step 0c-2, from SG ``src/utils/physics_utils.py``.

Moved verbatim: ``ask_gemini_about_physical_properties`` (text-only fallback),
``_parse_physics_response``, ``ask_gemini_about_physical_properties_with_image``,
``query_physical_properties_with_images``.

Changes: the API key and model name are arguments (SG read both from
``configs/config_DROID.yaml``; r2s passes ``$GEMINI_API_KEY``), the binary-mask
folder is an argument (SG: ``<mask_dir>/../input_mask``, before the driver moved it
to ``data/input_mask/<s>``), and ``google.generativeai`` is imported lazily.
"""

import json
import os
import time


def ask_gemini_about_physical_properties(object_name, model, retries=3):
    """Query Gemini for estimated mass and friction of an object."""

    prompt = (
        f"You are an asset augmentation tool for simulation environments. Please analyze the physical properties of the object: {object_name}'.\n"
        f"Return three values: estimated with, height and length of the object, estimated mass in kilograms, coefficient of friction and surface type.\n"
        f"Use consistent and reasonable values so that similar assets receive similar properties.\n"
        f"Output surface type should be chosen from the following list: [Glass, Water, Emission, Plastic, Rough, Smooth, Reflective, Metal, Iron, Aluminium, Copper, Gold].\n"
        f"Use this format exactly: width: <value in meter>, height: <value in meter>, length: <value in meter>, mass: <value in kg>, friction: <value>, surface: <surface type>.\n"
        f"If unknown, make a reasonable guess based on common use and surface.\n"
        f"Do not ask followup questions. The output must contain no additional commentary or extraneous text."
    )

    for attempt in range(retries):
        try:
            response = model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            print(f"[WARNING] Attempt {attempt+1} failed for {object_name}: {e}")
            time.sleep(2)
    return "Error: Failed to retrieve response"


def _parse_physics_response(answer):
    width, height, length, mass, friction, surface = [x.strip() for x in answer.split(",")]
    if width.split(": ")[0] != "width":
        raise ValueError(f"Unexpected response format for width: {width}")
    width = float(width.split(": ")[1])
    if height.split(": ")[0] != "height":
        raise ValueError(f"Unexpected response format for height: {height}")
    height = float(height.split(": ")[1])
    if length.split(": ")[0] != "length":
        raise ValueError(f"Unexpected response format for length: {length}")
    length = float(length.split(": ")[1])
    if mass.split(": ")[0] != "mass":
        raise ValueError(f"Unexpected response format for mass: {mass}")
    mass = float(mass.split(": ")[1])
    if friction.split(": ")[0] != "friction":
        raise ValueError(f"Unexpected response format for friction: {friction}")
    friction = float(friction.split(": ")[1])
    if surface.split(": ")[0] != "surface":
        raise ValueError(f"Unexpected response format for surface: {surface}")
    surface = surface.split(": ")[1]
    return width, height, length, mass, friction, surface


def ask_gemini_about_physical_properties_with_image(object_name, cropped_image_path, mask_path, model, retries=3):
    """Query Gemini for physical properties using the cropped object image + binary mask."""
    from PIL import Image as _PILImage

    prompt = (
        f"You are an asset augmentation tool for simulation environments. "
        f"Analyze the physical properties of the object '{object_name}' shown in the "
        f"attached cropped image. The second attached image is a binary mask isolating "
        f"the object's pixels — use it to disambiguate the object from the background.\n"
        f"Return: estimated width, height and length of the object (in meters), "
        f"estimated mass (kg), coefficient of friction, and surface type.\n"
        f"Use consistent, reasonable values so that similar assets receive similar properties.\n"
        f"Surface type must be one of: [Glass, Water, Emission, Plastic, Rough, Smooth, "
        f"Reflective, Metal, Iron, Aluminium, Copper, Gold].\n"
        f"Use this format exactly: width: <value in meter>, height: <value in meter>, "
        f"length: <value in meter>, mass: <value in kg>, friction: <value>, surface: <surface type>.\n"
        f"If unknown, make a reasonable visual guess. No commentary, no follow-up questions."
    )

    crop = _PILImage.open(cropped_image_path).convert("RGB")
    mask_img = _PILImage.open(mask_path).convert("L")

    for attempt in range(retries):
        try:
            response = model.generate_content([prompt, crop, mask_img])
            return response.text.strip()
        except Exception as e:
            print(f"[WARNING] Image-grounded attempt {attempt+1} failed for {object_name}: {e}")
            time.sleep(2)
    return "Error: Failed to retrieve response"


def query_physical_properties_with_images(input_json, output_json, mask_dir, input_mask_dir, api_key, model_name,
                                          delay=1.0, answers_out=None):
    """Ask Gemini for width/height/length/mass/friction/surface of every object in ``result.json``.

    Uses ``<mask_dir>/cropped/image_NNNN_segmented.png`` and
    ``<input_mask_dir>/image_NNNN_mask.png`` (from each entry's ``count``).
    """
    import google.generativeai as genai

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name=model_name)

    with open(input_json, "r") as f:
        object_entries = json.load(f)

    results = []
    raw_answers = []
    for entry in object_entries:
        object_name = entry["object_name"].strip()
        count = entry["count"]
        if isinstance(count, list):
            count_id = count[0]
        else:
            count_id = count
        cropped = os.path.join(mask_dir, "cropped", f"image_{count_id:04d}_segmented.png")
        mask_path = os.path.join(input_mask_dir, f"image_{count_id:04d}_mask.png")

        if not (os.path.exists(cropped) and os.path.exists(mask_path)):
            print(f"[WARN] missing cropped/mask for {object_name} (count={count_id}); "
                  f"falling back to text-only query.")
            answer = ask_gemini_about_physical_properties(object_name, model)
        else:
            print(f"[INFO] Querying Gemini (image+mask) for: {object_name} (id: {count_id})")
            answer = ask_gemini_about_physical_properties_with_image(object_name, cropped, mask_path, model)
        raw_answers.append({"object_name": object_name, "count": count, "answer": answer})

        try:
            width, height, length, mass, friction, surface = _parse_physics_response(answer)
        except Exception as e:
            print(f"[ERROR] Failed to parse response for {object_name}: {e}")
            width = height = length = mass = friction = None
            surface = None

        results.append({
            "object_name": object_name,
            "count": count,
            "width": width,
            "height": height,
            "length": length,
            "mass": mass,
            "friction": friction,
            "surface": surface,
        })
        time.sleep(delay)

    with open(output_json, "w") as f:
        json.dump(results, f, indent=4)
    if answers_out:
        with open(answers_out, "w") as f:
            json.dump({"model_name": model_name, "answers": raw_answers}, f, indent=2)
    print(f"[INFO] Image-grounded results saved to {output_json}")
    return results
