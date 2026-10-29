#!/usr/bin/env python3
"""Step 0c-2 worker: image-grounded Gemini physics query (env ``gemini``).

SG ``src/pipeline/prompt_physics_updated.py``: for every object of
``masks/result.json`` Gemini (``models/gemini-2.5-flash``) gets the cropped object
image and its binary inpaint mask and returns width/height/length (m), mass (kg),
friction and surface type -> ``masks/physical_properties.json``. Needs
``$GEMINI_API_KEY``; the raw replies are saved next to the output.
"""

import argparse
import os
import sys
from pathlib import Path

# sg_glue lives in r2s/gen; append (not prepend) so it cannot shadow env packages.
sys.path.append(str(Path(__file__).resolve().parents[1]))

from sg_glue import physics_utils  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--mask-dir", required=True, help="<meta>/masks (result.json + cropped/)")
    p.add_argument("--input-mask-dir", required=True, help="binary masks (<scene_data>/input_mask/<s>)")
    p.add_argument("--out", required=True, help="physical_properties.json")
    p.add_argument("--answers-out", default=None, help="raw Gemini replies (JSON)")
    p.add_argument("--model-name", default="models/gemini-2.5-flash")
    p.add_argument("--delay", type=float, default=1.0)
    args = p.parse_args(argv)

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("needs GEMINI_API_KEY in the environment (or run the step with --replay)")
    physics_utils.query_physical_properties_with_images(
        os.path.join(args.mask_dir, "result.json"), args.out, args.mask_dir, args.input_mask_dir,
        api_key, args.model_name, delay=args.delay, answers_out=args.answers_out,
    )


if __name__ == "__main__":
    main()
