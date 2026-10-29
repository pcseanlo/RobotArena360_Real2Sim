"""scene_generation (SG) glue code used by the step-0 workers, copied with path fixes.

The workers in ``r2s/gen/workers/`` run inside the model conda envs and never
import the ``r2s`` package. Every worker appends ``r2s/gen`` to ``sys.path`` and
imports this folder as the package ``sg_glue`` (``from sg_glue import gembox``,
``from sg_glue.match_find_rot import process_mesh``); modules in here import each
other with relative imports (``from .gemini_utils import ...``), never flat. The
repo env imports the light ones as ``r2s.gen.sg_glue`` (tests; ``inputs.segment``
validates a replay with ``segment_replay``).

Modules import only the standard library and what their env provides. Function
bodies, constants and operation order are SG's; each module's docstring lists
what changed. Third-party code (LightGlue, MINIMA weights, Genesis 0.2.1, SAM3,
Depth-Anything) stays external.

    gemini_utils.py       <- src/utils/gemini_utils.py (plot_segmentation_masks, ...)
    gembox.py             <- src/pipeline/segment_object_gembox_updated.py (Gemini boxes + SAM3)
    segment_replay.py     new: saved Gemini answers -> SAM3 prompts (inputs.segment --replay-gemini)
    pointcloud.py         <- SamGemini.mask_to_point_cloud
    physics_utils.py      <- src/utils/physics_utils.py
    matching_utils.py     <- src/utils/matching_utils.py
    dataloader_utils.py   <- src/utils/dataloader_utils.py (SamGemini, scene loading only)
    simulator_genesis.py  <- src/sim/simulator_genesis.py (mesh scene only)
    minima_sp_lg.py       <- src/utils/mini_utils.py + src/utils/MINIMA/{load_model,data_io_sp_lg}.py
    match_find_rot.py     <- src/pipeline/MatchFindRot.py (the original, as in production)
    match_config.yaml     <- the keys of configs/config_DROID.yaml that MatchFindRot reads
    align_inpaint.py      <- src/pipeline/align_inpaint_updated.py
    hunyuanworld.py       <- src/utils/hunyuanworld.py
"""
