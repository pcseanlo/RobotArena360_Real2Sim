import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from r2s.config import unset_variables  # noqa: E402

# Without configs/local.env (a fresh clone, CI) the machine folders are unset. Point
# them at a folder that does not exist, so Paths.load() works and every test that
# needs real data skips.
for _name in unset_variables():
    os.environ[_name] = f"/nonexistent/{_name}"
