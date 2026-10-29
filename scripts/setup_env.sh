#!/usr/bin/env bash
# Build the Python environment r2s runs in: Eden's locked dependencies (with the
# `vis` extra = gsplat, plyfile, viser) plus the few extras r2s needs. Eden itself
# and this repo are added through .pth files, so nothing is installed into (or
# written inside) the Eden checkout.
#
#   bash scripts/setup_env.sh          # Eden = $R2S_EDEN, env = $R2S_PYTHON's folder (configs/local.env)
#   EDEN=/path/to/Eden ENV_DIR=/path/to/env bash scripts/setup_env.sh
#
# Then build gsplat's CUDA extension once (needs a node with nvcc 13.0; ~6 min):
#   source scripts/env.sh && python -c "from gsplat.cuda._backend import _C"
set -euo pipefail

REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO/scripts/env.sh"
EDEN="${EDEN:-${R2S_EDEN:?set R2S_EDEN in configs/local.env}}"
ENV_DIR="${ENV_DIR:-$(dirname "$(dirname "${R2S_PYTHON:?set R2S_WORK_ROOT in configs/local.env}")")}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$R2S_WORK_ROOT/uv-cache}"   # keep the multi-GB cache off $HOME
export UV_LINK_MODE=copy

echo "[setup] Eden checkout : $EDEN ($(git -C "$EDEN" rev-parse --abbrev-ref HEAD) @ $(git -C "$EDEN" rev-parse --short HEAD))"
echo "[setup] environment   : $ENV_DIR"

# 1. Eden's locked dependencies, without installing Eden itself (no build files in the checkout).
UV_PROJECT_ENVIRONMENT="$ENV_DIR" UV_PYTHON="$(cat "$EDEN/.python-version" 2>/dev/null || echo 3.12)" \
  uv sync --project "$EDEN" --frozen --extra vis --no-install-project --no-dev

# 2. What r2s needs beyond Eden's lock, without changing any package the lock installed
#    (cma: pose fit; ninja: gsplat's JIT build; pytest: tests; open3d: world.collider's
#    Poisson mesher, the version the old ply2glb env had).
LOCKED="$(mktemp)"
uv pip freeze --python "$ENV_DIR/bin/python" > "$LOCKED"
uv pip install --python "$ENV_DIR/bin/python" -c "$LOCKED" cma ninja pytest open3d==0.19.0
rm -f "$LOCKED"

# 3. Make Eden and r2s importable (read-only source checkouts).
SITE="$("$ENV_DIR/bin/python" -c 'import site; print(site.getsitepackages()[0])')"
echo "$EDEN" > "$SITE/_eden_checkout.pth"
echo "$REPO" > "$SITE/_r2s_repo.pth"

PYTHONDONTWRITEBYTECODE=1 "$ENV_DIR/bin/python" - <<'EOF'
import eden, gsplat, genesis, cma, open3d, r2s, torch
print(f"[setup] ok: eden={eden.__file__} gsplat={gsplat.__version__} genesis={genesis.__version__} "
      f"torch={torch.__version__} open3d={open3d.__version__}")
EOF
