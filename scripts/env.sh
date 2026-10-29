# Source this before running any r2s step:   source scripts/env.sh
#
# Loads this machine's folders from configs/local.env (gitignored; template:
# configs/local.env.example), puts the repo's Python env first on PATH and sets what
# gsplat's CUDA extension needs (it JIT-compiles once, into $TORCH_EXTENSIONS_DIR,
# for every GPU type in TORCH_CUDA_ARCH_LIST). A variable already set in the
# environment keeps its value.

_r2s_repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]:-${(%):-%x}}")/.." && pwd)"
export R2S_REPO="$_r2s_repo"

if [[ -f "$R2S_REPO/configs/local.env" ]]; then
  while IFS= read -r _r2s_line || [[ -n "$_r2s_line" ]]; do
    case "$_r2s_line" in [A-Za-z_]*=*) ;; *) continue ;; esac
    _r2s_key="${_r2s_line%%=*}"
    case "$_r2s_key" in *[!A-Za-z0-9_]*) continue ;; esac
    eval "[ -n \"\${$_r2s_key:-}\" ]" || eval "export $_r2s_line"
  done < "$R2S_REPO/configs/local.env"
else
  echo "[r2s] no configs/local.env: cp configs/local.env.example configs/local.env and fill in this machine's folders" >&2
fi
unset _r2s_line _r2s_key

if [[ -n "${R2S_WORK_ROOT:-}" ]]; then
  export R2S_PYTHON="${R2S_PYTHON:-$R2S_WORK_ROOT/envs/eden/bin/python}"
  export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$R2S_WORK_ROOT/cache/torch_extensions}"
  export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-$R2S_WORK_ROOT/cache/numba}"   # Genesis' numba cache, not site-packages
  export PATH="$(dirname "$R2S_PYTHON"):${CUDA_HOME:+$CUDA_HOME/bin:}$PATH"
else
  echo "[r2s] R2S_WORK_ROOT is not set (configs/local.env)" >&2
fi
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0;8.6;8.9;9.0;12.0}"
export PYTHONDONTWRITEBYTECODE=1   # never drop __pycache__ into read-only checkouts

r2s() { "${R2S_PYTHON:?R2S_PYTHON is not set (configs/local.env)}" -m r2s "$@"; }
unset _r2s_repo
