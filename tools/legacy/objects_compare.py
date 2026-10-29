"""Compare collision / object outputs of two runs (new r2s vs old code, or old code vs disk).

    python tools/legacy/objects_compare.py json A.json B.json [--atol 0] [--rtol 0] [--ignore key ...]
    python tools/legacy/objects_compare.py mesh A.glb B.glb
    python tools/legacy/objects_compare.py log A/fit.log B/fit.log      # paths stripped
    python tools/legacy/objects_compare.py batch pairs.json --out report.json

``json`` is a recursive diff: dict keys, list lengths, strings and bools must match;
numbers are compared with ``|a-b| <= atol + rtol*|b|`` (NaN == NaN, -0.0 == 0.0) and the
max absolute / relative difference is reported. Provenance keys (file paths,
timestamps) are ignored by default (``DEFAULT_IGNORE``). ``mesh`` loads both GLBs with
trimesh (``force="mesh", process=False``) and compares the vertex and face arrays.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

# Keys whose values are provenance (paths of the run that produced the file).
DEFAULT_IGNORE = ("source", "source_dump", "scene_dir", "inputs", "descriptor", "scene_meta_dir")


def _is_num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


class JsonDiff:
    def __init__(self, atol: float = 0.0, rtol: float = 0.0, ignore=DEFAULT_IGNORE, max_report: int = 20):
        self.atol, self.rtol = float(atol), float(rtol)
        self.ignore = set(ignore)
        self.max_report = max_report
        self.n_numbers = 0
        self.n_num_diff = 0
        self.max_abs = 0.0
        self.max_rel = 0.0
        self.max_abs_path = ""
        self.mismatches: list[str] = []
        self.n_mismatch = 0

    def _bad(self, path: str, msg: str) -> None:
        self.n_mismatch += 1
        if len(self.mismatches) < self.max_report:
            self.mismatches.append(f"{path or '<root>'}: {msg}")

    def compare(self, a: Any, b: Any, path: str = "") -> None:
        if _is_num(a) and _is_num(b):
            self.n_numbers += 1
            fa, fb = float(a), float(b)
            if math.isnan(fa) and math.isnan(fb):
                return
            if math.isinf(fa) or math.isinf(fb):
                if fa != fb:
                    self.n_num_diff += 1
                    self._bad(path, f"{a!r} != {b!r}")
                return
            d = abs(fa - fb)
            rel = d / max(abs(fb), 1e-300) if d else 0.0
            if d > self.max_abs:
                self.max_abs, self.max_abs_path = d, path
            self.max_rel = max(self.max_rel, rel)
            if d > self.atol + self.rtol * abs(fb):
                self.n_num_diff += 1
                self._bad(path, f"{a!r} != {b!r} (|d|={d:.3g})")
            return
        if type(a) is not type(b) and not (a is None or b is None):
            self._bad(path, f"type {type(a).__name__} != {type(b).__name__}: {str(a)[:60]!r} vs {str(b)[:60]!r}")
            return
        if isinstance(a, dict):
            for k in sorted(set(a) | set(b)):
                if k in self.ignore:
                    continue
                p = f"{path}.{k}" if path else str(k)
                if k not in a:
                    self._bad(p, "missing in A")
                elif k not in b:
                    self._bad(p, "missing in B")
                else:
                    self.compare(a[k], b[k], p)
            return
        if isinstance(a, list):
            if len(a) != len(b):
                self._bad(path, f"list length {len(a)} != {len(b)}")
            for i, (x, y) in enumerate(zip(a, b)):
                self.compare(x, y, f"{path}[{i}]")
            return
        if a != b:
            self._bad(path, f"{str(a)[:80]!r} != {str(b)[:80]!r}")

    @property
    def equal(self) -> bool:
        return self.n_mismatch == 0

    def summary(self) -> dict[str, Any]:
        return {
            "equal": self.equal,
            "numbers": self.n_numbers,
            "numbers_over_tol": self.n_num_diff,
            "other_mismatches": self.n_mismatch - self.n_num_diff,
            "max_abs": self.max_abs,
            "max_abs_at": self.max_abs_path,
            "max_rel": self.max_rel,
            "first": self.mismatches,
        }


def _alias(value: Any, aliases) -> Any:
    if isinstance(value, str):
        for old, new in aliases:
            if value.startswith(old):
                value = new + value[len(old):]
        return value
    if isinstance(value, dict):
        return {k: _alias(v, aliases) for k, v in value.items()}
    if isinstance(value, list):
        return [_alias(v, aliases) for v in value]
    return value


def compare_json(a_path, b_path, *, atol=0.0, rtol=0.0, ignore=DEFAULT_IGNORE, aliases=()) -> dict[str, Any]:
    """``aliases``: ``[(prefix, replacement), ...]`` applied to every string of both files
    (e.g. another mount point of the same data root, ``R2S_OLD_PATH_ALIASES``)."""
    a = _alias(json.loads(Path(a_path).read_text()), aliases)
    b = _alias(json.loads(Path(b_path).read_text()), aliases)
    d = JsonDiff(atol=atol, rtol=rtol, ignore=ignore)
    d.compare(a, b)
    return {"kind": "json", "a": str(a_path), "b": str(b_path), **d.summary()}


def compare_mesh(a_path, b_path) -> dict[str, Any]:
    import numpy as np
    import trimesh

    a = trimesh.load(str(a_path), force="mesh", process=False)
    b = trimesh.load(str(b_path), force="mesh", process=False)
    va, vb = np.asarray(a.vertices), np.asarray(b.vertices)
    fa, fb = np.asarray(a.faces), np.asarray(b.faces)
    out: dict[str, Any] = {
        "kind": "mesh",
        "a": str(a_path),
        "b": str(b_path),
        "vertices": [list(va.shape), list(vb.shape)],
        "faces": [list(fa.shape), list(fb.shape)],
        "bytes_equal": Path(a_path).read_bytes() == Path(b_path).read_bytes(),
    }
    if va.shape == vb.shape and fa.shape == fb.shape:
        out["faces_equal"] = bool((fa == fb).all())
        out["vertices_bit_equal"] = bool((va == vb).all())
        out["max_abs"] = float(np.abs(va - vb).max()) if va.size else 0.0
        out["equal"] = out["faces_equal"] and out["vertices_bit_equal"]
    else:
        out["equal"] = False
    return out


_PATH_RE = re.compile(r"(/[^\s'\",)]+)+")
_TIME_RE = re.compile(r"\(\d+(\.\d+)? s\)")


def _normalize_log(text: str) -> list[str]:
    return [_TIME_RE.sub("(<t> s)", _PATH_RE.sub("<path>", line)).rstrip() for line in text.splitlines() if line.strip()]


def compare_log(a_path, b_path) -> dict[str, Any]:
    a = _normalize_log(Path(a_path).read_text())
    b = _normalize_log(Path(b_path).read_text())
    diff = [f"A: {x}\nB: {y}" for x, y in zip(a, b) if x != y]
    return {
        "kind": "log",
        "a": str(a_path),
        "b": str(b_path),
        "lines": [len(a), len(b)],
        "equal": a == b,
        "first": diff[:10],
    }


def compare_any(kind: str, a, b, **kw) -> dict[str, Any]:
    if kind == "json":
        return compare_json(a, b, **kw)
    if kind == "mesh":
        return compare_mesh(a, b)
    if kind == "log":
        return compare_log(a, b)
    raise ValueError(kind)


def _print(result: dict[str, Any]) -> None:
    status = "EQUAL" if result.get("equal") else "DIFF"
    head = f"[{result['kind']}] {status}  {result.get('label', '')}".rstrip()
    print(head)
    print(f"    A {result['a']}\n    B {result['b']}")
    for k in ("numbers", "numbers_over_tol", "other_mismatches", "max_abs", "max_abs_at", "max_rel", "vertices", "faces", "bytes_equal", "lines"):
        if k in result:
            print(f"    {k}: {result[k]}")
    for line in result.get("first") or []:
        print(f"      {line}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    pj = sub.add_parser("json")
    pj.add_argument("a")
    pj.add_argument("b")
    pj.add_argument("--atol", type=float, default=0.0)
    pj.add_argument("--rtol", type=float, default=0.0)
    pj.add_argument("--ignore", nargs="*", default=list(DEFAULT_IGNORE))
    pj.add_argument("--alias", nargs=2, action="append", default=[], metavar=("PREFIX", "REPLACEMENT"),
                    help="default: $R2S_OLD_PATH_ALIASES")
    pm = sub.add_parser("mesh")
    pm.add_argument("a")
    pm.add_argument("b")
    pl = sub.add_parser("log")
    pl.add_argument("a")
    pl.add_argument("b")
    pb = sub.add_parser("batch", help='JSON list of {"label", "kind", "a", "b", ["atol", "rtol", "ignore"]}')
    pb.add_argument("pairs")
    pb.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    if args.cmd == "batch":
        results = []
        for item in json.loads(Path(args.pairs).read_text()):
            kw = {k: item[k] for k in ("atol", "rtol", "ignore", "aliases") if k in item and item["kind"] == "json"}
            try:
                r = compare_any(item["kind"], item["a"], item["b"], **kw)
            except FileNotFoundError as exc:
                r = {"kind": item["kind"], "a": item["a"], "b": item["b"], "equal": False, "first": [f"missing: {exc}"]}
            r["label"] = item.get("label", "")
            _print(r)
            results.append(r)
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(json.dumps(results, indent=2) + "\n")
        return 0 if all(r.get("equal") for r in results) else 1
    if args.cmd == "json":
        if args.alias:
            aliases = [tuple(x) for x in args.alias]
        else:
            from old_repo import PATH_ALIASES  # sibling module
            aliases = PATH_ALIASES
        r = compare_json(args.a, args.b, atol=args.atol, rtol=args.rtol, ignore=args.ignore, aliases=aliases)
    else:
        r = compare_any(args.cmd, args.a, args.b)
    _print(r)
    return 0 if r.get("equal") else 1


if __name__ == "__main__":
    sys.exit(main())
