"""Compare surface-step outputs (JSON descriptors and PLYs) against a reference.

    python tools/legacy/surface_compare.py OLD NEW [OLD NEW ...] [--tol 1e-9] [--out report.json]

JSON: recursive walk; numbers are compared by absolute difference and reported as the
max abs diff per key path (list indices collapsed to ``[]``); strings/bools/None must be
equal; missing keys, length or type mismatches are structural differences. Provenance
fields (source file paths) are ignored by default, see ``DEFAULT_IGNORE``.

PLY: every element must have the same count and properties; each property reports its
max abs diff (list properties such as ``face.vertex_indices`` are compared row by row).
A byte-identical file short-circuits the comparison.

Exit status is 0 when every pair is within ``--tol`` and structurally equal, 1 otherwise.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# Dotted key paths (fnmatch patterns, list indices written as ``[]``) that carry
# provenance rather than results: absolute paths of the input files.
DEFAULT_IGNORE = (
    "source.collider_glb",
    "source.result",
    "source.scene_scale",
    "source.operation_surface_json",
    "source.gaussian_ply",
    "parameters.camera_intrinsics",
    "parameters.camera_extrinsics",
    "*.placement_lineage",  # provenance string; renamed from the old script name to r2s.align.scale_fit
)


@dataclass
class Report:
    kind: str
    old: str
    new: str
    numeric: dict[str, dict[str, float]] = field(default_factory=dict)  # path -> {max_abs, count}
    mismatches: list[str] = field(default_factory=list)  # non-numeric value differences
    structural: list[str] = field(default_factory=list)  # missing keys, shapes, types
    ignored: list[str] = field(default_factory=list)
    note: str = ""

    @property
    def max_abs(self) -> float:
        return max((v["max_abs"] for v in self.numeric.values()), default=0.0)

    def verdict(self, tol: float) -> str:
        if self.structural or self.mismatches:
            return "DIFFERS"
        if self.max_abs == 0.0:
            return "identical"
        return "within-tol" if self.max_abs <= tol else "DIFFERS"

    def add_numeric(self, path: str, diff: float, count: int = 1) -> None:
        entry = self.numeric.setdefault(path, {"max_abs": 0.0, "count": 0})
        entry["max_abs"] = max(entry["max_abs"], float(diff))
        entry["count"] += int(count)


# ------------------------------------------------------------------ JSON
def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _num_diff(a: float, b: float) -> float:
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return 0.0
    if a == b:  # also equal infinities
        return 0.0
    return abs(float(a) - float(b))


def _join(path: str, key: str) -> str:
    return f"{path}.{key}" if path else key


def _walk(a: Any, b: Any, path: str, report: Report, ignore: tuple[str, ...]) -> None:
    if any(fnmatch.fnmatchcase(path, pattern) for pattern in ignore):
        if path not in report.ignored:
            report.ignored.append(path)
        return
    if _is_number(a) and _is_number(b):
        report.add_numeric(path or "<root>", _num_diff(a, b))
        if isinstance(a, int) != isinstance(b, int):
            report.mismatches.append(f"{path}: int/float {a!r} vs {b!r}")
        return
    if type(a) is not type(b):
        report.structural.append(f"{path}: type {type(a).__name__} vs {type(b).__name__}")
        return
    if isinstance(a, dict):
        for key in a.keys() - b.keys():
            report.structural.append(f"{_join(path, key)}: only in old")
        for key in b.keys() - a.keys():
            report.structural.append(f"{_join(path, key)}: only in new")
        for key in a.keys() & b.keys():
            _walk(a[key], b[key], _join(path, key), report, ignore)
        return
    if isinstance(a, list):
        if len(a) != len(b):
            report.structural.append(f"{path}: length {len(a)} vs {len(b)}")
            return
        for item_a, item_b in zip(a, b):
            _walk(item_a, item_b, f"{path}[]", report, ignore)
        return
    if a != b:
        report.mismatches.append(f"{path}: {a!r} vs {b!r}")


def compare_json(old: Path, new: Path, ignore: tuple[str, ...]) -> Report:
    report = Report("json", str(old), str(new))
    _walk(json.loads(old.read_text()), json.loads(new.read_text()), "", report, ignore)
    return report


# ------------------------------------------------------------------ PLY
def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def _column_diff(a: np.ndarray, b: np.ndarray) -> float:
    if a.dtype == object or b.dtype == object:  # list property: one array per row
        worst = 0.0
        for row_a, row_b in zip(a, b):
            row_a, row_b = np.asarray(row_a), np.asarray(row_b)
            if row_a.shape != row_b.shape:
                return math.inf
            if row_a.size:
                worst = max(worst, float(np.max(np.abs(row_a.astype(np.float64) - row_b.astype(np.float64)))))
        return worst
    a64, b64 = a.astype(np.float64), b.astype(np.float64)
    both_nan = np.isnan(a64) & np.isnan(b64)
    diff = np.where(both_nan | (a64 == b64), 0.0, np.abs(a64 - b64))
    return float(diff.max()) if diff.size else 0.0


def compare_ply(old: Path, new: Path) -> Report:
    from plyfile import PlyData

    report = Report("ply", str(old), str(new))
    if old.stat().st_size == new.stat().st_size and _sha256(old) == _sha256(new):
        report.note = "byte-identical"
    ply_a, ply_b = PlyData.read(str(old)), PlyData.read(str(new))
    if ply_a.text != ply_b.text or ply_a.byte_order != ply_b.byte_order:
        report.structural.append(
            f"format: {'ascii' if ply_a.text else ply_a.byte_order} vs {'ascii' if ply_b.text else ply_b.byte_order}"
        )
    names_a = [el.name for el in ply_a.elements]
    names_b = [el.name for el in ply_b.elements]
    if names_a != names_b:
        report.structural.append(f"elements: {names_a} vs {names_b}")
    for name in [n for n in names_a if n in names_b]:
        el_a, el_b = ply_a[name], ply_b[name]
        if el_a.count != el_b.count:
            report.structural.append(f"{name}: count {el_a.count} vs {el_b.count}")
            continue
        props_a = [(p.name, getattr(p, "val_dtype", None)) for p in el_a.properties]
        props_b = [(p.name, getattr(p, "val_dtype", None)) for p in el_b.properties]
        if props_a != props_b:
            report.structural.append(f"{name}: properties {props_a} vs {props_b}")
        for prop in [p for p, _ in props_a if p in {q for q, _ in props_b}]:
            report.add_numeric(f"{name}.{prop}", _column_diff(el_a[prop], el_b[prop]), el_a.count)
    return report


# ------------------------------------------------------------------ CLI
def compare(old: Path, new: Path, ignore: tuple[str, ...]) -> Report:
    if old.suffix != new.suffix:
        raise SystemExit(f"cannot compare {old} with {new}: different file types")
    if old.suffix == ".json":
        return compare_json(old, new, ignore)
    if old.suffix == ".ply":
        return compare_ply(old, new)
    raise SystemExit(f"unsupported file type: {old}")


def print_report(report: Report, tol: float, top: int) -> None:
    verdict = report.verdict(tol)
    extra = f" ({report.note})" if report.note else ""
    print(f"[{verdict}] {report.kind} max_abs={report.max_abs:.3e}{extra}")
    print(f"  old: {report.old}")
    print(f"  new: {report.new}")
    leaves = sum(int(v["count"]) for v in report.numeric.values())
    print(f"  numeric keys={len(report.numeric)} values={leaves}  ignored={report.ignored or '-'}")
    differing = sorted(
        ((k, v) for k, v in report.numeric.items() if v["max_abs"] > 0.0),
        key=lambda item: -item[1]["max_abs"],
    )
    for key, value in differing[:top]:
        print(f"    {value['max_abs']:.3e}  {key}  (n={value['count']})")
    if len(differing) > top:
        print(f"    ... {len(differing) - top} more differing keys")
    for line in report.structural[:top]:
        print(f"  structural: {line}")
    for line in report.mismatches[:top]:
        print(f"  value: {line}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="+", type=Path, help="OLD NEW pairs")
    parser.add_argument("--tol", type=float, default=1e-9, help="max abs diff still counted as a match")
    parser.add_argument("--ignore", action="append", default=[], help="extra key-path pattern to skip")
    parser.add_argument("--no-default-ignore", action="store_true", help="also compare source paths")
    parser.add_argument("--top", type=int, default=25, help="differing keys to print per pair")
    parser.add_argument("--out", type=Path, default=None, help="write all reports as JSON here")
    args = parser.parse_args(argv)
    if len(args.paths) % 2:
        parser.error("paths must come in OLD NEW pairs")
    ignore = tuple(args.ignore) + (() if args.no_default_ignore else DEFAULT_IGNORE)

    reports = []
    for old, new in zip(args.paths[0::2], args.paths[1::2]):
        report = compare(old, new, ignore)
        print_report(report, args.tol, args.top)
        reports.append(report)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(
                [{**vars(r), "max_abs": r.max_abs, "verdict": r.verdict(args.tol)} for r in reports],
                indent=2,
            )
            + "\n"
        )
    return 0 if all(r.verdict(args.tol) != "DIFFERS" for r in reports) else 1


if __name__ == "__main__":
    sys.exit(main())
