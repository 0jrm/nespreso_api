#!/usr/bin/env python3
"""
Fix misnamed SMAP SSS L3 V6 files so filename DOY matches content center time.

Default is dry-run. Use --apply to perform renames.
Colliding targets: keep an already-correct file; quarantine duplicates.
Cross-year moves are allowed (e.g. 2024_001 with 2023-12-20 content -> 2023/DOY354).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

SSS_ROOT_DEFAULT = Path(
    os.environ.get("SATDB_SSS_ROOT", "/Net/work/ozavala/DATA/GOFFISH/SSS/SMAP_Global/")
)
V6_RE = re.compile(
    r"^RSS_smap_SSS_L3_8day_running_(\d{4})_(\d{3})_FNL_v06\.0\.nc$"
)


def _content_center(path: Path) -> Tuple[datetime, int, int]:
    """Return (center_datetime, year, doy) from the file's time coordinate."""
    from netCDF4 import Dataset, num2date

    with Dataset(path) as ds:
        tvar = ds.variables["time"]
        # RSS L3 typically has a single time value
        raw = tvar[0] if getattr(tvar, "shape", None) else tvar[:]
        try:
            units = tvar.units
        except AttributeError:
            units = "seconds since 2000-01-01T00:00:00Z"
        calendar = getattr(tvar, "calendar", "standard")
        center_dt = num2date(raw, units=units, calendar=calendar)
        # num2date may return cftime or datetime
        if hasattr(center_dt, "year"):
            center = datetime(center_dt.year, center_dt.month, center_dt.day)
        else:
            center = datetime.strptime(str(center_dt)[:10], "%Y-%m-%d")
    return center, center.year, center.timetuple().tm_yday


def _target_name(year: int, doy: int) -> str:
    return f"RSS_smap_SSS_L3_8day_running_{year}_{doy:03d}_FNL_v06.0.nc"


def _is_correctly_named(path: Path, content_year: int, content_doy: int) -> bool:
    m = V6_RE.match(path.name)
    if not m:
        return False
    return int(m.group(1)) == content_year and int(m.group(2)) == content_doy


def scan_actions(sss_root: Path) -> List[Dict[str, Any]]:
    actions: List[Dict[str, Any]] = []
    # First pass: gather all V6 files with content centers
    entries: List[Dict[str, Any]] = []
    for year_dir in sorted(p for p in sss_root.iterdir() if p.is_dir() and p.name.isdigit()):
        for path in sorted(year_dir.glob("*v06.0.nc")):
            if "_l2c" in path.name:
                continue
            m = V6_RE.match(path.name)
            if not m:
                continue
            try:
                center, cy, cdoy = _content_center(path)
            except Exception as e:
                actions.append(
                    {
                        "path": str(path),
                        "action": "error",
                        "error": str(e),
                    }
                )
                continue
            entries.append(
                {
                    "path": path,
                    "file_year": int(m.group(1)),
                    "file_doy": int(m.group(2)),
                    "content_time": center.strftime("%Y-%m-%d"),
                    "target_year": cy,
                    "target_doy": cdoy,
                    "correct": _is_correctly_named(path, cy, cdoy),
                }
            )

    by_target: Dict[Tuple[int, int], List[Dict[str, Any]]] = defaultdict(list)
    for e in entries:
        by_target[(e["target_year"], e["target_doy"])].append(e)

    quarantine_root = sss_root / "_quarantine_v6_misname"

    for (ty, tdoy), group in sorted(by_target.items()):
        target_dir = sss_root / f"{ty:04d}"
        target_path = target_dir / _target_name(ty, tdoy)
        # Prefer already-correct member
        correct = [g for g in group if g["correct"]]
        incorrect = [g for g in group if not g["correct"]]

        keeper: Optional[Dict[str, Any]] = None
        if correct:
            keeper = correct[0]
            for g in correct[1:]:
                actions.append(_quarantine_action(g, quarantine_root, reason="duplicate_correct"))
            for g in incorrect:
                actions.append(_quarantine_action(g, quarantine_root, reason="duplicate_of_correct"))
        elif len(group) == 1:
            g = group[0]
            if g["correct"]:
                actions.append({**_base(g), "action": "noop"})
            else:
                actions.append(
                    {
                        **_base(g),
                        "action": "rename",
                        "dest": str(target_path),
                    }
                )
        else:
            # Multiple incorrect pointing at same target: keep first, quarantine rest
            keeper = group[0]
            actions.append(
                {
                    **_base(keeper),
                    "action": "rename",
                    "dest": str(target_path),
                }
            )
            for g in group[1:]:
                actions.append(_quarantine_action(g, quarantine_root, reason="collision_duplicate"))

        # If keeper is correct and dest exists as same path, noop already handled
        if keeper and keeper["correct"]:
            actions.append({**_base(keeper), "action": "noop"})

    # Deduplicate noops/renames carefully: rebuild unique by path
    by_path: Dict[str, Dict[str, Any]] = {}
    for a in actions:
        p = a.get("path")
        if not p:
            continue
        prev = by_path.get(p)
        # Prefer quarantine/error/rename over noop
        rank = {"error": 3, "quarantine": 2, "rename": 1, "noop": 0}
        if prev is None or rank.get(a["action"], 0) >= rank.get(prev["action"], 0):
            by_path[p] = a
    return list(by_path.values())


def _base(g: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "path": str(g["path"]),
        "file_year": g["file_year"],
        "file_doy": g["file_doy"],
        "content_time": g["content_time"],
        "target_year": g["target_year"],
        "target_doy": g["target_doy"],
    }


def _quarantine_action(g: Dict[str, Any], quarantine_root: Path, reason: str) -> Dict[str, Any]:
    src = Path(g["path"])
    dest = quarantine_root / src.parent.name / src.name
    return {
        **_base(g),
        "action": "quarantine",
        "dest": str(dest),
        "reason": reason,
    }


def apply_actions(actions: List[Dict[str, Any]], dry_run: bool = True) -> Dict[str, int]:
    counts = defaultdict(int)
    renames = [a for a in actions if a["action"] == "rename"]
    quarantines = [a for a in actions if a["action"] == "quarantine"]

    # Quarantine first so duplicate sources are out of the way
    for a in quarantines:
        src = Path(a["path"])
        dest = Path(a["dest"])
        counts["quarantine_planned"] += 1
        if dry_run:
            counts["quarantine_dry"] += 1
            continue
        if not src.exists():
            counts["missing_src"] += 1
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            dest = dest.with_name(dest.stem + f"__dup_{os.getpid()}" + dest.suffix)
        shutil.move(str(src), str(dest))
        counts["quarantined"] += 1

    # Phase 1: move every rename source to a unique temp name (allows chains)
    temps: List[Tuple[Path, Path, Dict[str, Any]]] = []
    for a in renames:
        src = Path(a["path"])
        dest = Path(a["dest"])
        counts["rename_planned"] += 1
        if not src.exists():
            counts["missing_src"] += 1
            continue
        if src.resolve() == dest.resolve():
            counts["noop"] += 1
            continue
        tmp = src.with_name(src.name + f".repoint_tmp_{os.getpid()}")
        if dry_run:
            counts["rename_dry"] += 1
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        os.rename(src, tmp)
        temps.append((tmp, dest, a))

    # Phase 2: move temps into final destinations
    for tmp, dest, a in temps:
        if dest.exists():
            qdir = Path(a["path"]).parent.parent / "_quarantine_v6_misname" / dest.parent.name
            qdir.mkdir(parents=True, exist_ok=True)
            qpath = qdir / (dest.name + f".collision_{os.getpid()}")
            os.rename(tmp, qpath)
            counts["collision_on_apply"] += 1
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
            os.rename(tmp, dest)
            counts["renamed"] += 1

    counts["noop_or_other"] += sum(1 for a in actions if a["action"] == "noop")
    counts["errors"] += sum(1 for a in actions if a["action"] == "error")
    return dict(counts)


def audit_sample(sss_root: Path, n: int = 20) -> List[str]:
    problems = []
    files = []
    for year_dir in sorted(p for p in sss_root.iterdir() if p.is_dir() and p.name.isdigit()):
        files.extend(sorted(year_dir.glob("RSS_smap_SSS_L3_8day_running_*_FNL_v06.0.nc")))
    if not files:
        return ["no v06 files found"]
    step = max(1, len(files) // n)
    sample = files[::step][:n]
    for path in sample:
        if "_l2c" in path.name:
            continue
        m = V6_RE.match(path.name)
        if not m:
            problems.append(f"{path.name}: bad name")
            continue
        try:
            _, cy, cdoy = _content_center(path)
        except Exception as e:
            problems.append(f"{path.name}: read error {e}")
            continue
        fy, fdoy = int(m.group(1)), int(m.group(2))
        if fy != cy or fdoy != cdoy:
            problems.append(f"{path.name}: file=({fy},{fdoy}) content=({cy},{cdoy})")
    return problems


def main():
    ap = argparse.ArgumentParser(description="Fix V6 SSS filename/content DOY mismatch")
    ap.add_argument("--sss-root", type=Path, default=SSS_ROOT_DEFAULT)
    ap.add_argument("--apply", action="store_true", help="Perform renames (default: dry-run)")
    ap.add_argument("--manifest", type=Path, default=None)
    ap.add_argument(
        "--from-manifest",
        type=Path,
        default=None,
        help="Skip rescan; apply/report actions from an existing manifest JSON",
    )
    ap.add_argument("--audit", action="store_true", help="After apply/dry-run, sample-audit alignment")
    args = ap.parse_args()

    sss_root = args.sss_root
    if args.from_manifest:
        summary = json.loads(Path(args.from_manifest).read_text())
        actions = summary["actions"]
        print(f"Loaded {len(actions)} actions from {args.from_manifest}")
        print("Summary:", summary.get("counts"))
        manifest = Path(args.from_manifest)
    else:
        actions = scan_actions(sss_root)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
        manifest = args.manifest or (
            Path(__file__).resolve().parent / f"rename_manifest_{stamp}.json"
        )
        summary = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "sss_root": str(sss_root),
            "dry_run": not args.apply,
            "counts": {
                "total": len(actions),
                "rename": sum(1 for a in actions if a["action"] == "rename"),
                "quarantine": sum(1 for a in actions if a["action"] == "quarantine"),
                "noop": sum(1 for a in actions if a["action"] == "noop"),
                "error": sum(1 for a in actions if a["action"] == "error"),
            },
            "actions": actions,
        }
        manifest.write_text(json.dumps(summary, indent=2))
        print(f"Wrote manifest {manifest}")
        print("Summary:", summary["counts"])

    result = apply_actions(actions, dry_run=not args.apply)
    print("Apply result:", result)

    if args.audit or args.apply:
        problems = audit_sample(sss_root)
        if problems:
            print("AUDIT PROBLEMS:")
            for p in problems:
                print(" ", p)
        else:
            print("AUDIT OK: sample filenames match content DOY")


if __name__ == "__main__":
    main()
