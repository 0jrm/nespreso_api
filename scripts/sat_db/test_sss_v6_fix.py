#!/usr/bin/env python3
"""Unit tests for SSS V6 rename helpers, archive abandon logic, and SSS path candidates."""
from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[2]
SAT_DB = Path(__file__).resolve().parent
sys.path.insert(0, str(SAT_DB))
sys.path.insert(0, str(ROOT / "scripts"))


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestSssCandidates(unittest.TestCase):
    def test_candidate_order(self):
        usa = _load("update_sat_archive", SAT_DB / "update_sat_archive.py")
        d = datetime(2026, 8, 1, tzinfo=timezone.utc)
        paths = usa._sss_l3_candidates("/tmp/sss", d)
        self.assertTrue(paths[0].endswith("_FNL_v06.0.nc"))
        self.assertTrue(paths[1].endswith("_FNL_v05.0.nc"))
        self.assertTrue(paths[2].endswith("_FNL_v06.0_l2c.nc"))
        self.assertIn("/2026/", paths[0])
        self.assertIn("_213_", paths[0])  # Aug 1 2026 = DOY 213


class TestArchiveAbandon(unittest.TestCase):
    def test_abandoned_skipped(self):
        bna = _load("build_nespreso_archive", ROOT / "scripts" / "build_nespreso_archive.py")
        cfg = bna.ArchiveConfig(
            output_dir=tempfile.mkdtemp(),
            checkpoint_file=os.path.join(tempfile.mkdtemp(), "ckpt.json"),
            retry_attempts=3,
        )
        # Avoid signal handlers / log noise in unit context
        builder = bna.ArchiveBuilder.__new__(bna.ArchiveBuilder)
        builder.config = cfg
        builder.checkpoint_data = {
            "20190708": {
                "date": "20190708",
                "processed": False,
                "output_file": None,
                "retry_count": 10,
                "abandoned": True,
                "abandon_reason": "SMAP safehold 2019 (no SSS L3)",
                "error": "safehold",
            }
        }
        builder.running = True
        ok = bna.ArchiveBuilder._process_date(builder, "20190708", force=False)
        self.assertFalse(ok)

    def test_retry_cap_sets_abandoned(self):
        bna = _load("build_nespreso_archive", ROOT / "scripts" / "build_nespreso_archive.py")
        cfg = bna.ArchiveConfig(
            output_dir=tempfile.mkdtemp(),
            checkpoint_file=os.path.join(tempfile.mkdtemp(), "ckpt.json"),
            retry_attempts=3,
        )
        builder = bna.ArchiveBuilder.__new__(bna.ArchiveBuilder)
        builder.config = cfg
        builder.running = True
        builder.checkpoint_data = {
            "20200101": {
                "date": "20200101",
                "processed": False,
                "output_file": None,
                "retry_count": 3,
                "abandoned": False,
                "error": "boom",
            }
        }
        ok = bna.ArchiveBuilder._process_date(builder, "20200101", force=False)
        self.assertFalse(ok)
        self.assertTrue(builder.checkpoint_data["20200101"]["abandoned"])


class TestRenameHelpers(unittest.TestCase):
    def test_target_name(self):
        fix = _load("fix_v6_sss_filenames", SAT_DB / "fix_v6_sss_filenames.py")
        self.assertEqual(
            fix._target_name(2026, 177),
            "RSS_smap_SSS_L3_8day_running_2026_177_FNL_v06.0.nc",
        )


class TestL2cPath(unittest.TestCase):
    def test_l2c_output_path(self):
        agg = _load("sss_l2c_aggregate", SAT_DB / "sss_l2c_aggregate.py")
        p = agg.l2c_output_path("/data/sss", datetime(2026, 8, 1))
        self.assertTrue(str(p).endswith("_FNL_v06.0_l2c.nc"))
        self.assertIn("2026_213", str(p))


if __name__ == "__main__":
    unittest.main()
