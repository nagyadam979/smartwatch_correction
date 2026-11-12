import hashlib
import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from prepare_tour_data import _decode_hr, decode_int24_le


ROOT = Path(__file__).resolve().parent
RAW = ROOT / "09_05"
PROCESSED = RAW / "processed"


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class DecoderTests(unittest.TestCase):
    def test_signed_24_bit_boundaries(self):
        raw = np.array([
            [0x00, 0x00, 0x00], [0xFF, 0xFF, 0x7F],
            [0x00, 0x00, 0x80], [0xFF, 0xFF, 0xFF],
        ], dtype=np.uint8)
        np.testing.assert_array_equal(decode_int24_le(raw), [0, 8388607, -8388608, -1])

    def test_huawei_signed_byte_hr(self):
        decoded = _decode_hr(pd.Series([-1, -127, -100, -90, 0, 157]))
        self.assertTrue(pd.isna(decoded.iloc[0]))
        np.testing.assert_array_equal(decoded.iloc[1:].to_numpy(dtype=float), [129, 156, 166, 0, 157])


@unittest.skipUnless((PROCESSED / "manifest.json").is_file(), "A feldolgozott fixture nem található.")
class ProcessedBundleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = json.loads((PROCESSED / "manifest.json").read_text(encoding="utf-8"))

    def test_raw_source_checksums_still_match_manifest(self):
        for source in self.manifest["sources"]:
            self.assertEqual(file_hash(RAW / source["file"]), source["sha256"])

    def test_native_bdf_row_counts(self):
        self.assertEqual(pq.read_metadata(PROCESSED / "bdf" / "ecg.parquet").num_rows, 23_878_800)
        self.assertEqual(pq.read_metadata(PROCESSED / "bdf" / "accelerometer.parquet").num_rows, 1_492_425)
        self.assertEqual(pq.read_metadata(PROCESSED / "bdf" / "activity.parquet").num_rows, 59_697)

    def test_huawei_fused_view_is_unique_and_decoded(self):
        frame = pd.read_parquet(PROCESSED / "gadgetbridge" / "huawei_activity_fused_1min.parquet")
        self.assertEqual(len(frame), 3372)
        self.assertFalse(frame["TIMESTAMP"].duplicated().any())
        self.assertTrue((frame["SOURCE"] == 11).all())
        self.assertTrue((frame["TIMESTAMP"].diff().dropna() >= 0).all())
        self.assertEqual(float(frame.loc[frame["HEART_RATE"] == -90, "heart_rate_bpm"].iloc[0]), 166.0)

    def test_gpx_is_normalized_and_crosschecked_only(self):
        route = pd.read_parquet(PROCESSED / "gpx" / "track_points.parquet")
        self.assertEqual(len(route), 8761)
        self.assertTrue(route["timestamp_utc"].is_monotonic_increasing)
        self.assertAlmostEqual(float(route["delta_distance_m"].sum()) / 1000, 52.183, places=3)

    def test_fit_is_primary_and_has_no_bdf_overlap(self):
        fit = self.manifest["fit"]
        self.assertTrue(fit["is_fit"])
        self.assertTrue(fit["crc_and_size_integrity"])
        self.assertEqual(fit["unified_record_count"], 9688)
        self.assertFalse(fit["heart_rate_present"])
        self.assertFalse(fit["cadence_present"])
        self.assertFalse(fit["power_present"])
        self.assertAlmostEqual(fit["session_summary"]["total_distance"], 51900.97, places=2)

        result = json.loads((RAW / "analysis" / "analysis.json").read_text(encoding="utf-8"))
        self.assertEqual(result["fit_primary"]["start_local"], "2026-09-05T11:10:19+02:00")
        self.assertAlmostEqual(result["fit_primary"]["distance_km"], 51.90097, places=5)
        self.assertEqual(result["fit_bdf_overlap_minutes"], 0)
        self.assertTrue(result["fit_payload_available"])


if __name__ == "__main__":
    unittest.main()
