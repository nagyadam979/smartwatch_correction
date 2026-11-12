import unittest
import sqlite3
import tempfile
import zipfile
from pathlib import Path

import numpy as np

from analyze_hr import (
    contiguous_intervals,
    decode_int24_le,
    detect_qrs,
    estimate_alignment,
    estimate_offset,
    physical_scale,
    select_calibration,
    load_watch_data,
)


class BDFDecodeTests(unittest.TestCase):
    def test_signed_24_bit_decode_and_physical_scale(self):
        raw = np.array(
            [
                [0x00, 0x00, 0x00],
                [0xFF, 0xFF, 0x7F],
                [0x00, 0x00, 0x80],
                [0xFF, 0xFF, 0xFF],
            ],
            dtype=np.uint8,
        )
        decoded = decode_int24_le(raw)
        np.testing.assert_array_equal(decoded, [0, 8388607, -8388608, -1])
        scaled = physical_scale(np.array([0, 50, 100]), -1, 1, 0, 100)
        np.testing.assert_allclose(scaled, [-1, 0, 1])


class IntervalTests(unittest.TestCase):
    def test_gap_segmentation(self):
        intervals, threshold = contiguous_intervals([0.0, 1.0, 2.0, 8.1, 9.0, 20.0])
        self.assertEqual(threshold, 5.0)
        self.assertEqual(intervals, [(0.0, 2.0), (8.1, 9.0), (20.0, 20.0)])


class WatchZipTests(unittest.TestCase):
    @staticmethod
    def database_bytes(heart_rates):
        connection = sqlite3.connect(":memory:")
        connection.execute(
            "CREATE TABLE GENERIC_HEART_RATE_SAMPLE "
            "(TIMESTAMP INTEGER NOT NULL, HEART_RATE INTEGER NOT NULL)"
        )
        connection.executemany(
            "INSERT INTO GENERIC_HEART_RATE_SAMPLE VALUES (?, ?)",
            [(1788280000000 + i * 1000, value) for i, value in enumerate(heart_rates)],
        )
        database = connection.serialize()
        connection.close()
        return database

    def test_canonical_zip_database_is_loaded_without_extraction(self):
        canonical = self.database_bytes([70, 71])
        backup = self.database_bytes([150])
        with tempfile.TemporaryDirectory() as directory:
            archive_path = Path(directory) / "export.zip"
            with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
                archive.writestr("files/Gadgetbridge_backup", backup)
                archive.writestr("database/Gadgetbridge", canonical)
            watch = load_watch_data(archive_path)
            self.assertEqual(watch.archive_member, "database/Gadgetbridge")
            self.assertEqual(watch.source_row_count, 2)
            np.testing.assert_array_equal(watch.heart_rate, [70, 71])
            self.assertEqual(
                sorted(path.name for path in Path(directory).iterdir()), ["export.zip"]
            )

    def test_direct_sqlite_file_remains_supported(self):
        with tempfile.TemporaryDirectory() as directory:
            database_path = Path(directory) / "Gadgetbridge"
            database_path.write_bytes(self.database_bytes([72, 73]))
            watch = load_watch_data(database_path)
            self.assertIsNone(watch.archive_member)
            np.testing.assert_array_equal(watch.heart_rate, [72, 73])

    def test_unique_header_confirmed_fallback_is_loaded(self):
        database = self.database_bytes([80, 81, 82])
        with tempfile.TemporaryDirectory() as directory:
            archive_path = Path(directory) / "fallback.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("metadata.json", "{}")
                archive.writestr("export/current.sqlite", database)
            watch = load_watch_data(archive_path)
            self.assertEqual(watch.archive_member, "export/current.sqlite")
            np.testing.assert_array_equal(watch.heart_rate, [80, 81, 82])

    def test_ambiguous_sqlite_fallback_is_rejected(self):
        database = self.database_bytes([80])
        with tempfile.TemporaryDirectory() as directory:
            archive_path = Path(directory) / "ambiguous.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("backup/a.sqlite", database)
                archive.writestr("backup/b.sqlite", database)
            with self.assertRaisesRegex(ValueError, "nem egyértelmű"):
                load_watch_data(archive_path)


class AlignmentTests(unittest.TestCase):
    def test_known_lookup_offset_is_recovered(self):
        timestamp = np.arange(0.0, 900.0, 1.0)

        def truth(t):
            t = np.asarray(t)
            return 75.0 + 9.0 * np.sin(t / 31.0) + 3.0 * np.sin(t / 7.3)

        expected_offset = -7.4
        watch = truth(timestamp + expected_offset)
        recovered, score, zero_score = estimate_offset(
            timestamp, watch, truth, max_lag_seconds=15.0
        )
        self.assertAlmostEqual(recovered, expected_offset, delta=0.11)
        self.assertLess(score, zero_score)

    def test_known_offset_and_drift_are_recovered(self):
        timestamp = np.arange(0.0, 7200.0, 1.0)
        center = float(np.median(timestamp))
        expected_offset = -5.6
        expected_drift = 4.0 / np.ptp(timestamp)

        def truth(t):
            t = np.asarray(t)
            return 78.0 + 11.0 * np.sin(t / 29.0) + 4.0 * np.sin(t / 8.1)

        class SyntheticReference:
            @staticmethod
            def evaluate(query):
                query = np.asarray(query)
                return truth(query), np.ones(query.shape)

        watch = truth(
            timestamp
            + expected_offset
            + expected_drift * (timestamp - center)
        )
        alignment = estimate_alignment(timestamp, watch, SyntheticReference(), 15.0)
        self.assertTrue(alignment.drift_applied)
        self.assertAlmostEqual(alignment.offset_seconds, expected_offset, delta=0.25)
        self.assertAlmostEqual(
            alignment.drift_seconds_per_second, expected_drift, delta=8e-5
        )


class QRSQualityTests(unittest.TestCase):
    def test_baseline_drift_and_transient_artifact_preserve_clean_beats(self):
        sample_rate = 200.0
        duration = 90.0
        time = np.arange(int(sample_rate * duration)) / sample_rate
        ecg = 1.8 * np.sin(2 * np.pi * 0.18 * time)
        ecg += 0.1 * np.sin(2 * np.pi * 0.04 * time)
        for beat in np.arange(0.8, duration - 0.3, 0.75):
            ecg += 1.2 * np.exp(-0.5 * ((time - beat) / 0.018) ** 2)
            ecg -= 0.25 * np.exp(-0.5 * ((time - (beat - 0.035)) / 0.014) ** 2)
        ecg += 0.03 * np.sin(2 * np.pi * 17.0 * time)
        ecg += 3.0 * np.exp(-0.5 * ((time - 44.0) / 0.04) ** 2)

        reference = detect_qrs(ecg, sample_rate, 0.0)
        usable_blocks = [block for block in reference.blocks if block.usable]
        self.assertEqual(len(usable_blocks), 3)
        self.assertGreaterEqual(sum(len(block.peaks) for block in usable_blocks), 115)
        self.assertTrue(
            all(block.morphology_consistency >= 0.80 for block in usable_blocks)
        )

    def test_flatline_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "minőségkapun"):
            detect_qrs(np.zeros(12000), 100.0, 0.0)

    def test_sustained_high_frequency_artifact_rejects_only_local_block(self):
        sample_rate = 200.0
        duration = 90.0
        time = np.arange(int(sample_rate * duration)) / sample_rate
        ecg = 0.4 * np.sin(2 * np.pi * 0.2 * time)
        for beat in np.arange(0.8, duration - 0.3, 0.75):
            ecg += 1.2 * np.exp(-0.5 * ((time - beat) / 0.018) ** 2)
            ecg -= 0.25 * np.exp(-0.5 * ((time - (beat - 0.035)) / 0.014) ** 2)
        contaminated = (time >= 35.0) & (time <= 55.0)
        ecg[contaminated] += 1.2 * np.sin(2 * np.pi * 11.0 * time[contaminated])
        ecg[contaminated] += 0.5 * np.sin(2 * np.pi * 19.0 * time[contaminated])

        reference = detect_qrs(ecg, sample_rate, 0.0)
        self.assertEqual([block.usable for block in reference.blocks], [True, False, True])
        self.assertIn("off_QRS_artifact", reference.blocks[1].reason)


class CalibrationTests(unittest.TestCase):
    def test_holdout_selects_a_supported_correction(self):
        timestamp = np.arange(6000.0)
        watch = 75.0 + 12.0 * np.sin(timestamp / 41.0)
        reference = watch + 4.0
        calibration = select_calibration(timestamp, watch, reference)
        self.assertIn(calibration.name, {"bias", "affine"})
        self.assertAlmostEqual(calibration.slope, 1.0, delta=0.01)
        self.assertAlmostEqual(calibration.intercept, 4.0, delta=0.05)
        self.assertLess(
            calibration.candidates[calibration.name]["mae"],
            calibration.candidates["identity"]["mae"],
        )
        self.assertGreaterEqual(
            calibration.candidates[calibration.name]["block_count"], 5
        )
        self.assertGreaterEqual(
            calibration.candidates[calibration.name]["block_win_rate"], 0.70
        )

    def test_too_few_holdout_blocks_falls_back_to_identity(self):
        timestamp = np.arange(1000.0)
        watch = 75.0 + 12.0 * np.sin(timestamp / 41.0)
        reference = watch + 4.0
        calibration = select_calibration(timestamp, watch, reference)
        self.assertEqual(calibration.name, "identity")

    def test_split_is_strictly_after_alignment_training_cutoff(self):
        timestamp = np.arange(1200.0)
        watch = 75.0 + 12.0 * np.sin(timestamp / 41.0)
        reference = watch + 1.0
        alignment_cutoff = 800.0
        calibration = select_calibration(
            timestamp,
            watch,
            reference,
            train_fraction=0.50,
            minimum_split_timestamp=alignment_cutoff,
        )
        self.assertGreater(calibration.split_timestamp, alignment_cutoff)
        self.assertEqual(
            calibration.split_timestamp,
            np.nextafter(alignment_cutoff, np.inf),
        )


if __name__ == "__main__":
    unittest.main()
