"""Characterization tests for the doctor module's pure helpers.

They pin current behaviour with synthetic inputs only: no real binaries or
packages are probed beyond names that cannot exist.
"""

import unittest
from unittest import mock

from cluster_mlip import doctor
from cluster_mlip.doctor import (
    MISSING_OPTIONAL,
    MISSING_REQUIRED,
    OK,
    DoctorCheck,
    format_report,
)


class BinaryAndModuleChecks(unittest.TestCase):
    def test_binary_found_reports_first_hit(self):
        with mock.patch.object(doctor.shutil, "which", side_effect=[None, "/opt/g09"]):
            check = doctor._binary_check("gaussian", ["g16", "g09"], "jobs")
        self.assertEqual(check.status, OK)
        self.assertEqual(check.detail, "'g09' found at /opt/g09")

    def test_binary_missing_is_optional(self):
        with mock.patch.object(doctor.shutil, "which", return_value=None):
            check = doctor._binary_check("gaussian", ["g16", "g09"], "jobs")
        self.assertEqual(check.status, MISSING_OPTIONAL)
        self.assertEqual(check.detail, "none of 'g16/g09' on PATH; needed for jobs")

    def test_module_check_present_and_absent(self):
        self.assertEqual(doctor._module_check("json", "json", "x").status, OK)
        missing = doctor._module_check("nope", "definitely_not_a_module_xyz", "feature")
        self.assertEqual(missing.status, MISSING_OPTIONAL)
        self.assertIn("needed for feature", missing.detail)

    def test_module_check_value_error_counts_as_missing(self):
        with mock.patch.object(doctor.importlib.util, "find_spec", side_effect=ValueError):
            self.assertEqual(doctor._module_check("m", "m", "x").status, MISSING_OPTIONAL)


class RunChecks(unittest.TestCase):
    def test_old_python_is_required_failure(self):
        fake = mock.Mock(major=3, minor=9)
        with mock.patch.object(doctor.sys, "version_info", fake):
            checks = doctor.run_checks()
        self.assertEqual(checks[0].name, "python")
        self.assertEqual(checks[0].status, MISSING_REQUIRED)
        self.assertIn("below the required 3.10", checks[0].detail)

    def test_check_names(self):
        names = [c.name for c in doctor.run_checks()]
        self.assertEqual(
            names, ["python", "strings", "formchk", "gaussian", "mace-torch", "torch"]
        )


class FormatReport(unittest.TestCase):
    def test_all_ok(self):
        text, worst = format_report([DoctorCheck("a", OK, "fine")])
        self.assertEqual(worst, OK)
        self.assertIn("[OK  ] a  fine", text)
        self.assertTrue(text.endswith("All checks passed."))

    def test_worst_status_wins_and_names_are_padded(self):
        checks = [
            DoctorCheck("a", MISSING_OPTIONAL, "x"),
            DoctorCheck("long", MISSING_REQUIRED, "y"),
            DoctorCheck("b", OK, "z"),
        ]
        text, worst = format_report(checks)
        self.assertEqual(worst, MISSING_REQUIRED)
        self.assertIn("[WARN] a     x", text)
        self.assertIn("[FAIL] long  y", text)
        self.assertTrue(text.endswith("fix it before running the pipeline."))

    def test_optional_only_message_and_empty_list(self):
        text, worst = format_report([DoctorCheck("a", MISSING_OPTIONAL, "x")])
        self.assertEqual(worst, MISSING_OPTIONAL)
        self.assertIn("Optional tools are missing", text)
        _, worst_empty = format_report([])
        self.assertEqual(worst_empty, OK)


if __name__ == "__main__":
    unittest.main()
