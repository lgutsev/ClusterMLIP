import unittest

from cluster_mlip.models import Atom, Record
from cluster_mlip.stratify import (
    charge_spin_class,
    classify_record,
    compactness_class,
    coordination_class,
    displacement_class,
    pes_region,
    provenance_tier,
    strata_value,
    stratum_key,
)


class BondingGraphTests(unittest.TestCase):
    def test_bonded_pair_is_low_coordination_and_compact(self):
        atoms = [Atom("Fe", 0, 0, 0), Atom("O", 1.6, 0, 0)]  # well within Fe-O bonding distance
        self.assertEqual(coordination_class(atoms), "low_coordination")
        self.assertEqual(compactness_class(atoms), "compact")

    def test_distant_pair_is_fragmenting(self):
        atoms = [Atom("Fe", 0, 0, 0), Atom("Fe", 10.0, 0, 0)]  # far outside any bonding cutoff
        self.assertEqual(compactness_class(atoms), "fragmenting")

    def test_well_coordinated_needs_every_atom_well_bonded(self):
        # coordination_class reports the *worst* atom's degree, so a "star"
        # (one well-bonded center, four singly-bonded outer atoms) is still
        # low_coordination overall -- that's the point, it flags the frame
        # containing an under-coordinated atom regardless of a well-bonded
        # core. Getting "well_coordinated" needs every atom well-bonded, e.g.
        # a compact octahedron of one element where even the two longer
        # "diagonal" distances still fall inside the bonding cutoff.
        star = [
            Atom("Fe", 0, 0, 0),
            Atom("O", 1.9, 0, 0),
            Atom("O", -1.9, 0, 0),
            Atom("O", 0, 1.9, 0),
            Atom("O", 0, -1.9, 0),
        ]
        self.assertEqual(coordination_class(star), "low_coordination")

        octahedron = [
            Atom("Fe", 1.3, 0, 0), Atom("Fe", -1.3, 0, 0),
            Atom("Fe", 0, 1.3, 0), Atom("Fe", 0, -1.3, 0),
            Atom("Fe", 0, 0, 1.3), Atom("Fe", 0, 0, -1.3),
        ]
        self.assertEqual(coordination_class(octahedron), "well_coordinated")
        self.assertEqual(compactness_class(octahedron), "compact")

    def test_unknown_element_is_unclassified_not_guessed(self):
        atoms = [Atom("Xx", 0, 0, 0), Atom("O", 1.5, 0, 0)]
        self.assertEqual(coordination_class(atoms), "unclassified")
        self.assertEqual(compactness_class(atoms), "unclassified")

    def test_empty_atom_list_is_unclassified(self):
        self.assertEqual(coordination_class([]), "unclassified")
        self.assertEqual(compactness_class([]), "unclassified")


class ConfigTypeAxisTests(unittest.TestCase):
    def test_pes_region_strips_rattled_suffix_first(self):
        self.assertEqual(pes_region("minimum"), "minimum")
        self.assertEqual(pes_region("minimum_rattled"), "minimum")
        self.assertEqual(pes_region("first_order_saddle_rattled"), "saddle")
        self.assertEqual(pes_region("transition_state"), "saddle")
        self.assertEqual(pes_region("irc_forward"), "irc")
        self.assertEqual(pes_region("warehouse_structure"), "other")

    def test_displacement_class(self):
        self.assertEqual(displacement_class("minimum"), "relaxed")
        self.assertEqual(displacement_class("minimum_rattled"), "rattled")


class ChargeSpinClassTests(unittest.TestCase):
    def test_buckets(self):
        self.assertEqual(charge_spin_class(0, 1), "neutral_closed_shell")
        self.assertEqual(charge_spin_class(0, 3), "neutral_open_shell_low")
        self.assertEqual(charge_spin_class(0, 5), "neutral_open_shell_high")
        self.assertEqual(charge_spin_class(0, 29), "neutral_open_shell_high")
        self.assertEqual(charge_spin_class(-1, 2), "charged")
        self.assertEqual(charge_spin_class(1, 1), "charged")

    def test_threshold_is_configurable(self):
        self.assertEqual(charge_spin_class(0, 5, high_spin_threshold=9), "neutral_open_shell_low")


def _record_with_state_inference(tag: str) -> Record:
    return Record(
        "r", "s", [Atom("Fe", 0, 0, 0)], 0, 1, "minimum",
        metadata={"state_inference": tag} if tag else {},
    )


class ProvenanceTierTests(unittest.TestCase):
    def test_tiers(self):
        self.assertEqual(provenance_tier(_record_with_state_inference("")), "validated")
        self.assertEqual(provenance_tier(_record_with_state_inference("filename")), "filename_derived")
        self.assertEqual(
            provenance_tier(_record_with_state_inference("default_unmatched_singlet")), "fallback_guess"
        )
        self.assertEqual(
            provenance_tier(_record_with_state_inference("electron_parity_fallback")), "fallback_guess"
        )


class ClassifyRecordTests(unittest.TestCase):
    def test_classify_record_bundles_all_axes(self):
        record = Record(
            "r1", "s", [Atom("Fe", 0, 0, 0), Atom("O", 1.6, 0, 0)], 0, 3, "minimum_rattled",
        )
        strata = classify_record(record)
        self.assertEqual(strata["pes_region"], "minimum")
        self.assertEqual(strata["displacement_class"], "rattled")
        self.assertEqual(strata["coordination_class"], "low_coordination")
        self.assertEqual(strata["compactness_class"], "compact")
        self.assertEqual(strata["charge_spin_class"], "neutral_open_shell_low")
        self.assertEqual(strata["provenance_tier"], "validated")

    def test_stratum_key_joins_requested_fields_only(self):
        record = Record("r1", "s", [Atom("Fe", 0, 0, 0), Atom("O", 1.6, 0, 0)], 0, 1, "minimum")
        strata = classify_record(record)
        self.assertEqual(strata_value(strata, "pes_region"), "minimum")
        self.assertEqual(
            stratum_key(strata, ("pes_region", "charge_spin_class")),
            "minimum|neutral_closed_shell",
        )
        self.assertEqual(stratum_key(strata, ()), "")


if __name__ == "__main__":
    unittest.main()
