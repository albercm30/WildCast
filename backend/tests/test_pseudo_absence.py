"""Unit tests for app.ml.pseudo_absence -- pure pandas logic, no network."""
import datetime as dt
import unittest

import pandas as pd

from app.ml.pseudo_absence import build_labeled_dataset, generate_pseudo_absences


def _occurrences():
    """
    Two mammal species in one area (bear: summer-heavy; wolf: spread across
    the year) plus one bird in the same area, to check taxon-class grouping.
    """
    rows = []
    for i, month in enumerate([5, 6, 6, 7, 8, 8, 9]):  # bear: clustered in summer
        rows.append({"species": "Ursus arctos", "taxon_class": "Mammalia", "area_id": "yellowstone",
                      "lat": 44.6 + i * 0.001, "lon": -110.5, "date": dt.date(2023, month, 10 + i)})
    for i, month in enumerate([1, 3, 5, 7, 9, 11]):  # wolf: spread across the year
        rows.append({"species": "Canis lupus", "taxon_class": "Mammalia", "area_id": "yellowstone",
                      "lat": 44.6 + i * 0.001, "lon": -110.5, "date": dt.date(2023, month, 15)})
    for i, month in enumerate([2, 4, 6, 8]):  # eagle: a different taxon class, same area
        rows.append({"species": "Haliaeetus leucocephalus", "taxon_class": "Aves", "area_id": "yellowstone",
                      "lat": 44.6 + i * 0.001, "lon": -110.5, "date": dt.date(2023, month, 20)})
    return pd.DataFrame(rows)


class GeneratePseudoAbsencesTests(unittest.TestCase):
    def test_pseudo_absences_are_labeled_for_the_target_species(self):
        occ = _occurrences()
        bg = generate_pseudo_absences(occ, ratio=1.0, min_gap_days=1)
        bear_bg = bg[bg["species"] == "Ursus arctos"]
        self.assertGreater(len(bear_bg), 0)
        self.assertTrue((bear_bg["label"] == 0).all())

    def test_bear_background_dates_come_from_other_species_not_itself(self):
        # Target-group background sampling must draw dates from OTHER
        # species' records, never manufacture a date bear wasn't near.
        occ = _occurrences()
        bg = generate_pseudo_absences(occ, ratio=2.0, min_gap_days=1)
        bear_bg_dates = set(pd.to_datetime(bg[bg["species"] == "Ursus arctos"]["date"]).dt.date)
        wolf_dates = set(occ[occ["species"] == "Canis lupus"]["date"])
        self.assertTrue(bear_bg_dates.issubset(wolf_dates))

    def test_background_never_reuses_a_taxon_class_species_has_no_relation_to(self):
        # Bird (Aves) records must never become mammal pseudo-absence background.
        occ = _occurrences()
        bg = generate_pseudo_absences(occ, ratio=2.0, min_gap_days=1)
        bear_bg_dates = set(pd.to_datetime(bg[bg["species"] == "Ursus arctos"]["date"]).dt.date)
        eagle_dates = set(occ[occ["species"] == "Haliaeetus leucocephalus"]["date"])
        self.assertEqual(bear_bg_dates & eagle_dates, set())

    def test_min_gap_days_excludes_near_positive_dates(self):
        # A background candidate within `min_gap_days` of one of the
        # species' OWN real sightings must never be used (would mislabel a
        # near-certain presence as an absence).
        occ = _occurrences()
        bear_dates = set(occ[occ["species"] == "Ursus arctos"]["date"])
        bg = generate_pseudo_absences(occ, ratio=3.0, min_gap_days=5)
        bear_bg_dates = pd.to_datetime(bg[bg["species"] == "Ursus arctos"]["date"]).dt.date
        for bg_date in bear_bg_dates:
            for real_date in bear_dates:
                self.assertGreaterEqual(abs((bg_date - real_date).days), 5)


class BuildLabeledDatasetTests(unittest.TestCase):
    def test_combines_positives_and_negatives(self):
        occ = _occurrences()
        dataset = build_labeled_dataset(occ, ratio=1.0, min_gap_days=1)
        self.assertEqual((dataset["label"] == 1).sum(), len(occ))
        self.assertGreater((dataset["label"] == 0).sum(), 0)

    def test_positives_are_unlabeled_input_rows_unchanged(self):
        occ = _occurrences()
        dataset = build_labeled_dataset(occ, ratio=1.0, min_gap_days=1)
        positives = dataset[dataset["label"] == 1]
        self.assertEqual(set(positives["species"]), set(occ["species"]))


if __name__ == "__main__":
    unittest.main()
