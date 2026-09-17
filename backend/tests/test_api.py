"""
Tests for the Flask API (app.main), via Flask's own test client -- no
running server, no network, no sockets. Like test_prediction_service.py,
these rely on the model + cache already present in the repo.
"""
import unittest
from unittest.mock import patch

from app.main import create_app
from app.ml import prediction_service as ps
from tests.test_prediction_service import _synthetic_normals


class ApiTestCase(unittest.TestCase):
    def setUp(self):
        self.client = create_app().test_client()


class HealthTests(ApiTestCase):
    def test_health_reports_model_loaded_and_all_areas(self):
        resp = self.client.get("/api/health")
        self.assertEqual(resp.status_code, 200)
        body = resp.get_json()
        self.assertEqual(body["status"], "ok")
        self.assertTrue(body["model_loaded"])
        self.assertEqual(
            set(body["areas"]),
            {
                "yellowstone", "kruger", "brisbane_urban", "svalbard_arctic", "amazon_rainforest",
                "serengeti", "borneo_rainforest", "patagonia", "scottish_highlands", "costa_rica_cloud_forest",
            },
        )
        # Real vs. synthetic-demo data labeling must always be present and
        # never silently missing -- see app.services.data_source_marker.
        self.assertIn("data_source", body)
        self.assertIn(body["data_source"].get("source"), ("real", "synthetic_demo", "unknown"))

    def test_cors_header_present(self):
        resp = self.client.get("/api/health")
        self.assertEqual(resp.headers.get("Access-Control-Allow-Origin"), "*")


class AreasTests(ApiTestCase):
    def test_list_areas(self):
        resp = self.client.get("/api/areas")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(len(resp.get_json()), 10)

    def test_get_one_area(self):
        resp = self.client.get("/api/areas/yellowstone")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.get_json()["name"], "Yellowstone National Park, USA")

    def test_unknown_area_is_404(self):
        resp = self.client.get("/api/areas/atlantis")
        self.assertEqual(resp.status_code, 404)
        self.assertIn("error", resp.get_json())

    def test_area_species(self):
        resp = self.client.get("/api/areas/kruger/species")
        self.assertEqual(resp.status_code, 200)
        names = {s["scientific_name"] for s in resp.get_json()}
        self.assertIn("Panthera leo", names)


class PredictEndpointTests(ApiTestCase):
    def test_valid_request(self):
        resp = self.client.get("/api/predict?area_id=yellowstone&date=2027-06-15")
        self.assertEqual(resp.status_code, 200)
        body = resp.get_json()
        self.assertEqual(body["area_id"], "yellowstone")
        self.assertGreater(len(body["predictions"]), 0)
        self.assertIn("data_source", body)

    def test_missing_params_is_400(self):
        resp = self.client.get("/api/predict?area_id=yellowstone")
        self.assertEqual(resp.status_code, 400)

    def test_bad_date_format_is_400(self):
        resp = self.client.get("/api/predict?area_id=yellowstone&date=15-06-2027")
        self.assertEqual(resp.status_code, 400)

    def test_unknown_area_is_404(self):
        resp = self.client.get("/api/predict?area_id=atlantis&date=2027-06-15")
        self.assertEqual(resp.status_code, 404)

    def test_species_filter(self):
        resp = self.client.get("/api/predict?area_id=yellowstone&date=2027-06-15&species=Canis%20lupus")
        body = resp.get_json()
        self.assertEqual(len(body["predictions"]), 1)
        self.assertEqual(body["predictions"][0]["scientific_name"], "Canis lupus")

    def test_species_not_in_area_is_404(self):
        resp = self.client.get("/api/predict?area_id=yellowstone&date=2027-06-15&species=Panthera%20leo")
        self.assertEqual(resp.status_code, 404)

    def test_predictions_carry_why_factors(self):
        resp = self.client.get("/api/predict?area_id=yellowstone&date=2027-06-15&species=Ursus%20arctos")
        body = resp.get_json()
        self.assertIn("factors", body["predictions"][0])
        self.assertGreater(len(body["predictions"][0]["factors"]), 0)

    def test_new_pilot_areas_serve_predictions(self):
        # The Arctic and Amazon areas added alongside the original three
        # pilots go through the exact same endpoint -- confirms the
        # expansion is fully wired, not just present in the data files.
        for area_id, species in (("svalbard_arctic", "Ursus maritimus"), ("amazon_rainforest", "Panthera onca")):
            resp = self.client.get(f"/api/predict?area_id={area_id}&date=2027-06-15&species={species.replace(' ', '%20')}")
            self.assertEqual(resp.status_code, 200, f"{area_id} failed: {resp.get_json()}")
            body = resp.get_json()
            self.assertEqual(len(body["predictions"]), 1)
            self.assertEqual(body["predictions"][0]["scientific_name"], species)


class BestWindowEndpointTests(ApiTestCase):
    def test_valid_request(self):
        resp = self.client.get("/api/best-window?area_id=yellowstone&species=Ursus%20arctos")
        self.assertEqual(resp.status_code, 200)
        body = resp.get_json()
        self.assertGreater(len(body["points"]), 0)

    def test_missing_params_is_400(self):
        resp = self.client.get("/api/best-window?area_id=yellowstone")
        self.assertEqual(resp.status_code, 400)


class PredictLocationEndpointTests(ApiTestCase):
    """
    'Explore Anywhere' endpoint tests. GBIF and live climate normals are
    mocked so these run offline and deterministically -- the wiring
    (query-param validation, bounds, response shape) is what's under test
    here, not GBIF/Open-Meteo's actual live answers.
    """

    def setUp(self):
        super().setUp()
        ps._presence_cache.clear()

    def test_missing_lat_lon_is_400(self):
        resp = self.client.get("/api/predict-location?date=2027-06-15")
        self.assertEqual(resp.status_code, 400)

    def test_out_of_range_lat_is_400(self):
        resp = self.client.get("/api/predict-location?lat=999&lon=0&date=2027-06-15")
        self.assertEqual(resp.status_code, 400)

    def test_missing_date_is_400(self):
        resp = self.client.get("/api/predict-location?lat=44.65&lon=-110.45")
        self.assertEqual(resp.status_code, 400)

    def test_bad_date_is_400(self):
        resp = self.client.get("/api/predict-location?lat=44.65&lon=-110.45&date=not-a-date")
        self.assertEqual(resp.status_code, 400)

    def test_valid_request_near_yellowstone(self):
        with (
            patch("app.services.gbif_client.presence_count", return_value=200),
            patch("app.services.inaturalist_client.presence_count", return_value=0),
            patch("app.services.weather_client.climate_normals", return_value=_synthetic_normals()),
        ):
            resp = self.client.get("/api/predict-location?lat=44.65&lon=-110.45&date=2027-06-15")
        self.assertEqual(resp.status_code, 200)
        body = resp.get_json()
        self.assertEqual(body["nearest_area"]["id"], "yellowstone")
        self.assertGreater(len(body["predictions"]), 0)
        self.assertIn("data_source", body)

    def test_live_weather_outage_is_a_clean_503_not_a_crash(self):
        with (
            patch("app.services.gbif_client.presence_count", return_value=200),
            patch("app.services.inaturalist_client.presence_count", return_value=0),
            patch("app.services.weather_client.climate_normals", side_effect=RuntimeError("network down")),
        ):
            resp = self.client.get("/api/predict-location?lat=44.65&lon=-110.45&date=2027-06-15")
        self.assertEqual(resp.status_code, 503)
        self.assertIn("error", resp.get_json())

    def test_radius_is_clamped_not_rejected(self):
        # A wildly out-of-range radius should be clamped to the [10, 300]km
        # bound, not cause an error -- callers shouldn't have to know the
        # exact bound to use the endpoint at all.
        with (
            patch("app.services.gbif_client.presence_count", return_value=200),
            patch("app.services.inaturalist_client.presence_count", return_value=0),
            patch("app.services.weather_client.climate_normals", return_value=_synthetic_normals()),
        ):
            resp = self.client.get("/api/predict-location?lat=44.65&lon=-110.45&date=2027-06-15&radius_km=99999")
        self.assertEqual(resp.status_code, 200)


class BestWindowLocationEndpointTests(ApiTestCase):
    def setUp(self):
        super().setUp()
        ps._presence_cache.clear()

    def test_missing_species_is_400(self):
        resp = self.client.get("/api/best-window-location?lat=44.65&lon=-110.45")
        self.assertEqual(resp.status_code, 400)

    def test_valid_request(self):
        with (
            patch("app.services.gbif_client.presence_count", return_value=200),
            patch("app.services.inaturalist_client.presence_count", return_value=0),
            patch("app.services.weather_client.climate_normals", return_value=_synthetic_normals()),
        ):
            resp = self.client.get(
                "/api/best-window-location?lat=44.65&lon=-110.45&species=Ursus%20arctos"
            )
        self.assertEqual(resp.status_code, 200)
        self.assertGreater(len(resp.get_json()["points"]), 0)

    def test_species_not_confirmed_nearby_is_404(self):
        # Both sources must report nothing for this to 404 -- GBIF alone
        # coming up empty isn't enough, since iNaturalist is an independent
        # corroboration source (see app.ml.prediction_service._local_evidence_cached)
        # and could still confirm the species on its own. This test was a
        # real regression on GitHub's CI runners (which have live internet,
        # unlike the sandbox this was built in): with only GBIF mocked to 0,
        # a real live iNaturalist record of a grizzly bear near Yellowstone
        # (a real, correct sighting -- grizzlies do live there) made the
        # species pass the presence gate and return 200 instead of 404.
        with (
            patch("app.services.gbif_client.presence_count", return_value=0),
            patch("app.services.inaturalist_client.presence_count", return_value=0),
            patch("app.services.weather_client.climate_normals", return_value=_synthetic_normals()),
        ):
            resp = self.client.get(
                "/api/best-window-location?lat=44.65&lon=-110.45&species=Ursus%20arctos"
            )
        self.assertEqual(resp.status_code, 404)


class NotFoundTests(ApiTestCase):
    def test_unknown_route_is_json_404(self):
        resp = self.client.get("/api/does-not-exist")
        self.assertEqual(resp.status_code, 404)
        self.assertIn("error", resp.get_json())


if __name__ == "__main__":
    unittest.main()
