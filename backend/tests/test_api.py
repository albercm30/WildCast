"""
Tests for the Flask API (app.main), via Flask's own test client -- no
running server, no network, no sockets. Like test_prediction_service.py,
these rely on the model + cache already present in the repo.
"""
import unittest

from app.main import create_app


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
            {"yellowstone", "kruger", "brisbane_urban", "svalbard_arctic", "amazon_rainforest"},
        )

    def test_cors_header_present(self):
        resp = self.client.get("/api/health")
        self.assertEqual(resp.headers.get("Access-Control-Allow-Origin"), "*")


class AreasTests(ApiTestCase):
    def test_list_areas(self):
        resp = self.client.get("/api/areas")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(len(resp.get_json()), 5)

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


class NotFoundTests(ApiTestCase):
    def test_unknown_route_is_json_404(self):
        resp = self.client.get("/api/does-not-exist")
        self.assertEqual(resp.status_code, 404)
        self.assertIn("error", resp.get_json())


if __name__ == "__main__":
    unittest.main()
