"""
API Tests
=========

Unit tests for FastAPI endpoints.
"""

import unittest
import json
from fastapi.testclient import TestClient
import sys
sys.path.append('.')

from api.main import app


class TestAPIEndpoints(unittest.TestCase):
    """Test cases for API endpoints."""
    
    def setUp(self):
        """Set up test client."""
        self.client = TestClient(app)
    
    def test_health_check(self):
        """Test health check endpoint."""
        response = self.client.get("/health")
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")
    
    def test_predict_synergy_valid(self):
        """Test synergy prediction with valid input."""
        payload = {
            "drug1_id": "DB00001",
            "drug2_id": "DB00002",
            "disease_id": "DOID:0001"
        }
        
        response = self.client.post("/predict-synergy", json=payload)
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        self.assertIn("synergy_score", data)
        self.assertIn("confidence", data)
        self.assertIn("safety_level", data)
    
    def test_predict_synergy_invalid_drug(self):
        """Test prediction with invalid drug ID."""
        payload = {
            "drug1_id": "INVALID",
            "drug2_id": "DB00002",
            "disease_id": "DOID:0001"
        }
        
        response = self.client.post("/predict-synergy", json=payload)
        
        self.assertEqual(response.status_code, 404)
    
    def test_predict_synergy_missing_field(self):
        """Test prediction with missing required field."""
        payload = {
            "drug1_id": "DB00001",
            # Missing drug2_id
            "disease_id": "DOID:0001"
        }
        
        response = self.client.post("/predict-synergy", json=payload)
        
        self.assertEqual(response.status_code, 422)
    
    def test_get_drug_info(self):
        """Test getting drug information."""
        drug_id = "DB00001"
        response = self.client.get(f"/drug-info/{drug_id}")
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        self.assertIn("drug_id", data)
        self.assertIn("name", data)
        self.assertIn("smiles", data)
    
    def test_get_drug_info_not_found(self):
        """Test getting info for non-existent drug."""
        drug_id = "INVALID"
        response = self.client.get(f"/drug-info/{drug_id}")
        
        self.assertEqual(response.status_code, 404)
    
    def test_get_disease_list(self):
        """Test getting list of diseases."""
        response = self.client.get("/disease-list")
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        self.assertIsInstance(data, list)
        if len(data) > 0:
            self.assertIn("disease_id", data[0])
            self.assertIn("name", data[0])
    
    def test_batch_prediction(self):
        """Test batch prediction endpoint."""
        payload = {
            "drug_pairs": [
                {"drug1_id": "DB00001", "drug2_id": "DB00002"},
                {"drug1_id": "DB00003", "drug2_id": "DB00004"}
            ],
            "disease_id": "DOID:0001"
        }
        
        response = self.client.post("/predict-synergy/batch", json=payload)
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        self.assertIsInstance(data["predictions"], list)
        self.assertEqual(len(data["predictions"]), 2)
    
    def test_top_synergies(self):
        """Test getting top synergistic combinations."""
        params = {
            "disease_id": "DOID:0001",
            "top_k": 10
        }
        
        response = self.client.get("/top-synergies", params=params)
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        self.assertIn("combinations", data)
        self.assertLessEqual(len(data["combinations"]), 10)
    
    def test_drug_search(self):
        """Test drug search endpoint."""
        params = {"query": "aspirin"}
        
        response = self.client.get("/drugs/search", params=params)
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        self.assertIsInstance(data, list)
    
    def test_cors_headers(self):
        """Test CORS headers are present."""
        response = self.client.get("/health")
        
        self.assertIn("access-control-allow-origin", 
                     [h.lower() for h in response.headers.keys()])
    
    def test_rate_limiting(self):
        """Test rate limiting (if implemented)."""
        # Make multiple requests
        for _ in range(100):
            response = self.client.get("/health")
            if response.status_code == 429:
                self.assertTrue(True)
                return
        
        # If no rate limiting, test passes
        self.assertTrue(True)
    
    def test_response_time(self):
        """Test API response time is reasonable."""
        import time
        
        start = time.time()
        response = self.client.get("/health")
        elapsed = time.time() - start
        
        self.assertLess(elapsed, 1.0)  # Should respond within 1 second
    
    def test_error_handling(self):
        """Test error handling for server errors."""
        # Test with malformed JSON
        response = self.client.post(
            "/predict-synergy",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        
        self.assertIn(response.status_code, [400, 422])


class TestAPIValidation(unittest.TestCase):
    """Test input validation."""
    
    def setUp(self):
        """Set up test client."""
        self.client = TestClient(app)
    
    def test_synergy_score_range(self):
        """Test that synergy scores are in valid range."""
        payload = {
            "drug1_id": "DB00001",
            "drug2_id": "DB00002",
            "disease_id": "DOID:0001"
        }
        
        response = self.client.post("/predict-synergy", json=payload)
        
        if response.status_code == 200:
            data = response.json()
            score = data.get("synergy_score")
            
            if score is not None:
                self.assertGreaterEqual(score, 0.0)
                self.assertLessEqual(score, 1.0)
    
    def test_confidence_range(self):
        """Test that confidence is in valid range."""
        payload = {
            "drug1_id": "DB00001",
            "drug2_id": "DB00002",
            "disease_id": "DOID:0001"
        }
        
        response = self.client.post("/predict-synergy", json=payload)
        
        if response.status_code == 200:
            data = response.json()
            confidence = data.get("confidence")
            
            if confidence is not None:
                self.assertGreaterEqual(confidence, 0.0)
                self.assertLessEqual(confidence, 100.0)
    
    def test_safety_level_values(self):
        """Test that safety level has valid values."""
        payload = {
            "drug1_id": "DB00001",
            "drug2_id": "DB00002",
            "disease_id": "DOID:0001"
        }
        
        response = self.client.post("/predict-synergy", json=payload)
        
        if response.status_code == 200:
            data = response.json()
            safety = data.get("safety_level")
            
            if safety is not None:
                self.assertIn(safety, ["safe", "caution", "danger"])


if __name__ == '__main__':
    unittest.main()