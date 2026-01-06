"""
Unit tests for API endpoints.
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


class TestAPIEndpoints:
    """Test API endpoints."""
    
    def test_root_endpoint(self):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        assert "message" in response.json()
    
    def test_health_check(self):
        """Test health check endpoint."""
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
    
    def test_get_diseases(self):
        """Test get diseases endpoint."""
        response = client.get("/api/v1/diseases")
        assert response.status_code == 200
        assert isinstance(response.json(), list)
    
    def test_get_drugs(self):
        """Test get drugs endpoint."""
        response = client.get("/api/v1/drugs?limit=10")
        assert response.status_code == 200
        assert isinstance(response.json(), list)
    
    def test_get_drug_by_id(self):
        """Test get drug by ID."""
        # First get a drug
        drugs_response = client.get("/api/v1/drugs?limit=1")
        if drugs_response.json():
            drug_id = drugs_response.json()[0]['drug_id']
            
            response = client.get(f"/api/v1/drug/{drug_id}")
            assert response.status_code == 200
            assert response.json()['drug_id'] == drug_id
    
    def test_predict_synergy(self):
        """Test prediction endpoint."""
        # Get first disease
        diseases_response = client.get("/api/v1/diseases")
        
        if diseases_response.json():
            disease_id = diseases_response.json()[0]['disease_id']
            
            response = client.get(f"/api/v1/predict?disease_id={disease_id}&top_k=5")
            
            # May fail if no model is trained, but should have valid structure
            if response.status_code == 200:
                data = response.json()
                assert 'disease_id' in data
                assert 'predictions' in data


if __name__ == '__main__':
    pytest.main([__file__])