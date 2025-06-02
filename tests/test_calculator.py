import pytest
from fastapi.testclient import TestClient
from main import app, reset_calculation_history, CalculatorService

client = TestClient(app)

@pytest.fixture(autouse=True)
def reset_history():
    """Reset history before each test"""
    reset_calculation_history()

def test_basic_addition():
    response = client.post("/calculate/basic", json={
        "a": 10, 
        "b": 5, 
        "operation": "add"
    })
    assert response.status_code == 200
    data = response.json()
    assert data["result"] == 15
    assert data["status"] == "success"

def test_basic_division_by_zero():
    response = client.post("/calculate/basic", json={
        "a": 10, 
        "b": 0, 
        "operation": "divide"
    })
    assert response.status_code == 400

def test_advanced_sqrt():
    response = client.post("/calculate/advanced", json={
        "value": 16, 
        "operation": "sqrt"
    })
    assert response.status_code == 200
    data = response.json()
    assert data["result"] == 4.0

def test_expression_evaluation():
    response = client.post("/calculate/expression", json={
        "expression": "(10 + 5) * 2"
    })
    assert response.status_code == 200
    data = response.json()
    assert data["result"] == 30

def test_calculator_service_directly():
    """Test the service class directly"""
    calc = CalculatorService()
    result = calc.perform_basic_calculation(10, 5, "add")
    assert result == 15

def test_history():
    # Perform a calculation
    client.post("/calculate/basic", json={"a": 1, "b": 1, "operation": "add"})
    
    # Check history
    response = client.get("/history")
    assert response.status_code == 200
    data = response.json()
    assert data["total_count"] == 1
    assert len(data["calculations"]) == 1

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"