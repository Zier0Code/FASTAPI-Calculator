import pytest
from main import safe_eval

def test_addition():
    assert safe_eval("2 + 3") == 5

def test_subtraction():
    assert safe_eval("10 - 4") == 6

def test_multiplication():
    assert safe_eval("3 * 7") == 21

def test_division():
    assert safe_eval("8 / 2") == 4

def test_operator_precedence():
    assert safe_eval("2 + 3 * 4") == 14

def test_parentheses():
    assert safe_eval("(2 + 3) * 4") == 20

def test_math_functions():
    assert safe_eval("sqrt(16)") == 4
    assert round(safe_eval("log(100, 10)"), 5) == 2
    assert safe_eval("abs(-7)") == 7

def test_invalid_expression():
    with pytest.raises(ValueError):
        safe_eval("2 + ")

def test_division_by_zero():
    with pytest.raises(ValueError):
        safe_eval("1 / 0")

def test_unsafe_expression():
    with pytest.raises(ValueError):
        # Trying to access a function not allowed
        safe_eval("__import__('os').system('echo hello')")