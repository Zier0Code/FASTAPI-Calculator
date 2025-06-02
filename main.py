"""
Professional Calculator API built with FastAPI
Fixed version addressing common validation issues
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Union
import math
import logging
from datetime import datetime
from enum import Enum
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Calculator API",
    description="A professional calculator API with comprehensive mathematical operations",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models - Simplified and more robust
class BasicCalculationRequest(BaseModel):
    a: float = Field(..., description="First operand", example=10.5)
    b: float = Field(..., description="Second operand", example=5.2)
    operation: str = Field(..., description="Operation: add, subtract, multiply, divide", example="add")

class AdvancedCalculationRequest(BaseModel):
    value: float = Field(..., description="Input value", example=16.0)
    operation: str = Field(..., description="Operation: sqrt, power, log, sin, cos, tan", example="sqrt")
    base: Optional[float] = Field(None, description="Base for logarithm or power operations", example=2.0)

class ExpressionRequest(BaseModel):
    expression: str = Field(..., description="Mathematical expression", max_length=200, example="(10 + 5) * 2")

class CalculationResponse(BaseModel):
    result: float = Field(..., description="Calculation result")
    operation: str = Field(..., description="Operation performed")
    timestamp: str = Field(..., description="Calculation timestamp")
    status: str = Field(default="success", description="Operation status")

class HistoryResponse(BaseModel):
    calculations: List[CalculationResponse] = Field(default=[])
    total_count: int = Field(default=0)

# In-memory storage
calculation_history: List[CalculationResponse] = []

def add_to_history(response: CalculationResponse):
    """Add calculation to history"""
    calculation_history.append(response)
    if len(calculation_history) > 100:
        calculation_history.pop(0)

def safe_float_conversion(value: Union[int, float]) -> float:
    """Safely convert numeric values to float"""
    try:
        result = float(value)
        if math.isnan(result) or math.isinf(result):
            raise ValueError("Result is not a valid number")
        return result
    except (ValueError, OverflowError):
        raise HTTPException(status_code=400, detail="Invalid numeric result")

# Core calculation functions
def perform_basic_calculation(a: float, b: float, operation: str) -> float:
    """Perform basic arithmetic operations"""
    operation = operation.lower().strip()
    
    if operation in ["add", "+"]:
        return a + b
    elif operation in ["subtract", "-"]:
        return a - b
    elif operation in ["multiply", "*", "×"]:
        return a * b
    elif operation in ["divide", "/", "÷"]:
        if b == 0:
            raise HTTPException(status_code=400, detail="Division by zero is not allowed")
        return a / b
    else:
        raise HTTPException(status_code=400, detail=f"Unknown operation: {operation}")

def perform_advanced_calculation(value: float, operation: str, base: Optional[float] = None) -> float:
    """Perform advanced mathematical operations"""
    operation = operation.lower().strip()
    
    try:
        if operation == "sqrt":
            if value < 0:
                raise HTTPException(status_code=400, detail="Cannot calculate square root of negative number")
            return math.sqrt(value)
        
        elif operation == "power":
            if base is None:
                raise HTTPException(status_code=400, detail="Base is required for power operation")
            return pow(base, value)
        
        elif operation == "log":
            if value <= 0:
                raise HTTPException(status_code=400, detail="Logarithm input must be positive")
            if base and base > 0 and base != 1:
                return math.log(value, base)
            return math.log(value)
        
        elif operation == "sin":
            return math.sin(math.radians(value))
        
        elif operation == "cos":
            return math.cos(math.radians(value))
        
        elif operation == "tan":
            return math.tan(math.radians(value))
        
        else:
            raise HTTPException(status_code=400, detail=f"Unknown advanced operation: {operation}")
            
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=400, detail=f"Calculation error: {str(e)}")

def evaluate_expression(expression: str) -> float:
    """Safely evaluate mathematical expressions"""
    try:
        # Clean the expression
        expression = expression.strip()
        
        # Basic validation - only allow safe characters
        safe_pattern = r'^[0-9+\-*/.() ]+$'
        if not re.match(safe_pattern, expression):
            raise HTTPException(status_code=400, detail="Expression contains invalid characters")
        
        # Replace common symbols
        expression = expression.replace('×', '*').replace('÷', '/')
        
        # Evaluate safely
        result = eval(expression, {"__builtins__": {}})
        
        if not isinstance(result, (int, float)):
            raise HTTPException(status_code=400, detail="Expression must evaluate to a number")
        
        return float(result)
        
    except SyntaxError:
        raise HTTPException(status_code=400, detail="Invalid mathematical expression")
    except ZeroDivisionError:
        raise HTTPException(status_code=400, detail="Division by zero in expression")
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=400, detail=f"Expression evaluation error: {str(e)}")

# API Routes
@app.get("/", response_class=HTMLResponse)
async def root():
    """Welcome page with API information"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Calculator API</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 50px auto; padding: 20px; background: #f8f9fa; }
            .container { background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            .header { text-align: center; color: #333; margin-bottom: 30px; }
            .endpoint { background: #f8f9fa; padding: 20px; margin: 15px 0; border-radius: 8px; border-left: 4px solid #007bff; }
            .method { background: #007bff; color: white; padding: 5px 12px; border-radius: 4px; font-weight: bold; font-size: 12px; }
            .example { background: #e9ecef; padding: 10px; border-radius: 4px; margin-top: 10px; font-family: monospace; }
            .links { text-align: center; margin-top: 30px; }
            .links a { margin: 0 15px; padding: 10px 20px; background: #007bff; color: white; text-decoration: none; border-radius: 5px; }
            .links a:hover { background: #0056b3; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🧮 Calculator API</h1>
                <p>Professional calculator service built with FastAPI</p>
            </div>
            
            <h2>Available Endpoints:</h2>
            
            <div class="endpoint">
                <span class="method">POST</span> <strong>/calculate/basic</strong>
                <p>Perform basic arithmetic operations</p>
                <div class="example">{"a": 10, "b": 5, "operation": "add"}</div>
            </div>
            
            <div class="endpoint">
                <span class="method">POST</span> <strong>/calculate/advanced</strong>
                <p>Perform advanced mathematical operations</p>
                <div class="example">{"value": 16, "operation": "sqrt"}</div>
            </div>
            
            <div class="endpoint">
                <span class="method">POST</span> <strong>/calculate/expression</strong>
                <p>Evaluate mathematical expressions</p>
                <div class="example">{"expression": "(10 + 5) * 2"}</div>
            </div>
            
            <div class="endpoint">
                <span class="method">GET</span> <strong>/history</strong>
                <p>Get calculation history</p>
            </div>
            
            <div class="links">
                <a href="/docs">📚 Interactive API Docs</a>
                <a href="/redoc">📖 ReDoc Documentation</a>
            </div>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.post("/calculate/basic", response_model=CalculationResponse)
async def calculate_basic(request: BasicCalculationRequest):
    """
    Perform basic arithmetic calculations
    
    Supported operations: add, subtract, multiply, divide
    """
    try:
        result = perform_basic_calculation(request.a, request.b, request.operation)
        result = safe_float_conversion(result)
        
        response = CalculationResponse(
            result=result,
            operation=f"{request.a} {request.operation} {request.b}",
            timestamp=datetime.now().isoformat(),
            status="success"
        )
        
        add_to_history(response)
        logger.info(f"Basic calculation: {response.operation} = {response.result}")
        
        return response
        
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/calculate/advanced", response_model=CalculationResponse)
async def calculate_advanced(request: AdvancedCalculationRequest):
    """
    Perform advanced mathematical calculations
    
    Supported operations: sqrt, power, log, sin, cos, tan
    """
    try:
        result = perform_advanced_calculation(request.value, request.operation, request.base)
        result = safe_float_conversion(result)
        
        operation_desc = f"{request.operation}({request.value})"
        if request.base is not None:
            operation_desc = f"{request.operation}({request.value}, base={request.base})"
        
        response = CalculationResponse(
            result=result,
            operation=operation_desc,
            timestamp=datetime.now().isoformat(),
            status="success"
        )
        
        add_to_history(response)
        logger.info(f"Advanced calculation: {response.operation} = {response.result}")
        
        return response
        
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/calculate/expression", response_model=CalculationResponse)
async def calculate_expression(request: ExpressionRequest):
    """
    Evaluate mathematical expressions
    
    Supports: +, -, *, /, (), basic mathematical expressions
    """
    try:
        result = evaluate_expression(request.expression)
        result = safe_float_conversion(result)
        
        response = CalculationResponse(
            result=result,
            operation=f"eval({request.expression})",
            timestamp=datetime.now().isoformat(),
            status="success"
        )
        
        add_to_history(response)
        logger.info(f"Expression evaluation: {response.operation} = {response.result}")
        
        return response
        
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/history", response_model=HistoryResponse)
async def get_history(limit: int = Query(default=10, ge=1, le=100)):
    """Get calculation history"""
    try:
        recent_calculations = calculation_history[-limit:] if calculation_history else []
        
        return HistoryResponse(
            calculations=list(reversed(recent_calculations)),
            total_count=len(calculation_history)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving history: {str(e)}")

@app.delete("/history")
async def clear_history():
    """Clear calculation history"""
    try:
        global calculation_history
        calculation_history.clear()
        logger.info("Calculation history cleared")
        return {"message": "History cleared successfully", "status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing history: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "total_calculations": len(calculation_history),
        "version": "1.0.0"
    }

# Global exception handler
@app.exception_handler(422)
async def validation_exception_handler(request, exc):
    """Handle validation errors"""
    return {
        "error": "Validation Error",
        "message": "Please check your request format",
        "details": str(exc)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True,
        log_level="info"
    )