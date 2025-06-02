"""
Professional Calculator API built with FastAPI
Updated for Pydantic V2 compatibility and pytest testing
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Union
import math
import logging
from datetime import datetime
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

# Pydantic models - Updated for V2 compatibility
class BasicCalculationRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "a": 10.5,
                "b": 5.2,
                "operation": "add"
            }
        }
    )
    
    a: float = Field(..., description="First operand")
    b: float = Field(..., description="Second operand")
    operation: str = Field(..., description="Operation: add, subtract, multiply, divide")

class AdvancedCalculationRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "value": 16.0,
                "operation": "sqrt"
            }
        }
    )
    
    value: float = Field(..., description="Input value")
    operation: str = Field(..., description="Operation: sqrt, power, log, sin, cos, tan")
    base: Optional[float] = Field(None, description="Base for logarithm or power operations")

class ExpressionRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "expression": "(10 + 5) * 2"
            }
        }
    )
    
    expression: str = Field(..., description="Mathematical expression", max_length=200)

class CalculationResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "result": 15.7,
                "operation": "10.5 add 5.2",
                "timestamp": "2024-01-01T12:00:00",
                "status": "success"
            }
        }
    )
    
    result: float = Field(..., description="Calculation result")
    operation: str = Field(..., description="Operation performed")
    timestamp: str = Field(..., description="Calculation timestamp")
    status: str = Field(default="success", description="Operation status")

class HistoryResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "calculations": [
                    {
                        "result": 15.7,
                        "operation": "10.5 add 5.2",
                        "timestamp": "2024-01-01T12:00:00",
                        "status": "success"
                    }
                ],
                "total_count": 1
            }
        }
    )
    
    calculations: List[CalculationResponse] = Field(default=[])
    total_count: int = Field(default=0)

class ErrorResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "error": "Calculation Error",
                "message": "Division by zero is not allowed",
                "status": "error"
            }
        }
    )
    
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    status: str = Field(default="error", description="Response status")

# In-memory storage (in production, use proper database)
calculation_history: List[CalculationResponse] = []

def add_to_history(response: CalculationResponse) -> None:
    """Add calculation to history"""
    calculation_history.append(response)
    if len(calculation_history) > 100:
        calculation_history.pop(0)

def safe_float_conversion(value: Union[int, float]) -> float:
    """Safely convert numeric values to float"""
    try:
        result = float(value)
        if math.isnan(result):
            raise ValueError("Result is NaN")
        if math.isinf(result):
            raise ValueError("Result is infinite")
        return result
    except (ValueError, OverflowError) as e:
        raise HTTPException(status_code=400, detail=f"Invalid numeric result: {str(e)}")

# Calculator service class for better organization
class CalculatorService:
    @staticmethod
    def perform_basic_calculation(a: float, b: float, operation: str) -> float:
        """Perform basic arithmetic operations"""
        operation = operation.lower().strip()
        
        operations = {
            "add": lambda x, y: x + y,
            "+": lambda x, y: x + y,
            "subtract": lambda x, y: x - y,
            "-": lambda x, y: x - y,
            "multiply": lambda x, y: x * y,
            "*": lambda x, y: x * y,
            "×": lambda x, y: x * y,
            "divide": lambda x, y: x / y if y != 0 else (_ for _ in ()).throw(ZeroDivisionError("Division by zero")),
            "/": lambda x, y: x / y if y != 0 else (_ for _ in ()).throw(ZeroDivisionError("Division by zero")),
            "÷": lambda x, y: x / y if y != 0 else (_ for _ in ()).throw(ZeroDivisionError("Division by zero"))
        }
        
        if operation not in operations:
            raise HTTPException(status_code=400, detail=f"Unknown operation: {operation}")
        
        try:
            return operations[operation](a, b)
        except ZeroDivisionError:
            raise HTTPException(status_code=400, detail="Division by zero is not allowed")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Calculation error: {str(e)}")

    @staticmethod
    def perform_advanced_calculation(value: float, operation: str, base: Optional[float] = None) -> float:
        """Perform advanced mathematical operations"""
        operation = operation.lower().strip()
        
        try:
            if operation == "sqrt":
                if value < 0:
                    raise ValueError("Cannot calculate square root of negative number")
                return math.sqrt(value)
            
            elif operation == "power":
                if base is None:
                    raise ValueError("Base is required for power operation")
                return pow(base, value)
            
            elif operation == "log":
                if value <= 0:
                    raise ValueError("Logarithm input must be positive")
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
                raise ValueError(f"Unknown advanced operation: {operation}")
                
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Calculation error: {str(e)}")

    @staticmethod
    def evaluate_expression(expression: str) -> float:
        """Safely evaluate mathematical expressions"""
        try:
            # Clean the expression
            expression = expression.strip()
            
            # Basic validation - only allow safe characters
            safe_pattern = r'^[0-9+\-*/.() ]+$'
            if not re.match(safe_pattern, expression):
                raise ValueError("Expression contains invalid characters")
            
            # Replace common symbols
            expression = expression.replace('×', '*').replace('÷', '/')
            
            # Evaluate safely with restricted builtins
            allowed_names = {
                "__builtins__": {},
                "abs": abs,
                "round": round,
                "pow": pow,
                "max": max,
                "min": min
            }
            
            result = eval(expression, allowed_names)
            
            if not isinstance(result, (int, float)):
                raise ValueError("Expression must evaluate to a number")
            
            return float(result)
            
        except SyntaxError:
            raise HTTPException(status_code=400, detail="Invalid mathematical expression syntax")
        except ZeroDivisionError:
            raise HTTPException(status_code=400, detail="Division by zero in expression")
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Expression evaluation error: {str(e)}")

# Initialize calculator service
calc_service = CalculatorService()

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
            body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
                   max-width: 900px; margin: 0 auto; padding: 20px; background: #f8fafc; }
            .container { background: white; padding: 40px; border-radius: 12px; box-shadow: 0 4px 20px rgba(0,0,0,0.08); }
            .header { text-align: center; color: #1a202c; margin-bottom: 40px; }
            .header h1 { font-size: 2.5rem; margin-bottom: 10px; color: #2d3748; }
            .header p { font-size: 1.1rem; color: #718096; }
            .endpoint { background: #f7fafc; padding: 24px; margin: 20px 0; border-radius: 8px; 
                       border-left: 4px solid #4299e1; transition: all 0.2s; }
            .endpoint:hover { box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
            .method { background: #4299e1; color: white; padding: 6px 14px; border-radius: 6px; 
                     font-weight: 600; font-size: 12px; display: inline-block; margin-bottom: 12px; }
            .example { background: #2d3748; color: #e2e8f0; padding: 16px; border-radius: 6px; 
                      margin-top: 12px; font-family: 'SF Mono', Monaco, 'Cascadia Code', monospace; 
                      font-size: 14px; overflow-x: auto; }
            .links { text-align: center; margin-top: 40px; }
            .links a { margin: 0 12px; padding: 12px 24px; background: #4299e1; color: white; 
                      text-decoration: none; border-radius: 8px; font-weight: 500; transition: all 0.2s; }
            .links a:hover { background: #3182ce; transform: translateY(-1px); }
            .feature { color: #4a5568; margin-bottom: 8px; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🧮 Calculator API</h1>
                <p>Professional-grade calculator service built with FastAPI & Python</p>
            </div>
            
            <h2 style="color: #2d3748; margin-bottom: 24px;">Available Endpoints</h2>
            
            <div class="endpoint">
                <span class="method">POST</span> <strong>/calculate/basic</strong>
                <p class="feature">Perform basic arithmetic operations</p>
                <div class="example">{"a": 10.5, "b": 5.2, "operation": "add"}</div>
            </div>
            
            <div class="endpoint">
                <span class="method">POST</span> <strong>/calculate/advanced</strong>
                <p class="feature">Advanced mathematical functions</p>
                <div class="example">{"value": 16, "operation": "sqrt"}</div>
            </div>
            
            <div class="endpoint">
                <span class="method">POST</span> <strong>/calculate/expression</strong>
                <p class="feature">Evaluate complex mathematical expressions</p>
                <div class="example">{"expression": "(10 + 5) * 2 / 3"}</div>
            </div>
            
            <div class="endpoint">
                <span class="method">GET</span> <strong>/history</strong>
                <p class="feature">Retrieve calculation history with pagination</p>
                <div class="example">GET /history?limit=5</div>
            </div>
            
            <div class="links">
                <a href="/docs">📚 Interactive API Documentation</a>
                <a href="/redoc">📖 ReDoc Documentation</a>
                <a href="/health">💚 Health Check</a>
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
    
    **Supported operations:** add, subtract, multiply, divide
    """
    try:
        result = calc_service.perform_basic_calculation(request.a, request.b, request.operation)
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
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in basic calculation: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/calculate/advanced", response_model=CalculationResponse)
async def calculate_advanced(request: AdvancedCalculationRequest):
    """
    Perform advanced mathematical calculations
    
    **Supported operations:** sqrt, power, log, sin, cos, tan
    """
    try:
        result = calc_service.perform_advanced_calculation(request.value, request.operation, request.base)
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
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in advanced calculation: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/calculate/expression", response_model=CalculationResponse)
async def calculate_expression(request: ExpressionRequest):
    """
    Evaluate mathematical expressions
    
    **Supports:** +, -, *, /, (), parentheses, basic mathematical expressions
    """
    try:
        result = calc_service.evaluate_expression(request.expression)
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
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in expression evaluation: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/history", response_model=HistoryResponse)
async def get_history(limit: int = Query(default=10, ge=1, le=100, description="Number of calculations to return")):
    """Get calculation history with pagination"""
    try:
        recent_calculations = calculation_history[-limit:] if calculation_history else []
        
        return HistoryResponse(
            calculations=list(reversed(recent_calculations)),
            total_count=len(calculation_history)
        )
    except Exception as e:
        logger.error(f"Error retrieving history: {str(e)}")
        raise HTTPException(status_code=500, detail="Error retrieving calculation history")

@app.delete("/history")
async def clear_history():
    """Clear all calculation history"""
    try:
        global calculation_history
        count = len(calculation_history)
        calculation_history.clear()
        logger.info(f"Cleared {count} calculations from history")
        return {
            "message": f"Successfully cleared {count} calculations",
            "status": "success",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error clearing history: {str(e)}")
        raise HTTPException(status_code=500, detail="Error clearing calculation history")

@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "total_calculations": len(calculation_history),
        "version": "1.0.0",
        "service": "Calculator API"
    }

# Global exception handlers
@app.exception_handler(422)
async def validation_exception_handler(request, exc):
    """Handle Pydantic validation errors"""
    logger.warning(f"Validation error: {str(exc)}")
    return {
        "error": "Validation Error",
        "message": "Please check your request format and data types",
        "details": str(exc),
        "status": "error"
    }

@app.exception_handler(500)
async def internal_server_error_handler(request, exc):
    """Handle internal server errors"""
    logger.error(f"Internal server error: {str(exc)}")
    return {
        "error": "Internal Server Error",
        "message": "An unexpected error occurred",
        "status": "error"
    }

# Function to reset history (useful for testing
def reset_calculation_history():
    """Reset calculation history - useful for testing"""
    global calculation_history
    calculation_history.clear()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True,
        log_level="info"
    )