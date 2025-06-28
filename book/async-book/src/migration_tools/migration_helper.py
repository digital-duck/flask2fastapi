"""
Migration Helper Tools
Chapter 4-5: Utilities for converting Flask code to FastAPI

This module provides helper functions and classes to assist in migrating
Flask applications to FastAPI, including route conversion, model migration,
and pattern transformation utilities.

Usage:
    from migration_tools.migration_helper import FlaskToFastAPIConverter
    
    converter = FlaskToFastAPIConverter()
    fastapi_route = converter.convert_flask_route(flask_route_function)
"""

import re
import ast
import inspect
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass
from enum import Enum


class RouteMethod(Enum):
    """HTTP methods supported in route conversion"""
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"
    PATCH = "PATCH"


@dataclass
class FlaskRoute:
    """Represents a Flask route for conversion"""
    path: str
    methods: List[str]
    function_name: str
    function_code: str
    parameters: List[str]
    return_type: Optional[str] = None


@dataclass
class FastAPIRoute:
    """Represents the converted FastAPI route"""
    path: str
    method: str
    function_name: str
    function_code: str
    pydantic_models: List[str]
    dependencies: List[str]


class FlaskToFastAPIConverter:
    """Main converter class for Flask to FastAPI migration"""
    
    def __init__(self):
        self.pydantic_models: Dict[str, str] = {}
        self.dependencies: List[str] = []
    
    def convert_flask_route(self, flask_function: Callable) -> FastAPIRoute:
        """Convert a Flask route function to FastAPI equivalent"""
        
        # Extract Flask route information
        flask_route = self._extract_flask_route_info(flask_function)
        
        # Convert to FastAPI
        fastapi_route = self._convert_to_fastapi(flask_route)
        
        return fastapi_route
    
    def _extract_flask_route_info(self, func: Callable) -> FlaskRoute:
        """Extract route information from Flask function"""
        
        # Get function source code
        source = inspect.getsource(func)
        
        # Parse route decorator
        route_pattern = r'@app\.route\(['"]([^'"]*)['"](?:.*methods=\[([^\]]*)\])?.*\)'
        route_match = re.search(route_pattern, source)
        
        if not route_match:
            raise ValueError("No Flask route decorator found")
        
        path = route_match.group(1)
        methods_str = route_match.group(2) or "'GET'"
        methods = [m.strip(''"') for m in methods_str.split(',')]
        
        # Extract function parameters
        sig = inspect.signature(func)
        parameters = list(sig.parameters.keys())
        
        return FlaskRoute(
            path=path,
            methods=methods,
            function_name=func.__name__,
            function_code=source,
            parameters=parameters,
            return_type=str(sig.return_annotation) if sig.return_annotation != inspect.Signature.empty else None
        )
    
    def _convert_to_fastapi(self, flask_route: FlaskRoute) -> FastAPIRoute:
        """Convert Flask route to FastAPI equivalent"""
        
        # Convert path parameters
        fastapi_path = self._convert_path_parameters(flask_route.path)
        
        # Convert function to async
        fastapi_code = self._convert_function_to_async(flask_route)
        
        # Generate Pydantic models if needed
        models = self._generate_pydantic_models(flask_route)
        
        # Determine primary HTTP method
        primary_method = flask_route.methods[0] if flask_route.methods else "GET"
        
        return FastAPIRoute(
            path=fastapi_path,
            method=primary_method,
            function_name=flask_route.function_name,
            function_code=fastapi_code,
            pydantic_models=models,
            dependencies=self.dependencies.copy()
        )
    
    def _convert_path_parameters(self, flask_path: str) -> str:
        """Convert Flask path parameters to FastAPI format"""
        # Convert <param> to {param}
        fastapi_path = re.sub(r'<([^>]+)>', r'{\1}', flask_path)
        
        # Handle typed parameters <int:param> -> {param: int}
        fastapi_path = re.sub(r'<(int|float|string):([^>]+)>', r'{\2}', fastapi_path)
        
        return fastapi_path
    
    def _convert_function_to_async(self, flask_route: FlaskRoute) -> str:
        """Convert Flask function to async FastAPI function"""
        
        lines = flask_route.function_code.split('\n')
        converted_lines = []
        
        for line in lines:
            # Skip Flask decorator
            if '@app.route' in line:
                continue
            
            # Convert function definition to async
            if line.strip().startswith('def '):
                line = line.replace('def ', 'async def ')
            
            # Convert Flask imports
            line = self._convert_imports(line)
            
            # Convert request handling
            line = self._convert_request_handling(line)
            
            # Convert response handling
            line = self._convert_response_handling(line)
            
            converted_lines.append(line)
        
        return '\n'.join(converted_lines)
    
    def _convert_imports(self, line: str) -> str:
        """Convert Flask imports to FastAPI equivalents"""
        conversions = {
            'from flask import': 'from fastapi import',
            'jsonify': 'JSONResponse',
            'request': 'Request',
            'abort': 'HTTPException'
        }
        
        for flask_import, fastapi_import in conversions.items():
            line = line.replace(flask_import, fastapi_import)
        
        return line
    
    def _convert_request_handling(self, line: str) -> str:
        """Convert Flask request handling to FastAPI"""
        
        # Convert request.get_json() to dependency injection
        if 'request.get_json()' in line:
            line = line.replace('request.get_json()', 'request_data')
            if 'request_data' not in self.dependencies:
                self.dependencies.append('request_data: dict = Body(...)')
        
        # Convert request.args to query parameters
        if 'request.args.get(' in line:
            line = re.sub(
                r'request\.args\.get\(['"]([^'"]+)['"]\)',
                r'\1: Optional[str] = Query(None)',
                line
            )
        
        return line
    
    def _convert_response_handling(self, line: str) -> str:
        """Convert Flask response handling to FastAPI"""
        
        # Convert jsonify() to return dict
        line = re.sub(r'return jsonify\(([^)]+)\)', r'return \1', line)
        
        # Convert Flask abort() to HTTPException
        line = re.sub(
            r'abort\((\d+)\)',
            r'raise HTTPException(status_code=\1)',
            line
        )
        
        return line
    
    def _generate_pydantic_models(self, flask_route: FlaskRoute) -> List[str]:
        """Generate Pydantic models for request/response"""
        models = []
        
        # Analyze function to determine if models are needed
        if 'POST' in flask_route.methods or 'PUT' in flask_route.methods:
            model_name = f"{flask_route.function_name.title()}Request"
            model_code = f"""
class {model_name}(BaseModel):
    "Request model for {flask_route.function_name}"
    # Add fields based on analysis of request.get_json() usage
    pass
"""
            models.append(model_code.strip())
        
        return models


# Helper functions for common migration patterns
def convert_flask_error_handler(handler_func: Callable) -> str:
    """Convert Flask error handler to FastAPI exception handler"""
    
    source = inspect.getsource(handler_func)
    
    # Extract error code
    error_pattern = r'@app\.errorhandler\((\d+)\)'
    error_match = re.search(error_pattern, source)
    
    if not error_match:
        return source
    
    error_code = error_match.group(1)
    
    # Convert to FastAPI exception handler
    converted = f"""
@app.exception_handler({error_code})
async def handle_{error_code}_error(request: Request, exc: HTTPException):
    "Handle {error_code} errors"
    return JSONResponse(
        status_code={error_code},
        content={{"error": "Error message here"}}
    )
"""
    
    return converted.strip()


def convert_flask_middleware(middleware_func: Callable) -> str:
    """Convert Flask before_request/after_request to FastAPI middleware"""
    
    source = inspect.getsource(middleware_func)
    
    if '@app.before_request' in source:
        return """
@app.middleware("http")
async def before_request_middleware(request: Request, call_next):
    "Convert Flask before_request to FastAPI middleware"
    # Add your before request logic here
    response = await call_next(request)
    return response
"""
    
    elif '@app.after_request' in source:
        return """
@app.middleware("http")
async def after_request_middleware(request: Request, call_next):
    "Convert Flask after_request to FastAPI middleware"
    response = await call_next(request)
    # Add your after request logic here
    return response
"""
    
    return source


def analyze_flask_app(app_file_path: str) -> Dict[str, Any]:
    """Analyze a Flask application file and provide migration insights"""
    
    with open(app_file_path, 'r') as f:
        content = f.read()
    
    analysis = {
        "routes": [],
        "error_handlers": [],
        "middleware": [],
        "imports": [],
        "complexity_score": 0
    }
    
    # Find routes
    route_pattern = r'@app\.route\(['"]([^'"]*)['"].*?\)\s*def\s+(\w+)'
    routes = re.findall(route_pattern, content, re.DOTALL)
    analysis["routes"] = [{"path": path, "function": func} for path, func in routes]
    
    # Find error handlers
    error_pattern = r'@app\.errorhandler\((\d+)\)\s*def\s+(\w+)'
    errors = re.findall(error_pattern, content)
    analysis["error_handlers"] = [{"code": code, "function": func} for code, func in errors]
    
    # Find before/after request handlers
    middleware_pattern = r'@app\.(before_request|after_request)\s*def\s+(\w+)'
    middleware = re.findall(middleware_pattern, content)
    analysis["middleware"] = [{"type": mw_type, "function": func} for mw_type, func in middleware]
    
    # Analyze imports
    import_pattern = r'from flask import ([^\n]+)'
    imports = re.findall(import_pattern, content)
    analysis["imports"] = [imp.strip() for imp_line in imports for imp in imp_line.split(',')]
    
    # Calculate complexity score
    analysis["complexity_score"] = (
        len(analysis["routes"]) * 2 +
        len(analysis["error_handlers"]) * 1 +
        len(analysis["middleware"]) * 3 +
        len(analysis["imports"]) * 0.5
    )
    
    return analysis


def generate_migration_report(analysis: Dict[str, Any]) -> str:
    """Generate a migration complexity report"""
    
    report = f"""
Flask to FastAPI Migration Report
================================

Application Analysis:
- Routes found: {len(analysis['routes'])}
- Error handlers: {len(analysis['error_handlers'])}
- Middleware functions: {len(analysis['middleware'])}
- Flask imports: {len(analysis['imports'])}
- Complexity score: {analysis['complexity_score']:.1f}

Migration Complexity: {"Low" if analysis['complexity_score'] < 20 else "Medium" if analysis['complexity_score'] < 50 else "High"}

Routes to convert:
"""
    
    for route in analysis['routes']:
        report += f"  - {route['path']} -> {route['function']}()\n"
    
    if analysis['error_handlers']:
        report += "\nError handlers to convert:\n"
        for handler in analysis['error_handlers']:
            report += f"  - HTTP {handler['code']} -> {handler['function']}()\n"
    
    if analysis['middleware']:
        report += "\nMiddleware to convert:\n"
        for mw in analysis['middleware']:
            report += f"  - {mw['type']} -> {mw['function']}()\n"
    
    report += """
Recommended Migration Steps:
1. Set up FastAPI project structure
2. Convert route decorators to FastAPI equivalents
3. Add Pydantic models for request/response validation
4. Convert synchronous functions to async
5. Implement FastAPI middleware
6. Add FastAPI exception handlers
7. Update imports and dependencies
8. Test converted endpoints
9. Performance testing and optimization
"""
    
    return report


# Example usage
if __name__ == "__main__":
    print("Flask to FastAPI Migration Helper")
    print("=" * 40)
    
    # Example: Analyze the Flask app from our examples
    try:
        flask_app_path = "src/flask_examples/basic_app.py"
        analysis = analyze_flask_app(flask_app_path)
        report = generate_migration_report(analysis)
        print(report)
    except FileNotFoundError:
        print("Flask example file not found. Run from book root directory.")
    except Exception as e:
        print(f"Analysis failed: {e}")
    
    print("\n💡 This is a basic migration helper.")
    print("   For complex applications, manual review and testing is essential.")
