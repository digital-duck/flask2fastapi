"""
Basic FastAPI Application Example
Chapter 4-6: Demonstrates async FastAPI patterns after migration

This example shows the FastAPI equivalent with:
- Asynchronous request handling
- Pydantic models for validation
- Modern route definitions with type hints
- Automatic documentation generation

Usage:
    python src/fastapi_examples/basic_app.py
"""

import asyncio
import time
import logging
from datetime import datetime
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, EmailStr, Field
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app with metadata
app = FastAPI(
    title="FastAPI Async Example",
    description="Demonstrates async patterns for Flask migration",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# In-memory storage for demo (in real app, use async database)
users_db: List[dict] = []
user_counter = 0


# Pydantic models for automatic validation
class UserCreate(BaseModel):
    """User creation model with validation"""
    name: str = Field(..., min_length=1, max_length=80, description="User's full name")
    email: EmailStr = Field(..., description="User's email address")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "name": "John Doe",
                "email": "john.doe@example.com"
            }
        }
    }


class UserResponse(BaseModel):
    """User response model"""
    id: int = Field(..., description="User ID")
    name: str = Field(..., description="User's full name")
    email: str = Field(..., description="User's email address")
    created_at: datetime = Field(..., description="Creation timestamp")
    
    model_config = {"from_attributes": True}


class HealthResponse(BaseModel):
    """Health check response model"""
    status: str
    timestamp: float
    database: str
    uptime: float


class UsersListResponse(BaseModel):
    """Users list response model"""
    users: List[UserResponse]
    count: int
    processing_time: float
    note: str


# Dependency for simulating database session
async def get_db_session():
    """Simulate async database session"""
    # In real app, this would return async database session
    await asyncio.sleep(0.001)  # Simulate connection time
    return "async_db_session"


# Startup event
@app.on_event("startup")
async def startup_event():
    """Application startup tasks"""
    logger.info("FastAPI application starting up...")
    logger.info("Async database connections initialized")


@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown tasks"""
    logger.info("FastAPI application shutting down...")


# Routes
@app.get("/", response_model=dict)
async def home():
    """Home endpoint with async processing"""
    # Simulate some async work
    await asyncio.sleep(0.01)
    
    return {
        "message": "FastAPI Application", 
        "version": "1.0.0",
        "framework": "FastAPI (Asynchronous)",
        "features": ["async/await", "type hints", "automatic validation", "OpenAPI docs"]
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Async health check endpoint"""
    # Simulate async health checks
    await asyncio.sleep(0.01)
    
    return HealthResponse(
        status="healthy",
        timestamp=time.time(),
        database="connected",
        uptime=time.time() - app.state.start_time if hasattr(app.state, 'start_time') else 0
    )


@app.get("/users", response_model=UsersListResponse)
async def get_users(db_session=Depends(get_db_session)):
    """Get all users - asynchronous operation"""
    start_time = time.time()
    
    # Simulate async database query delay (non-blocking)
    await asyncio.sleep(0.1)
    
    processing_time = time.time() - start_time
    
    logger.info(f"Retrieved {len(users_db)} users in {processing_time:.3f}s (async)")
    
    return UsersListResponse(
        users=[UserResponse(**user) for user in users_db],
        count=len(users_db),
        processing_time=processing_time,
        note="This is an async operation - doesn't block other requests"
    )


@app.post("/users", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def create_user(user_data: UserCreate, db_session=Depends(get_db_session)):
    """Create new user - asynchronous operation with automatic validation"""
    global user_counter
    
    # Check if user exists (async)
    await asyncio.sleep(0.02)  # Simulate async database check
    
    existing_user = next((u for u in users_db if u['email'] == user_data.email), None)
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="User with this email already exists"
        )
    
    # Simulate async validation delay (non-blocking)
    await asyncio.sleep(0.05)
    
    user_counter += 1
    new_user = {
        "id": user_counter,
        "name": user_data.name,
        "email": user_data.email,
        "created_at": datetime.now()
    }
    
    users_db.append(new_user)
    
    logger.info(f"Created user: {new_user['email']} (async)")
    return UserResponse(**new_user)


@app.get("/users/{user_id}", response_model=UserResponse)
async def get_user(user_id: int, db_session=Depends(get_db_session)):
    """Get single user - asynchronous operation"""
    # Simulate async database lookup delay (non-blocking)
    await asyncio.sleep(0.05)
    
    user = next((u for u in users_db if u['id'] == user_id), None)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User with ID {user_id} not found"
        )
    
    return UserResponse(**user)


@app.get("/slow-operation")
async def slow_operation():
    """Simulate slow operation that doesn't block other requests"""
    logger.info("Starting slow async operation...")
    
    # This doesn't block other requests - they can run concurrently
    await asyncio.sleep(2)
    
    logger.info("Slow async operation completed")
    return {
        "message": "Slow operation completed (async)",
        "note": "This didn't block other requests - they could run concurrently"
    }


@app.get("/concurrent-demo")
async def concurrent_demo():
    """Demonstrate concurrent operations"""
    async def fetch_data(delay: float, data_id: int):
        """Simulate async data fetching"""
        await asyncio.sleep(delay)
        return f"Data {data_id} fetched after {delay}s"
    
    start_time = time.time()
    
    # These operations run concurrently
    tasks = [
        fetch_data(0.5, 1),
        fetch_data(0.3, 2), 
        fetch_data(0.7, 3)
    ]
    
    results = await asyncio.gather(*tasks)
    processing_time = time.time() - start_time
    
    return {
        "results": results,
        "processing_time": processing_time,
        "note": "All operations ran concurrently - total time ≈ max(individual times)",
        "sequential_time_would_be": 0.5 + 0.3 + 0.7,
        "actual_concurrent_time": processing_time
    }


# Exception handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    """Handle 404 errors"""
    return JSONResponse(
        status_code=404,
        content={"error": "Endpoint not found", "detail": str(exc)}
    )


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    """Handle 500 errors"""
    logger.error(f"Internal error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error"}
    )


if __name__ == "__main__":
    # Store startup time
    app.state.start_time = time.time()
    
    print("\n" + "="*60)
    print("Starting FastAPI Application")
    print("="*60)
    print("📍 URL: http://localhost:8000")
    print("📖 Interactive Docs: http://localhost:8000/docs")
    print("📋 ReDoc: http://localhost:8000/redoc")
    print("\n🚀 Endpoints:")
    print("   GET  /                 - Home")
    print("   GET  /health           - Health check")
    print("   GET  /users            - List users")
    print("   POST /users            - Create user (with validation)")
    print("   GET  /users/{id}       - Get user by ID")
    print("   GET  /slow-operation   - Slow non-blocking operation")
    print("   GET  /concurrent-demo  - Concurrent operations demo")
    print("\n✨ Features:")
    print("   - Asynchronous request handling")
    print("   - Automatic request/response validation")
    print("   - Interactive API documentation")
    print("   - Type hints and modern Python patterns")
    print("   - Concurrent request processing")
    print("="*60)
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000, 
        reload=True,
        log_level="info"
    )
