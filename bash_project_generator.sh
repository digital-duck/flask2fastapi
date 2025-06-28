#!/bin/bash

# File: create_book_structure.sh
# Bash script to create "From Flask to FastAPI: A Journey of Async Programming" book structure

set -euo pipefail

# Configuration
BOOK_ROOT="async-book"
BASE_PATH="${1:-.}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
log_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

log_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

log_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Create directory structure
create_directories() {
    log_info "Creating directory structure..."
    
    local directories=(
        # Source code and examples
        "src/flask_examples"
        "src/fastapi_examples"
        "src/async_patterns"
        "src/security"
        "src/utils"
        
        # Book chapters
        "chapters/01_executive_summary"
        "chapters/02_technical_architecture"
        "chapters/03_database_migration"
        "chapters/04_api_conversion"
        "chapters/05_authentication"
        "chapters/06_background_tasks"
        "chapters/07_testing_qa"
        "chapters/08_performance"
        "chapters/09_security_compliance"
        "chapters/10_production_operations"
        "chapters/appendices"
        
        # Book assets
        "assets/diagrams"
        "assets/images"
        "assets/charts"
        "assets/code_snippets"
        
        # Book build outputs
        "book_build/markdown"
        "book_build/pdf"
        "book_build/epub"
        "book_build/html"
        
        # Research and supporting materials
        "research/benchmarks"
        "research/case_studies"
        "research/industry_data"
        "research/references"
        
        # Templates
        "templates"
        
        # Tools
        "tools/build_scripts"
        "tools/validation"
        "tools/publishing"
    )
    
    for dir in "${directories[@]}"; do
        mkdir -p "${BOOK_ROOT}/${dir}"
        log_success "Created: ${dir}"
    done
}

# Create README.md
create_readme() {
    cat > "${BOOK_ROOT}/README.md" << 'EOF'
# From Flask to FastAPI: A Journey of Async Programming

*A comprehensive guide to modern Python async web development*

## 📖 About This Book

This book documents a complete journey from Flask to FastAPI, covering not just the technical migration but the entire transformation to async programming patterns. Written for developers, architects, and technical leaders who want to modernize their Python web applications.

## 🎯 What You'll Learn

- **Async Programming Fundamentals**: Master async/await patterns and concurrent programming
- **FastAPI Mastery**: Advanced FastAPI features, dependency injection, and middleware
- **Production Migration**: Real-world strategies for migrating existing applications
- **Performance Optimization**: Concrete techniques for improving application performance
- **Security & Compliance**: Enterprise-grade security implementation
- **Operational Excellence**: Production deployment, monitoring, and scaling

## 📚 Book Structure

This book is organized into 10 comprehensive chapters:

1. **Executive Summary & Business Case** - Strategic foundation
2. **Technical Architecture & Design** - System design and planning
3. **Database Layer Migration** - Async database patterns  
4. **API Endpoint Conversion** - Route migration strategies
5. **Authentication & Authorization** - Modern security implementation
6. **Background Tasks & Job Processing** - Async task processing
7. **Testing & Quality Assurance** - Comprehensive testing strategies
8. **Performance Optimization** - Advanced performance tuning
9. **Security & Compliance** - Enterprise security framework
10. **Production Deployment and Operations** - Complete operations guide

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Basic understanding of web development
- Familiarity with Flask (helpful but not required)

### Installation
```bash
git clone <repository-url>
cd async-book
pip install -r requirements.txt
```

### Running Examples
```bash
# Flask examples
python src/flask_examples/basic_app.py

# FastAPI examples  
python src/fastapi_examples/basic_app.py

# Async patterns
python src/async_patterns/async_basics.py
```

## 🛠️ Building the Book

```bash
# Build all formats
python tools/build_scripts/build_book.py

# Validate content
python tools/build_scripts/validate_content.py

# Build specific format
python tools/build_scripts/build_book.py --format pdf
```

## 🤝 Contributing

This book benefits from community contributions:

- **Code Examples**: Improve or add new examples
- **Case Studies**: Share real migration experiences  
- **Performance Data**: Contribute benchmark results
- **Corrections**: Fix errors or improve clarity

## 📄 License

This work is licensed under Creative Commons Attribution-ShareAlike 4.0 International License.

## 🌟 Acknowledgments

Created with insights from real-world Flask-to-FastAPI migrations and contributions from the Python community.

---

*Generated on $(date)*
EOF
    
    log_success "Created: README.md"
}

# Create .gitignore
create_gitignore() {
    cat > "${BOOK_ROOT}/.gitignore" << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
venv/
env/
ENV/

# Book build outputs
book_build/pdf/*.pdf
book_build/epub/*.epub
book_build/html/*.html

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Logs
*.log
logs/

# Temporary files
temp/
tmp/
.tmp/

# Research data (keep structure, ignore large files)
research/**/*.csv
research/**/*.json
research/**/*.xlsx
research/**/large_datasets/

# Secrets and config
.env
.env.local
config.local.yml
secrets.yml
EOF
    
    log_success "Created: .gitignore"
}

# Create requirements.txt
create_requirements() {
    cat > "${BOOK_ROOT}/requirements.txt" << 'EOF'
# Book building and publishing
markdown==3.5.1
mkdocs==1.5.3
mkdocs-material==9.4.7
pymdown-extensions==10.3.1
gitpython==3.1.40

# Code examples - Flask
Flask==2.3.3
Flask-SQLAlchemy==3.0.5
Flask-Login==0.6.3
requests==2.31.0

# Code examples - FastAPI
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.4.2
sqlalchemy[asyncio]==2.0.23
asyncpg==0.29.0
aiofiles==23.2.1
python-jose[cryptography]==3.3.0
python-multipart==0.0.6

# Testing
pytest==7.4.3
pytest-asyncio==0.21.1
httpx==0.25.2

# Performance and monitoring
psutil==5.9.6
prometheus-client==0.19.0

# Development tools
black==23.11.0
isort==5.12.0
flake8==6.1.0
mypy==1.7.1

# Documentation generation
sphinx==7.2.6
sphinx-rtd-theme==1.3.0

# Book compilation
weasyprint==60.2
pypdf2==3.0.1
EOF
    
    log_success "Created: requirements.txt"
}

# Create chapter templates
create_chapter_templates() {
    local chapters=(
        "01_executive_summary:Executive Summary & Business Case"
        "02_technical_architecture:Technical Architecture & Design"
        "03_database_migration:Database Layer Migration"
        "04_api_conversion:API Endpoint Conversion"
        "05_authentication:Authentication & Authorization"
        "06_background_tasks:Background Tasks & Job Processing"
        "07_testing_qa:Testing & Quality Assurance"
        "08_performance:Performance Optimization"
        "09_security_compliance:Security & Compliance"
        "10_production_operations:Production Deployment and Operations"
    )
    
    for chapter_info in "${chapters[@]}"; do
        IFS=':' read -r chapter_num title <<< "$chapter_info"
        local chapter_number=$(echo "$chapter_num" | sed 's/^0*//')
        
        cat > "${BOOK_ROOT}/chapters/${chapter_num}/README.md" << EOF
# Chapter ${chapter_number}: ${title}

## Overview

*Brief description of what this chapter covers*

## Learning Objectives

By the end of this chapter, you will:
- [ ] Objective 1
- [ ] Objective 2
- [ ] Objective 3

## Chapter Structure

### Section ${chapter_number}.1: Introduction
*Introduction content*

### Section ${chapter_number}.2: Core Concepts
*Core concepts content*

### Section ${chapter_number}.3: Implementation
*Implementation details*

### Section ${chapter_number}.4: Best Practices
*Best practices and recommendations*

### Section ${chapter_number}.5: Real-World Examples
*Practical examples and case studies*

## Code Examples

- [Example 1](../src/examples/chapter_${chapter_num}_example_1.py)
- [Example 2](../src/examples/chapter_${chapter_num}_example_2.py)

## Key Takeaways

- Key point 1
- Key point 2  
- Key point 3

## Next Steps

Continue to [Chapter $((chapter_number + 1))](../chapters/$(printf "%02d" $((chapter_number + 1)))_next_chapter/README.md)

---

*Last updated: $(date +%Y-%m-%d)*
EOF
        
        log_success "Created: chapters/${chapter_num}/README.md"
    done
}

# Create basic Flask example
create_flask_example() {
    cat > "${BOOK_ROOT}/src/flask_examples/basic_app.py" << 'EOF'
"""
Basic Flask Application Example
Demonstrates traditional Flask patterns before migration
"""

from flask import Flask, jsonify, request
from flask_sqlalchemy import SQLAlchemy
import time

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///example.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)


class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(80), nullable=False)
    email = db.Column(db.String(120), nullable=False)

    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'email': self.email
        }


@app.route('/')
def home():
    """Home endpoint"""
    return jsonify({"message": "Flask Application", "version": "1.0"})


@app.route('/users', methods=['GET'])
def get_users():
    """Get all users - synchronous operation"""
    start_time = time.time()
    
    # Simulate database query delay
    time.sleep(0.1)
    
    users = User.query.all()
    result = [user.to_dict() for user in users]
    
    processing_time = time.time() - start_time
    
    return jsonify({
        "users": result,
        "count": len(result),
        "processing_time": processing_time
    })


@app.route('/slow-operation', methods=['GET'])
def slow_operation():
    """Simulate slow operation that blocks other requests"""
    # This blocks the entire thread
    time.sleep(2)
    return jsonify({"message": "Slow operation completed"})


if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    
    print("Starting Flask application...")
    print("Note: This is a synchronous application - requests are processed sequentially")
    app.run(debug=True, port=5000)
EOF
    
    log_success "Created: src/flask_examples/basic_app.py"
}

# Create basic FastAPI example
create_fastapi_example() {
    cat > "${BOOK_ROOT}/src/fastapi_examples/basic_app.py" << 'EOF'
"""
Basic FastAPI Application Example
Demonstrates async FastAPI patterns after migration
"""

import asyncio
import time
from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, EmailStr

# Pydantic models for validation
class UserCreate(BaseModel):
    name: str
    email: EmailStr


class UserResponse(BaseModel):
    id: int
    name: str
    email: str


# FastAPI app
app = FastAPI(
    title="FastAPI Async Example",
    description="Demonstrates async patterns for Flask migration",
    version="1.0.0"
)

# In-memory storage for demo
users_db = []
user_counter = 0


@app.get("/")
async def home():
    """Home endpoint"""
    return {"message": "FastAPI Application", "version": "1.0"}


@app.get("/users", response_model=dict)
async def get_users():
    """Get all users - asynchronous operation"""
    start_time = time.time()
    
    # Simulate async database query delay (non-blocking)
    await asyncio.sleep(0.1)
    
    processing_time = time.time() - start_time
    
    return {
        "users": users_db,
        "count": len(users_db),
        "processing_time": processing_time
    }


@app.post("/users", response_model=UserResponse, status_code=201)
async def create_user(user_data: UserCreate):
    """Create new user - asynchronous operation"""
    global user_counter
    
    # Simulate async validation delay (non-blocking)
    await asyncio.sleep(0.05)
    
    user_counter += 1
    new_user = UserResponse(
        id=user_counter,
        name=user_data.name,
        email=user_data.email
    )
    
    users_db.append(new_user.model_dump())
    return new_user


@app.get("/slow-operation")
async def slow_operation():
    """Simulate slow operation that doesn't block other requests"""
    # This doesn't block other requests - they can run concurrently
    await asyncio.sleep(2)
    return {"message": "Slow operation completed (async)"}


@app.get("/concurrent-demo")
async def concurrent_demo():
    """Demonstrate concurrent operations"""
    async def fetch_data(delay: float, data_id: int):
        await asyncio.sleep(delay)
        return f"Data {data_id} fetched"
    
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
        "note": "All operations ran concurrently"
    }


if __name__ == "__main__":
    import uvicorn
    
    print("Starting FastAPI application...")
    print("Note: This is an async application - requests can be processed concurrently")
    print("Visit http://localhost:8000/docs for interactive API documentation")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
EOF
    
    log_success "Created: src/fastapi_examples/basic_app.py"
}

# Create build script template
create_build_script() {
    cat > "${BOOK_ROOT}/tools/build_scripts/build_book.py" << 'EOF'
#!/usr/bin/env python3
"""
Book Build Script
Compiles markdown chapters into various book formats
"""

import argparse
import os
import shutil
from pathlib import Path
import subprocess


def build_markdown():
    """Compile all chapters into single markdown file"""
    print("📝 Building markdown version...")
    
    chapters_dir = Path("chapters")
    output_file = Path("book_build/markdown/complete_book.md")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as outfile:
        # Write title page
        outfile.write("# From Flask to FastAPI: A Journey of Async Programming\n\n")
        
        