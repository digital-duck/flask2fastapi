# File: file_templates.py
"""
File Content Templates
Contains template functions for generating various file contents
"""

from datetime import datetime


def get_readme_content() -> str:
    """Generate README.md content for the book project"""
    return f"""# From Flask to FastAPI: A Journey of Async Programming

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
make build

# Validate content
make validate

# Build specific format
python tools/build_scripts/build_book.py --format pdf
```

## 🤝 Contributing

This book benefits from community contributions:

- **Code Examples**: Improve or add new examples
- **Case Studies**: Share real migration experiences  
- **Performance Data**: Contribute benchmark results
- **Corrections**: Fix errors or improve clarity

See [Contributing Guide](docs/contributing/contribution_guide.md) for details.

## 📄 License

This work is licensed under Creative Commons Attribution-ShareAlike 4.0 International License.

## 🌟 Acknowledgments

Created with insights from real-world Flask-to-FastAPI migrations and contributions from the Python community.

---

*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""


def get_gitignore_content() -> str:
    """Generate .gitignore content"""
    return """# Python
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
.venv/

# Book build outputs
book_build/pdf/*.pdf
book_build/epub/*.epub
book_build/html/*.html
book_build/latex/*.tex

# IDE
.vscode/
.idea/
*.swp
*.swo
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

# Book specific
.pytest_cache/
.coverage
htmlcov/
.tox/
"""


def get_requirements_content() -> str:
    """Generate requirements.txt content"""
    return """# Book building and publishing
click==8.1.7
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
pytest-cov==4.1.0
httpx==0.25.2

# Performance and monitoring
psutil==5.9.6
prometheus-client==0.19.0

# Development tools
black==23.11.0
isort==5.12.0
flake8==6.1.0
mypy==1.7.1
pre-commit==3.6.0

# Documentation generation
sphinx==7.2.6
sphinx-rtd-theme==1.3.0

# Book compilation
weasyprint==60.2
pypdf2==3.0.1
pandoc==2.3
"""


def get_pyproject_content() -> str:
    """Generate pyproject.toml content"""
    return """[build-system]
requires = ["setuptools>=45", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "from-flask-to-fastapi"
version = "1.0.0"
description = "A Journey of Async Programming - Complete migration guide from Flask to FastAPI"
authors = [
    {name = "Your Name", email = "your.email@example.com"},
]
license = {text = "CC BY-SA 4.0"}
readme = "README.md"
requires-python = ">=3.8"
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Topic :: Software Development :: Libraries :: Python Modules",
    "Topic :: Internet :: WWW/HTTP :: HTTP Servers",
]

[project.urls]
Homepage = "https://github.com/yourusername/from-flask-to-fastapi"
Documentation = "https://from-flask-to-fastapi.readthedocs.io/"
Repository = "https://github.com/yourusername/from-flask-to-fastapi"
Issues = "https://github.com/yourusername/from-flask-to-fastapi/issues"

[tool.black]
line-length = 88
target-version = ['py38']

[tool.isort]
profile = "black"
line_length = 88

[tool.mypy]
python_version = "3.8"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = "test_*.py"
python_functions = "test_*"
addopts = "-v --cov=src --cov-report=html --cov-report=term"
"""


def get_makefile_content() -> str:
    """Generate Makefile content"""
    return """# Makefile for "From Flask to FastAPI" book project

.PHONY: help install build validate clean serve examples lint test setup-dev book-stats

help:  ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \\033[36m%-15s\\033[0m %s\\n", $1, $2}'

install:  ## Install dependencies
	pip install -r requirements.txt
	pre-commit install

build:  ## Build the book in all formats
	python tools/build_scripts/build_book.py

validate:  ## Validate book content
	python tools/validation/validate_content.py

examples:  ## Run example applications
	@echo "Starting Flask example (port 5000)..."
	@python src/flask_examples/basic_app.py &
	@echo "Starting FastAPI example (port 8000)..."
	@python src/fastapi_examples/basic_app.py &
	@echo "Both examples running. Press Ctrl+C to stop."

async-demo:  ## Run async patterns demo
	python src/async_patterns/async_basics.py

clean:  ## Clean build artifacts
	rm -rf book_build/markdown/*.md
	rm -rf book_build/pdf/*.pdf
	rm -rf book_build/html/*.html
	rm -rf book_build/epub/*.epub
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -name "*.pyc" -delete

serve:  ## Serve HTML version locally
	@echo "Starting local server for HTML version..."
	@cd book_build/html && python -m http.server 8080

lint:  ## Run code linting
	black src/ tests/ tools/
	isort src/ tests/ tools/
	flake8 src/ tests/ tools/

test:  ## Run tests
	pytest tests/

setup-dev:  ## Setup development environment
	python -m venv venv
	@echo "Activate with: source venv/bin/activate (Linux/Mac) or venv\\\\Scripts\\\\activate (Windows)"
	@echo "Then run: make install"

book-stats:  ## Show book statistics
	@echo "Book Statistics:"
	@echo "=================="
	@echo "Chapters: $(find chapters -name "README.md" | wc -l)"
	@echo "Code examples: $(find src -name "*.py" | wc -l)"
	@echo "Total lines: $(find chapters src -name "*.md" -o -name "*.py" | xargs wc -l | tail -1)"
	@echo "Assets: $(find assets -type f | wc -l)"

update-toc:  ## Update table of contents
	python tools/automation/update_toc.py

pdf:  ## Build PDF only
	python tools/build_scripts/build_book.py --format pdf

html:  ## Build HTML only
	python tools/build_scripts/build_book.py --format html

epub:  ## Build EPUB only
	python tools/build_scripts/build_book.py --format epub
"""


def get_chapter_readme(chapter_num: int, title: str) -> str:
    """Generate chapter README template"""
    return f"""# Chapter {chapter_num}: {title}

## Overview

*Brief description of what this chapter covers*

## Learning Objectives

By the end of this chapter, you will:
- [ ] Objective 1
- [ ] Objective 2
- [ ] Objective 3

## Chapter Structure

### Section {chapter_num}.1: Introduction
*Introduction content*

### Section {chapter_num}.2: Core Concepts
*Core concepts content*

### Section {chapter_num}.3: Implementation
*Implementation details*

### Section {chapter_num}.4: Best Practices
*Best practices and recommendations*

### Section {chapter_num}.5: Real-World Examples
*Practical examples and case studies*

## Code Examples

- [Example 1](../../src/examples/chapter_{chapter_num:02d}_example_1.py)
- [Example 2](../../src/examples/chapter_{chapter_num:02d}_example_2.py)

## Diagrams and Assets

- [Architecture Diagram](../../assets/diagrams/chapter_{chapter_num:02d}_architecture.md)
- [Flow Chart](../../assets/diagrams/chapter_{chapter_num:02d}_flow.md)

## Key Takeaways

- Key point 1
- Key point 2  
- Key point 3

## Exercises

1. Exercise 1 description
2. Exercise 2 description

## Further Reading

- [Resource 1](link)
- [Resource 2](link)

## Next Steps

Continue to [Chapter {chapter_num + 1}](../chapters/{chapter_num + 1:02d}_next_chapter/README.md)

---

*Last updated: {datetime.now().strftime('%Y-%m-%d')}*
"""


def get_chapter_template() -> str:
    """Generate chapter markdown template"""
    return """# Chapter X: Chapter Title

## Learning Objectives
- [ ] Learning objective 1
- [ ] Learning objective 2
- [ ] Learning objective 3

## Introduction

Brief introduction to the chapter topic.

## Section X.1: Section Title

Content for this section.

### Subsection X.1.1: Subsection Title

Detailed content.

```python
# Code example
def example_function():
    \"\"\"Example function demonstrating a concept\"\"\"
    pass
```

## Section X.2: Another Section

More content.

### Code Examples

Reference to relevant code examples:

```python
# Include actual working code here
async def async_example():
    \"\"\"Async example function\"\"\"
    await some_async_operation()
    return "result"
```

## Section X.3: Best Practices

- Best practice 1
- Best practice 2
- Best practice 3

## Key Takeaways

- Important point 1
- Important point 2
- Important point 3

## Exercises

1. Exercise 1 description with clear instructions
2. Exercise 2 description with expected outcomes

## Further Reading

- [Resource 1](link) - Description of resource
- [Resource 2](link) - Description of resource

---

*Chapter X | Last updated: {datetime.now().strftime('%Y-%m-%d')}*
"""


def get_code_template() -> str:
    """Generate code example template"""
    return '''"""
Code Example Template
Chapter X: Chapter Title

Description: Brief description of what this code demonstrates

Learning objectives:
- Objective 1
- Objective 2

Usage:
    python code_example_template.py
"""

import asyncio
from typing import List, Optional, Dict, Any


class ExampleClass:
    """Example class demonstrating patterns"""
    
    def __init__(self, name: str):
        self.name = name
        self.data: Dict[str, Any] = {}
    
    async def async_method(self) -> str:
        """Example async method"""
        await asyncio.sleep(0.1)  # Simulate async operation
        return f"Hello from {self.name}"
    
    def sync_method(self) -> str:
        """Example synchronous method for comparison"""
        return f"Sync hello from {self.name}"


async def demonstrate_async_patterns():
    """Demonstrate async programming patterns"""
    print("Async Patterns Demonstration")
    print("=" * 30)
    
    # Create example instances
    examples = [
        ExampleClass("FastAPI"),
        ExampleClass("AsyncIO"),
        ExampleClass("Pydantic")
    ]
    
    # Sequential execution
    print("\\nSequential execution:")
    for example in examples:
        result = await example.async_method()
        print(f"  {result}")
    
    # Concurrent execution
    print("\\nConcurrent execution:")
    tasks = [example.async_method() for example in examples]
    results = await asyncio.gather(*tasks)
    for result in results:
        print(f"  {result}")


def demonstrate_sync_patterns():
    """Demonstrate synchronous patterns for comparison"""
    print("\\nSync Patterns Demonstration")
    print("=" * 30)
    
    examples = [
        ExampleClass("Flask"),
        ExampleClass("Django"),
        ExampleClass("Traditional")
    ]
    
    for example in examples:
        result = example.sync_method()
        print(f"  {result}")


async def main():
    """Main function demonstrating usage"""
    print("Code Example: Async vs Sync Patterns")
    print("=" * 40)
    
    # Demonstrate both patterns
    await demonstrate_async_patterns()
    demonstrate_sync_patterns()
    
    print("\\n✅ Example completed successfully!")


if __name__ == "__main__":
    asyncio.run(main())
'''