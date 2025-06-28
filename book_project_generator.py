#!/usr/bin/env python3
"""
Book Project Structure Generator
Creates the complete directory structure for "From Flask to FastAPI: A Journey of Async Programming"

Usage:
    python create_book_structure.py [--path /custom/path]
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime


class BookStructureGenerator:
    """Generates the complete book project structure"""
    
    def __init__(self, base_path: str = "."):
        self.base_path = Path(base_path)
        self.book_root = self.base_path / "async-book"
        
    def create_directory_structure(self):
        """Create the complete directory structure"""
        
        # Main directory structure
        directories = [
            # Source code and examples
            "src/flask_examples",
            "src/fastapi_examples", 
            "src/async_patterns",
            "src/security",
            "src/utils",
            
            # Book chapters
            "chapters/01_executive_summary",
            "chapters/02_technical_architecture", 
            "chapters/03_database_migration",
            "chapters/04_api_conversion",
            "chapters/05_authentication",
            "chapters/06_background_tasks",
            "chapters/07_testing_qa",
            "chapters/08_performance",
            "chapters/09_security_compliance",
            "chapters/10_production_operations",
            "chapters/appendices",
            
            # Book assets
            "assets/diagrams",
            "assets/images", 
            "assets/charts",
            "assets/code_snippets",
            
            # Book build outputs
            "book_build/markdown",
            "book_build/pdf",
            "book_build/epub", 
            "book_build/html",
            
            # Research and supporting materials
            "research/benchmarks",
            "research/case_studies",
            "research/industry_data",
            "research/references",
            
            # Templates
            "templates",
            
            # Tools
            "tools/build_scripts",
            "tools/validation",
            "tools/publishing"
        ]
        
        print(f"Creating book project structure at: {self.book_root}")
        
        # Create root directory
        self.book_root.mkdir(exist_ok=True)
        
        # Create all directories
        for directory in directories:
            dir_path = self.book_root / directory
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"✓ Created: {directory}")
            
        print(f"\n📁 Directory structure created successfully!")
        
    def create_initial_files(self):
        """Create initial files and templates"""
        
        files_to_create = {
            # Root files
            "README.md": self.get_readme_content(),
            ".gitignore": self.get_gitignore_content(),
            "requirements.txt": self.get_requirements_content(),
            "pyproject.toml": self.get_pyproject_content(),
            
            # Chapter templates
            "chapters/01_executive_summary/README.md": self.get_chapter_readme(1, "Executive Summary & Business Case"),
            "chapters/02_technical_architecture/README.md": self.get_chapter_readme(2, "Technical Architecture & Design"),
            "chapters/03_database_migration/README.md": self.get_chapter_readme(3, "Database Layer Migration"), 
            "chapters/04_api_conversion/README.md": self.get_chapter_readme(4, "API Endpoint Conversion"),
            "chapters/05_authentication/README.md": self.get_chapter_readme(5, "Authentication & Authorization"),
            "chapters/06_background_tasks/README.md": self.get_chapter_readme(6, "Background Tasks & Job Processing"),
            "chapters/07_testing_qa/README.md": self.get_chapter_readme(7, "Testing & Quality Assurance"),
            "chapters/08_performance/README.md": self.get_chapter_readme(8, "Performance Optimization"),
            "chapters/09_security_compliance/README.md": self.get_chapter_readme(9, "Security & Compliance"),
            "chapters/10_production_operations/README.md": self.get_chapter_readme(10, "Production Deployment and Operations"),
            
            # Templates
            "templates/chapter_template.md": self.get_chapter_template(),
            "templates/code_example_template.py": self.get_code_template(),
            
            # Build scripts
            "tools/build_scripts/build_book.py": self.get_build_script(),
            "tools/build_scripts/validate_content.py": self.get_validation_script(),
            
            # Example files
            "src/flask_examples/basic_app.py": self.get_flask_example(),
            "src/fastapi_examples/basic_app.py": self.get_fastapi_example(),
            "src/async_patterns/async_basics.py": self.get_async_patterns_example(),
        }
        
        print("\n📝 Creating initial files...")
        
        for file_path, content in files_to_create.items():
            full_path = self.book_root / file_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(content)
                
            print(f"✓ Created: {file_path}")
            
        print(f"\n📚 Initial files created successfully!")
        
    def get_readme_content(self):
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

## 📁 Project Structure

```
async-book/
├── chapters/           # Book chapters in markdown
├── src/               # Source code examples
├── assets/            # Diagrams, images, charts
├── book_build/        # Compiled book outputs
├── research/          # Supporting research
├── templates/         # Content templates
└── tools/            # Build and publishing tools
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

*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""

    def get_gitignore_content(self):
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
"""

    def get_requirements_content(self):
        """Generate requirements.txt content"""
        return """# Book building and publishing
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
"""

    def get_pyproject_content(self):
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
"""

    def get_chapter_readme(self, chapter_num: int, title: str):
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

- [Example 1](../src/examples/chapter_{chapter_num:02d}_example_1.py)
- [Example 2](../src/examples/chapter_{chapter_num:02d}_example_2.py)

## Key Takeaways

- Key point 1
- Key point 2  
- Key point 3

## Next Steps

Continue to [Chapter {chapter_num + 1}](../chapters/{chapter_num + 1:02d}_next_chapter/README.md)

---

*Last updated: {datetime.now().strftime('%Y-%m-%d')}*
"""

    def get_chapter_template(self):
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
    pass
```

## Section X.2: Another Section

More content.

## Key Takeaways

- Important point 1
- Important point 2
- Important point 3

## Exercises

1. Exercise 1 description
2. Exercise 2 description

## Further Reading

- [Resource 1](link)
- [Resource 2](link)

---

*Chapter X | Page X of XXX*
"""

    def get_code_template(self):
        """Generate code example template"""
        return '''"""
Code Example Template
Chapter X: Chapter Title

Description: Brief description of what this code demonstrates

Usage:
    python code_example_template.py
"""

import asyncio
from typing import List, Optional


class ExampleClass:
    """Example class demonstrating patterns"""
    
    def __init__(self, name: str):
        self.name = name
    
    async def async_method(self) -> str:
        """Example async method"""
        await asyncio.sleep(0.1)
        return f"Hello from {self.name}"


async def main():
    """Main function demonstrating usage"""
    example = ExampleClass("FastAPI")
    result = await example.async_method()
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
'''

    def get_build_script(self):
        """Generate book build script"""
        return '''#!/usr/bin/env python3
"""
Book Build Script
Compiles markdown chapters into various book formats
"""

import argparse
import os
import shutil
from pathlib import Path
import subprocess


class BookBuilder:
    """Builds the book in various formats"""
    
    def __init__(self, book_root: Path):
        self.book_root = book_root
        self.chapters_dir = book_root / "chapters"
        self.build_dir = book_root / "book_build"
        
    def build_markdown(self):
        """Compile all chapters into single markdown file"""
        print("📝 Building markdown version...")
        
        output_file = self.build_dir / "markdown" / "complete_book.md"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as outfile:
            # Write title page
            outfile.write("# From Flask to FastAPI: A Journey of Async Programming\\n\\n")
            
            # Combine all chapters
            for chapter_dir in sorted(self.chapters_dir.iterdir()):
                if chapter_dir.is_dir() and chapter_dir.name.startswith(('01_', '02_', '03_', '04_', '05_', '06_', '07_', '08_', '09_', '10_')):
                    readme_file = chapter_dir / "README.md"
                    if readme_file.exists():
                        with open(readme_file, 'r') as infile:
                            outfile.write(infile.read())
                            outfile.write("\\n\\n---\\n\\n")
        
        print(f"✅ Markdown built: {output_file}")
        
    def build_pdf(self):
        """Build PDF version using pandoc"""
        print("📄 Building PDF version...")
        
        markdown_file = self.build_dir / "markdown" / "complete_book.md"
        pdf_file = self.build_dir / "pdf" / "from_flask_to_fastapi.pdf"
        pdf_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            subprocess.run([
                "pandoc", str(markdown_file),
                "-o", str(pdf_file),
                "--pdf-engine=weasyprint",
                "--toc",
                "--toc-depth=3"
            ], check=True)
            print(f"✅ PDF built: {pdf_file}")
        except subprocess.CalledProcessError:
            print("❌ PDF build failed - ensure pandoc and weasyprint are installed")
        except FileNotFoundError:
            print("❌ pandoc not found - install pandoc to build PDF")
    
    def build_html(self):
        """Build HTML version"""
        print("🌐 Building HTML version...")
        
        # Simple HTML build - could be enhanced with mkdocs
        markdown_file = self.build_dir / "markdown" / "complete_book.md"
        html_file = self.build_dir / "html" / "index.html"
        html_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            subprocess.run([
                "pandoc", str(markdown_file),
                "-o", str(html_file),
                "--standalone",
                "--toc",
                "--css=style.css"
            ], check=True)
            print(f"✅ HTML built: {html_file}")
        except subprocess.CalledProcessError:
            print("❌ HTML build failed")
        except FileNotFoundError:
            print("❌ pandoc not found - install pandoc to build HTML")
    
    def build_all(self):
        """Build all formats"""
        self.build_markdown()
        self.build_pdf() 
        self.build_html()


def main():
    parser = argparse.ArgumentParser(description="Build the FastAPI book")
    parser.add_argument("--format", choices=["markdown", "pdf", "html", "all"], 
                       default="all", help="Format to build")
    parser.add_argument("--book-root", default=".", help="Book root directory")
    
    args = parser.parse_args()
    
    book_root = Path(args.book_root)
    builder = BookBuilder(book_root)
    
    if args.format == "markdown":
        builder.build_markdown()
    elif args.format == "pdf":
        builder.build_pdf()
    elif args.format == "html":
        builder.build_html()
    else:
        builder.build_all()


if __name__ == "__main__":
    main()
'''

    def get_validation_script(self):
        """Generate content validation script"""
        return '''#!/usr/bin/env python3
"""
Content Validation Script
Validates book content for consistency and quality
"""

import re
from pathlib import Path
from typing import List, Tuple


class ContentValidator:
    """Validates book content"""
    
    def __init__(self, book_root: Path):
        self.book_root = book_root
        self.chapters_dir = book_root / "chapters"
        self.errors = []
        self.warnings = []
    
    def validate_chapters(self) -> bool:
        """Validate all chapters"""
        print("🔍 Validating chapters...")
        
        for chapter_dir in sorted(self.chapters_dir.iterdir()):
            if chapter_dir.is_dir() and not chapter_dir.name.startswith('.'):
                self.validate_chapter(chapter_dir)
        
        return len(self.errors) == 0
    
    def validate_chapter(self, chapter_dir: Path):
        """Validate individual chapter"""
        readme_file = chapter_dir / "README.md"
        
        if not readme_file.exists():
            self.errors.append(f"Missing README.md in {chapter_dir.name}")
            return
        
        content = readme_file.read_text()
        
        # Check for required sections
        required_sections = ["Learning Objectives", "Key Takeaways"]
        for section in required_sections:
            if section not in content:
                self.warnings.append(f"{chapter_dir.name}: Missing '{section}' section")
        
        # Check for broken links
        links = re.findall(r'\\[([^\\]]+)\\]\\(([^\\)]+)\\)', content)
        for link_text, link_url in links:
            if link_url.startswith('../') or link_url.startswith('./'):
                # Validate internal links
                link_path = chapter_dir / link_url
                if not link_path.exists():
                    self.errors.append(f"{chapter_dir.name}: Broken link '{link_url}'")
    
    def validate_code_examples(self) -> bool:
        """Validate code examples"""
        print("🔍 Validating code examples...")
        
        src_dir = self.book_root / "src"
        if not src_dir.exists():
            self.warnings.append("No src directory found")
            return True
        
        python_files = list(src_dir.rglob("*.py"))
        
        for py_file in python_files:
            try:
                # Basic syntax check
                compile(py_file.read_text(), py_file, 'exec')
            except SyntaxError as e:
                self.errors.append(f"Syntax error in {py_file}: {e}")
        
        return len(self.errors) == 0
    
    def print_results(self):
        """Print validation results"""
        if self.errors:
            print(f"\\n❌ {len(self.errors)} errors found:")
            for error in self.errors:
                print(f"  • {error}")
        
        if self.warnings:
            print(f"\\n⚠️  {len(self.warnings)} warnings:")
            for warning in self.warnings:
                print(f"  • {warning}")
        
        if not self.errors and not self.warnings:
            print("\\n✅ All validations passed!")
        elif not self.errors:
            print("\\n✅ No errors found (warnings can be addressed)")


def main():
    book_root = Path(".")
    validator = ContentValidator(book_root)
    
    chapters_valid = validator.validate_chapters()
    code_valid = validator.validate_code_examples()
    
    validator.print_results()
    
    if not (chapters_valid and code_valid):
        exit(1)


if __name__ == "__main__":
    main()
'''

    def get_flask_example(self):
        """Generate Flask example"""
        return '''"""
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


@app.route('/users', methods=['POST'])
def create_user():
    """Create new user - synchronous operation"""
    data = request.get_json()
    
    if not data or 'name' not in data or 'email' not in data:
        return jsonify({"error": "Name and email required"}), 400
    
    # Simulate validation delay
    time.sleep(0.05)
    
    user = User(name=data['name'], email=data['email'])
    db.session.add(user)
    db.session.commit()
    
    return jsonify(user.to_dict()), 201


@app.route('/users/<int:user_id>', methods=['GET'])
def get_user(user_id):
    """Get single user - synchronous operation"""
    # Simulate database lookup delay
    time.sleep(0.05)
    
    user = User.query.get_or_404(user_id)
    return jsonify(user.to_dict())


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
'''