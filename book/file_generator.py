# File: file_generator.py
"""
Initial File Generator
Handles creation of initial files and templates for the book project
"""

from pathlib import Path
from typing import Dict, List
from datetime import datetime
import click

from file_templates import (
    get_readme_content,
    get_gitignore_content,
    get_requirements_content,
    get_pyproject_content,
    get_makefile_content,
    get_chapter_readme,
    get_chapter_template,
    get_code_template
)

from build_scripts import (
    get_build_script,
    get_validation_script,
    get_publish_script
)

from example_files import (
    get_flask_example,
    get_fastapi_example,
    get_async_patterns_example,
    get_migration_example
)


class InitialFileGenerator:
    """Generates initial files and templates for the book project"""
    
    def __init__(self, book_root: Path):
        self.book_root = book_root
        self.files_to_create = self._get_files_structure()
    
    def _get_files_structure(self) -> Dict[str, str]:
        """Define all files to be created with their content"""
        files = {}
        
        # Root configuration files
        files.update(self._get_root_files())
        
        # Chapter templates
        files.update(self._get_chapter_files())
        
        # Template files
        files.update(self._get_template_files())
        
        # Build and tool scripts
        files.update(self._get_tool_files())
        
        # Example code files
        files.update(self._get_example_files())
        
        # Documentation files
        files.update(self._get_documentation_files())
        
        return files
    
    def _get_root_files(self) -> Dict[str, str]:
        """Get root-level configuration files"""
        return {
            "README.md": get_readme_content(),
            ".gitignore": get_gitignore_content(),
            "requirements.txt": get_requirements_content(),
            "pyproject.toml": get_pyproject_content(),
            "Makefile": get_makefile_content(),
            "LICENSE": self._get_license_content(),
            ".pre-commit-config.yaml": self._get_precommit_config(),
        }
    
    def _get_chapter_files(self) -> Dict[str, str]:
        """Get chapter README files"""
        chapters = [
            (1, "Executive Summary & Business Case"),
            (2, "Technical Architecture & Design"),
            (3, "Database Layer Migration"),
            (4, "API Endpoint Conversion"),
            (5, "Authentication & Authorization"),
            (6, "Background Tasks & Job Processing"),
            (7, "Testing & Quality Assurance"),
            (8, "Performance Optimization"),
            (9, "Security & Compliance"),
            (10, "Production Deployment and Operations"),
        ]
        
        chapter_files = {}
        for num, title in chapters:
            path = f"chapters/{num:02d}_{title.lower().replace(' ', '_').replace('&', 'and')}/README.md"
            chapter_files[path] = get_chapter_readme(num, title)
        
        return chapter_files
    
    def _get_template_files(self) -> Dict[str, str]:
        """Get template files"""
        return {
            "templates/chapters/chapter_template.md": get_chapter_template(),
            "templates/code/code_example_template.py": get_code_template(),
            "templates/diagrams/diagram_template.md": self._get_diagram_template(),
        }
    
    def _get_tool_files(self) -> Dict[str, str]:
        """Get build and automation scripts"""
        return {
            "tools/build_scripts/build_book.py": get_build_script(),
            "tools/validation/validate_content.py": get_validation_script(),
            "tools/publishing/publish_book.py": get_publish_script(),
            "tools/automation/update_toc.py": self._get_toc_script(),
        }
    
    def _get_example_files(self) -> Dict[str, str]:
        """Get example code files"""
        return {
            "src/flask_examples/basic_app.py": get_flask_example(),
            "src/fastapi_examples/basic_app.py": get_fastapi_example(),
            "src/async_patterns/async_basics.py": get_async_patterns_example(),
            "src/migration_tools/migration_helper.py": get_migration_example(),
            "tests/examples/test_flask_app.py": self._get_test_example(),
        }
    
    def _get_documentation_files(self) -> Dict[str, str]:
        """Get documentation files"""
        return {
            "docs/development/setup.md": self._get_development_docs(),
            "docs/publishing/publishing_guide.md": self._get_publishing_docs(),
            "docs/contributing/contribution_guide.md": self._get_contributing_docs(),
        }
    
    def create_initial_files(self, verbose: bool = False) -> None:
        """Create all initial files"""
        created_count = 0
        
        for file_path, content in self.files_to_create.items():
            full_path = self.book_root / file_path
            
            # Create parent directories if they don't exist
            full_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write file content
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            if verbose:
                click.echo(f"  ✓ {file_path}")
            created_count += 1
        
        if not verbose:
            click.echo(f"✅ Created {created_count} initial files")
    
    def show_files(self) -> None:
        """Display the files that would be created"""
        categories = {
            "Configuration": [],
            "Chapters": [],
            "Templates": [],
            "Tools": [],
            "Examples": [],
            "Documentation": [],
        }
        
        for file_path in self.files_to_create.keys():
            if file_path.startswith("chapters/"):
                categories["Chapters"].append(file_path)
            elif file_path.startswith("templates/"):
                categories["Templates"].append(file_path)
            elif file_path.startswith("tools/"):
                categories["Tools"].append(file_path)
            elif file_path.startswith("src/") or file_path.startswith("tests/"):
                categories["Examples"].append(file_path)
            elif file_path.startswith("docs/"):
                categories["Documentation"].append(file_path)
            else:
                categories["Configuration"].append(file_path)
        
        for category, files in categories.items():
            if files:
                click.echo(f"{category}:")
                for file_path in sorted(files):
                    click.echo(f"  • {file_path}")
                click.echo()
    
    def get_file_count(self) -> int:
        """Get total number of files that will be created"""
        return len(self.files_to_create)
    
    # Helper methods for file content generation
    def _get_license_content(self) -> str:
        """Generate license content"""
        return """Creative Commons Attribution-ShareAlike 4.0 International License

Copyright (c) 2024 Your Name

This work is licensed under a Creative Commons Attribution-ShareAlike 4.0 
International License. You should have received a copy of the license along 
with this work. If not, see <http://creativecommons.org/licenses/by-sa/4.0/>.

You are free to:
- Share — copy and redistribute the material in any medium or format
- Adapt — remix, transform, and build upon the material for any purpose, 
  even commercially.

Under the following terms:
- Attribution — You must give appropriate credit, provide a link to the 
  license, and indicate if changes were made.
- ShareAlike — If you remix, transform, or build upon the material, you 
  must distribute your contributions under the same license as the original.
"""
    
    def _get_precommit_config(self) -> str:
        """Generate pre-commit configuration"""
        return """repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
      - id: check-merge-conflict
  
  - repo: https://github.com/psf/black
    rev: 22.12.0
    hooks:
      - id: black
        language_version: python3
  
  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort
        args: ["--profile", "black"]
  
  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8
        additional_dependencies: [flake8-docstrings]
"""
    
    def _get_diagram_template(self) -> str:
        """Generate diagram template"""
        return """# Diagram Template

## Diagram Title

### Purpose
Brief description of what this diagram illustrates.

### Mermaid Diagram
```mermaid
graph TD
    A[Start] --> B[Process]
    B --> C[End]
```

### Description
Detailed explanation of the diagram components and flow.

### Usage in Book
Reference to which chapter/section this diagram belongs to.
"""
    
    def _get_toc_script(self) -> str:
        """Generate table of contents update script"""
        return '''#!/usr/bin/env python3
"""
Table of Contents Update Script
Automatically updates the table of contents in README files
"""

import re
from pathlib import Path


def update_table_of_contents(book_root: Path):
    """Update table of contents in main README"""
    chapters_dir = book_root / "chapters"
    readme_path = book_root / "README.md"
    
    if not readme_path.exists():
        return
    
    # Read current README
    content = readme_path.read_text()
    
    # Generate new TOC
    toc_lines = ["## 📚 Book Structure", ""]
    
    for chapter_dir in sorted(chapters_dir.iterdir()):
        if chapter_dir.is_dir() and chapter_dir.name.startswith(tuple("0123456789")):
            chapter_readme = chapter_dir / "README.md"
            if chapter_readme.exists():
                first_line = chapter_readme.read_text().split('\\n')[0]
                chapter_title = first_line.replace('# ', '')
                toc_lines.append(f"- [{chapter_title}](chapters/{chapter_dir.name}/README.md)")
    
    toc_content = "\\n".join(toc_lines)
    
    # Replace existing TOC
    pattern = r"## 📚 Book Structure.*?(?=##|\Z)"
    new_content = re.sub(pattern, toc_content + "\\n\\n", content, flags=re.DOTALL)
    
    # Write updated README
    readme_path.write_text(new_content)
    print("✅ Table of contents updated")


if __name__ == "__main__":
    book_root = Path(".")
    update_table_of_contents(book_root)
'''
    
    def _get_test_example(self) -> str:
        """Generate test example"""
        return '''"""
Example test file for Flask application
"""

import pytest
from src.flask_examples.basic_app import app, db, User


@pytest.fixture
def client():
    """Test client fixture"""
    app.config['TESTING'] = True
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'
    
    with app.test_client() as client:
        with app.app_context():
            db.create_all()
            yield client


def test_home_endpoint(client):
    """Test home endpoint"""
    response = client.get('/')
    assert response.status_code == 200
    assert b'Flask Application' in response.data


def test_get_users_empty(client):
    """Test getting users when none exist"""
    response = client.get('/users')
    assert response.status_code == 200
    data = response.get_json()
    assert data['count'] == 0
    assert data['users'] == []
'''
    
    def _get_development_docs(self) -> str:
        """Generate development documentation"""
        return """# Development Setup Guide

## Prerequisites

- Python 3.8+
- Git
- Make (optional but recommended)

## Quick Start

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd async-book
   ```

2. Create virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\\Scripts\\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Install pre-commit hooks:
   ```bash
   pre-commit install
   ```

## Development Workflow

1. Create a new branch for your changes
2. Make your changes
3. Run tests: `make test`
4. Validate content: `make validate`
5. Build book: `make build`
6. Commit and push changes

## Project Structure

See the main README for detailed project structure information.
"""
    
    def _get_publishing_docs(self) -> str:
        """Generate publishing documentation"""
        return """# Publishing Guide

## Overview

This guide covers the process of publishing the book in various formats.

## Supported Formats

- **Markdown**: Single file compilation
- **PDF**: Print-ready format via LaTeX/Pandoc
- **EPUB**: E-book format for e-readers
- **HTML**: Web-friendly format

## Building the Book

### All Formats
```bash
make build
```

### Specific Format
```bash
python tools/build_scripts/build_book.py --format pdf
```

## Publishing Checklist

- [ ] All chapters completed
- [ ] Content validation passed
- [ ] Code examples tested
- [ ] Images and diagrams included
- [ ] Table of contents updated
- [ ] Final review completed

## Distribution

Details on how to distribute the completed book.
"""
    
    def _get_contributing_docs(self) -> str:
        """Generate contributing documentation"""
        return """# Contributing Guide

## Welcome Contributors!

We welcome contributions to make this book better for everyone.

## How to Contribute

### Reporting Issues
- Use GitHub issues for bugs or suggestions
- Provide clear description and steps to reproduce
- Include relevant code examples

### Contributing Content
- Fork the repository
- Create a feature branch
- Make your changes
- Submit a pull request

### Code Examples
- Ensure all code examples work
- Include tests where appropriate
- Follow coding standards (black, isort, flake8)

### Writing Guidelines
- Use clear, concise language
- Include practical examples
- Maintain consistent formatting
- Add diagrams where helpful

## Review Process

1. Submit pull request
2. Automated checks run
3. Manual review by maintainers
4. Address feedback if needed
5. Merge when approved

## Recognition

Contributors will be acknowledged in the book credits.
"""