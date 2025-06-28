# File: structure_generator.py
"""
Directory Structure Generator
Handles creation of the complete directory structure for the book project
"""

from pathlib import Path
from typing import List, Dict
import click


class DirectoryStructureGenerator:
    """Generates the complete directory structure for the book project"""
    
    def __init__(self, book_root: Path):
        self.book_root = book_root
        self.directories = self._get_directory_structure()
    
    def _get_directory_structure(self) -> List[str]:
        """Define the complete directory structure"""
        return [
            # Source code and examples
            "src/flask_examples",
            "src/fastapi_examples", 
            "src/async_patterns",
            "src/security",
            "src/utils",
            "src/migration_tools",
            "src/performance_tests",
            
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
            "assets/screenshots",
            
            # Book build outputs
            "book_build/markdown",
            "book_build/pdf",
            "book_build/epub", 
            "book_build/html",
            "book_build/latex",
            
            # Research and supporting materials
            "research/benchmarks",
            "research/case_studies",
            "research/industry_data",
            "research/references",
            "research/surveys",
            
            # Templates and scaffolding
            "templates/chapters",
            "templates/code",
            "templates/diagrams",
            
            # Development and build tools
            "tools/build_scripts",
            "tools/validation",
            "tools/publishing",
            "tools/automation",
            
            # Documentation
            "docs/development",
            "docs/publishing",
            "docs/contributing",
            
            # Configuration
            "config/build",
            "config/publishing",
            "config/development",
            
            # Tests for book code examples
            "tests/unit",
            "tests/integration",
            "tests/examples",
        ]
    
    def create_directories(self, verbose: bool = False) -> None:
        """Create all directories in the structure"""
        
        # Create root directory
        self.book_root.mkdir(parents=True, exist_ok=True)
        if verbose:
            click.echo(f"📁 Created root: {self.book_root}")
        
        # Create all subdirectories
        created_count = 0
        for directory in self.directories:
            dir_path = self.book_root / directory
            dir_path.mkdir(parents=True, exist_ok=True)
            
            if verbose:
                click.echo(f"  ✓ {directory}")
            created_count += 1
        
        if not verbose:
            click.echo(f"✅ Created {created_count} directories")
    
    def show_structure(self) -> None:
        """Display the directory structure that would be created"""
        click.echo("📁 Directory structure:")
        click.echo(f"async-book/")
        
        # Group directories by category
        categories = self._group_directories_by_category()
        
        for category, dirs in categories.items():
            click.echo(f"├── {category}/")
            for i, directory in enumerate(dirs):
                prefix = "│   ├──" if i < len(dirs) - 1 else "│   └──"
                # Remove the category prefix from directory name
                dir_name = directory.replace(f"{category}/", "", 1)
                click.echo(f"{prefix} {dir_name}/")
            click.echo("│")
    
    def _group_directories_by_category(self) -> Dict[str, List[str]]:
        """Group directories by their top-level category"""
        categories = {}
        
        for directory in self.directories:
            parts = directory.split('/')
            if len(parts) >= 2:
                category = parts[0]
                if category not in categories:
                    categories[category] = []
                categories[category].append(directory)
        
        # Sort categories and directories
        for category in categories:
            categories[category].sort()
        
        return dict(sorted(categories.items()))
    
    def get_directory_count(self) -> int:
        """Get total number of directories that will be created"""
        return len(self.directories) + 1  # +1 for root directory
    
    def validate_structure(self) -> bool:
        """Validate that all directories were created successfully"""
        missing_dirs = []
        
        for directory in self.directories:
            dir_path = self.book_root / directory
            if not dir_path.exists():
                missing_dirs.append(directory)
        
        if missing_dirs:
            click.echo(f"❌ Missing directories: {missing_dirs}", err=True)
            return False
        
        return True