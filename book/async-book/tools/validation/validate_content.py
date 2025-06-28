#!/usr/bin/env python3
"""
Content Validation Script
Validates book content for consistency and quality
"""

import re
import click
from pathlib import Path
from typing import List, Tuple, Dict


class ContentValidator:
    """Validates book content"""
    
    def __init__(self, book_root: Path):
        self.book_root = book_root
        self.chapters_dir = book_root / "chapters"
        self.src_dir = book_root / "src"
        self.errors: List[str] = []
        self.warnings: List[str] = []
    
    def validate_all(self) -> bool:
        """Run all validations"""
        click.echo("🔍 Validating book content...")
        
        self.validate_chapters()
        self.validate_code_examples()
        self.validate_links()
        self.validate_structure()
        
        return len(self.errors) == 0
    
    def validate_chapters(self) -> bool:
        """Validate all chapters"""
        click.echo("  📖 Validating chapters...")
        
        expected_chapters = [f"{i:02d}_" for i in range(1, 11)]
        found_chapters = []
        
        for chapter_dir in sorted(self.chapters_dir.iterdir()):
            if chapter_dir.is_dir() and not chapter_dir.name.startswith('.'):
                found_chapters.append(chapter_dir.name[:3])
                self._validate_chapter(chapter_dir)
        
        # Check for missing chapters
        for expected in expected_chapters:
            if expected not in found_chapters:
                self.errors.append(f"Missing chapter: {expected}")
        
        return len(self.errors) == 0
    
    def validate_code_examples(self) -> bool:
        """Validate code examples"""
        click.echo("  🐍 Validating code examples...")
        
        if not self.src_dir.exists():
            self.warnings.append("No src directory found")
            return True
        
        python_files = list(self.src_dir.rglob("*.py"))
        
        for py_file in python_files:
            try:
                # Basic syntax check
                compile(py_file.read_text(encoding='utf-8'), py_file, 'exec')
            except SyntaxError as e:
                self.errors.append(f"Syntax error in {py_file}: {e}")
            except UnicodeDecodeError as e:
                self.errors.append(f"Encoding error in {py_file}: {e}")
        
        return len(self.errors) == 0
    
    def validate_links(self) -> bool:
        """Validate internal links"""
        click.echo("  🔗 Validating links...")
        
        for chapter_dir in self.chapters_dir.iterdir():
            if chapter_dir.is_dir():
                readme_file = chapter_dir / "README.md"
                if readme_file.exists():
                    self._validate_chapter_links(readme_file, chapter_dir)
        
        return len(self.errors) == 0
    
    def validate_structure(self) -> bool:
        """Validate project structure"""
        click.echo("  📁 Validating structure...")
        
        required_dirs = [
            "chapters", "src", "assets", "book_build", 
            "research", "templates", "tools"
        ]
        
        for req_dir in required_dirs:
            if not (self.book_root / req_dir).exists():
                self.warnings.append(f"Missing recommended directory: {req_dir}")
        
        return True
    
    def _validate_chapter(self, chapter_dir: Path):
        """Validate individual chapter"""
        readme_file = chapter_dir / "README.md"
        
        if not readme_file.exists():
            self.errors.append(f"Missing README.md in {chapter_dir.name}")
            return
        
        try:
            content = readme_file.read_text(encoding='utf-8')
        except UnicodeDecodeError:
            self.errors.append(f"Encoding error in {readme_file}")
            return
        
        # Check for required sections
        required_sections = [
            "Learning Objectives", 
            "Overview",
            "Key Takeaways"
        ]
        
        for section in required_sections:
            if section not in content:
                self.warnings.append(f"{chapter_dir.name}: Missing '{section}' section")
        
        # Check for empty content
        if len(content.strip()) < 100:
            self.warnings.append(f"{chapter_dir.name}: Chapter appears to be very short")
    
    def _validate_chapter_links(self, readme_file: Path, chapter_dir: Path):
        """Validate links in a chapter"""
        try:
            content = readme_file.read_text(encoding='utf-8')
        except UnicodeDecodeError:
            return
        
        # Find markdown links
        links = re.findall(r'\[([^\]]+)\]\(([^\)]+)\)', content)
        
        for link_text, link_url in links:
            if link_url.startswith(('../', './')):
                # Validate internal links
                if link_url.startswith('../'):
                    link_path = chapter_dir.parent / link_url[3:]
                else:
                    link_path = chapter_dir / link_url[2:]
                
                if not link_path.exists():
                    self.errors.append(f"{chapter_dir.name}: Broken link '{link_url}'")
    
    def print_results(self):
        """Print validation results"""
        if self.errors:
            click.echo(f"\n❌ {len(self.errors)} errors found:")
            for error in self.errors:
                click.echo(f"  • {error}")
        
        if self.warnings:
            click.echo(f"\n⚠️  {len(self.warnings)} warnings:")
            for warning in self.warnings:
                click.echo(f"  • {warning}")
        
        if not self.errors and not self.warnings:
            click.echo("\n✅ All validations passed!")
        elif not self.errors:
            click.echo("\n✅ No errors found (warnings can be addressed)")
        
        return len(self.errors) == 0


@click.command()
@click.option('--book-root', default='.', help='Book root directory')
@click.option('--fix', is_flag=True, help='Attempt to fix some issues automatically')
def main(book_root, fix):
    """Validate book content for consistency and quality"""
    
    book_root = Path(book_root)
    validator = ContentValidator(book_root)
    
    success = validator.validate_all()
    validator.print_results()
    
    if not success:
        click.echo("\n💡 Run with --fix to attempt automatic fixes", err=True)
        exit(1)
    else:
        click.echo("\n🎉 Content validation completed successfully!")


if __name__ == "__main__":
    main()
