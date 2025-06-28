# File: build_scripts.py
"""
Build Scripts for Book Generation
Contains build script templates for compiling the book
"""


def get_build_script() -> str:
    """Generate book build script"""
    return '''#!/usr/bin/env python3
"""
Book Build Script
Compiles markdown chapters into various book formats
"""

import click
import shutil
from pathlib import Path
import subprocess
from datetime import datetime


class BookBuilder:
    """Builds the book in various formats"""
    
    def __init__(self, book_root: Path):
        self.book_root = book_root
        self.chapters_dir = book_root / "chapters"
        self.build_dir = book_root / "book_build"
        self.assets_dir = book_root / "assets"
        
    def build_markdown(self):
        """Compile all chapters into single markdown file"""
        click.echo("📝 Building markdown version...")
        
        output_file = self.build_dir / "markdown" / "complete_book.md"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as outfile:
            # Write title page
            outfile.write("# From Flask to FastAPI: A Journey of Async Programming\\n\\n")
            outfile.write(f"*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\\n\\n")
            outfile.write("---\\n\\n")
            
            # Add table of contents
            outfile.write("# Table of Contents\\n\\n")
            self._write_table_of_contents(outfile)
            outfile.write("\\n---\\n\\n")
            
            # Combine all chapters
            for chapter_dir in sorted(self.chapters_dir.iterdir()):
                if self._is_chapter_dir(chapter_dir):
                    readme_file = chapter_dir / "README.md"
                    if readme_file.exists():
                        outfile.write(f"\\n\\n# {chapter_dir.name.replace('_', ' ').title()}\\n\\n")
                        
                        content = readme_file.read_text(encoding='utf-8')
                        # Process content to fix relative links
                        processed_content = self._process_chapter_content(content, chapter_dir)
                        outfile.write(processed_content)
                        outfile.write("\\n\\n---\\n\\n")
        
        click.echo(f"✅ Markdown built: {output_file}")
        
    def build_pdf(self):
        """Build PDF version using pandoc"""
        click.echo("📄 Building PDF version...")
        
        markdown_file = self.build_dir / "markdown" / "complete_book.md"
        pdf_file = self.build_dir / "pdf" / "from_flask_to_fastapi.pdf"
        pdf_file.parent.mkdir(parents=True, exist_ok=True)
        
        if not markdown_file.exists():
            click.echo("❌ Markdown file not found. Run markdown build first.")
            return
        
        try:
            cmd = [
                "pandoc", str(markdown_file),
                "-o", str(pdf_file),
                "--pdf-engine=weasyprint",
                "--toc",
                "--toc-depth=3",
                "--number-sections",
                "--css", str(self.assets_dir / "styles" / "book.css")
            ]
            subprocess.run(cmd, check=True)
            click.echo(f"✅ PDF built: {pdf_file}")
        except subprocess.CalledProcessError:
            click.echo("❌ PDF build failed - ensure pandoc and weasyprint are installed")
        except FileNotFoundError:
            click.echo("❌ pandoc not found - install pandoc to build PDF")
    
    def build_html(self):
        """Build HTML version"""
        click.echo("🌐 Building HTML version...")
        
        markdown_file = self.build_dir / "markdown" / "complete_book.md"
        html_file = self.build_dir / "html" / "index.html"
        html_file.parent.mkdir(parents=True, exist_ok=True)
        
        if not markdown_file.exists():
            click.echo("❌ Markdown file not found. Run markdown build first.")
            return
        
        try:
            cmd = [
                "pandoc", str(markdown_file),
                "-o", str(html_file),
                "--standalone",
                "--toc",
                "--toc-depth=3",
                "--number-sections",
                "--css", "style.css",
                "--metadata", "title=From Flask to FastAPI"
            ]
            subprocess.run(cmd, check=True)
            
            # Copy CSS file
            css_source = self.assets_dir / "styles" / "book.css"
            css_dest = html_file.parent / "style.css"
            if css_source.exists():
                shutil.copy2(css_source, css_dest)
            
            click.echo(f"✅ HTML built: {html_file}")
        except subprocess.CalledProcessError:
            click.echo("❌ HTML build failed")
        except FileNotFoundError:
            click.echo("❌ pandoc not found - install pandoc to build HTML")
    
    def build_epub(self):
        """Build EPUB version"""
        click.echo("📚 Building EPUB version...")
        
        markdown_file = self.build_dir / "markdown" / "complete_book.md"
        epub_file = self.build_dir / "epub" / "from_flask_to_fastapi.epub"
        epub_file.parent.mkdir(parents=True, exist_ok=True)
        
        if not markdown_file.exists():
            click.echo("❌ Markdown file not found. Run markdown build first.")
            return
        
        try:
            cmd = [
                "pandoc", str(markdown_file),
                "-o", str(epub_file),
                "--toc",
                "--toc-depth=3",
                "--epub-metadata", str(self.assets_dir / "metadata.xml")
            ]
            subprocess.run(cmd, check=True)
            click.echo(f"✅ EPUB built: {epub_file}")
        except subprocess.CalledProcessError:
            click.echo("❌ EPUB build failed")
        except FileNotFoundError:
            click.echo("❌ pandoc not found - install pandoc to build EPUB")
    
    def build_all(self):
        """Build all formats"""
        self.build_markdown()
        self.build_pdf()
        self.build_html()
        self.build_epub()
    
    def clean(self):
        """Clean build directory"""
        if self.build_dir.exists():
            shutil.rmtree(self.build_dir)
        click.echo("🧹 Build directory cleaned")
    
    def _is_chapter_dir(self, path: Path) -> bool:
        """Check if directory is a chapter directory"""
        return (path.is_dir() and 
                not path.name.startswith('.') and
                path.name.startswith(('01_', '02_', '03_', '04_', '05_', 
                                     '06_', '07_', '08_', '09_', '10_')))
    
    def _write_table_of_contents(self, outfile):
        """Write table of contents"""
        for chapter_dir in sorted(self.chapters_dir.iterdir()):
            if self._is_chapter_dir(chapter_dir):
                readme_file = chapter_dir / "README.md"
                if readme_file.exists():
                    first_line = readme_file.read_text().split('\\n')[0]
                    chapter_title = first_line.replace('# ', '')
                    chapter_num = chapter_dir.name.split('_')[0]
                    outfile.write(f"{chapter_num}. {chapter_title}\\n")
    
    def _process_chapter_content(self, content: str, chapter_dir: Path) -> str:
        """Process chapter content to fix relative links"""
        # This would implement link processing logic
        # For now, return content as-is
        return content


@click.command()
@click.option('--format', '-f', 
              type=click.Choice(['markdown', 'pdf', 'html', 'epub', 'all']), 
              default='all',
              help='Format to build')
@click.option('--clean', '-c', is_flag=True, help='Clean build directory first')
@click.option('--book-root', default='.', help='Book root directory')
def main(format, clean, book_root):
    """Build the FastAPI book in various formats"""
    
    book_root = Path(book_root)
    builder = BookBuilder(book_root)
    
    if clean:
        builder.clean()
    
    if format == 'markdown':
        builder.build_markdown()
    elif format == 'pdf':
        builder.build_pdf()
    elif format == 'html':
        builder.build_html()
    elif format == 'epub':
        builder.build_epub()
    else:
        builder.build_all()
    
    click.echo("\\n📚 Build completed!")


if __name__ == "__main__":
    main()
'''


def get_validation_script() -> str:
    """Generate content validation script"""
    return '''#!/usr/bin/env python3
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
        links = re.findall(r'\\[([^\\]]+)\\]\\(([^\\)]+)\\)', content)
        
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
            click.echo(f"\\n❌ {len(self.errors)} errors found:")
            for error in self.errors:
                click.echo(f"  • {error}")
        
        if self.warnings:
            click.echo(f"\\n⚠️  {len(self.warnings)} warnings:")
            for warning in self.warnings:
                click.echo(f"  • {warning}")
        
        if not self.errors and not self.warnings:
            click.echo("\\n✅ All validations passed!")
        elif not self.errors:
            click.echo("\\n✅ No errors found (warnings can be addressed)")
        
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
        click.echo("\\n💡 Run with --fix to attempt automatic fixes", err=True)
        exit(1)
    else:
        click.echo("\\n🎉 Content validation completed successfully!")


if __name__ == "__main__":
    main()
'''


def get_publish_script() -> str:
    """Generate publishing script"""
    return '''#!/usr/bin/env python3
"""
Book Publishing Script
Handles publishing the book to various platforms
"""

import click
import shutil
from pathlib import Path
import subprocess
from datetime import datetime
import zipfile


class BookPublisher:
    """Publishes the book to various platforms"""
    
    def __init__(self, book_root: Path):
        self.book_root = book_root
        self.build_dir = book_root / "book_build"
        self.publish_dir = book_root / "publish"
        
    def prepare_release(self, version: str):
        """Prepare release package"""
        click.echo(f"📦 Preparing release {version}...")
        
        release_dir = self.publish_dir / f"release-{version}"
        release_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy built files
        formats = ["pdf", "epub", "html"]
        for fmt in formats:
            src_dir = self.build_dir / fmt
            if src_dir.exists():
                dst_dir = release_dir / fmt
                shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)
        
        # Create release archive
        archive_path = self.publish_dir / f"from-flask-to-fastapi-{version}.zip"
        with zipfile.ZipFile(archive_path, 'w') as zipf:
            for file_path in release_dir.rglob('*'):
                if file_path.is_file():
                    arcname = file_path.relative_to(release_dir)
                    zipf.write(file_path, arcname)
        
        click.echo(f"✅ Release package created: {archive_path}")
        return archive_path
    
    def publish_github_pages(self):
        """Publish HTML version to GitHub Pages"""
        click.echo("🌐 Publishing to GitHub Pages...")
        
        html_dir = self.build_dir / "html"
        if not html_dir.exists():
            click.echo("❌ HTML build not found. Run build first.")
            return
        
        # This would implement GitHub Pages publishing logic
        click.echo("📝 GitHub Pages publishing would be implemented here")
        click.echo("💡 Consider using GitHub Actions for automated publishing")
    
    def validate_release(self) -> bool:
        """Validate release files"""
        click.echo("🔍 Validating release files...")
        
        required_files = [
            self.build_dir / "pdf" / "from_flask_to_fastapi.pdf",
            self.build_dir / "epub" / "from_flask_to_fastapi.epub",
            self.build_dir / "html" / "index.html"
        ]
        
        missing_files = []
        for file_path in required_files:
            if not file_path.exists():
                missing_files.append(str(file_path))
        
        if missing_files:
            click.echo(f"❌ Missing files: {missing_files}")
            return False
        
        click.echo("✅ All required files present")
        return True


@click.command()
@click.option('--version', '-v', required=True, help='Release version')
@click.option('--github-pages', is_flag=True, help='Publish to GitHub Pages')
@click.option('--book-root', default='.', help='Book root directory')
def main(version, github_pages, book_root):
    """Publish the book to various platforms"""
    
    book_root = Path(book_root)
    publisher = BookPublisher(book_root)
    
    # Validate release
    if not publisher.validate_release():
        click.echo("❌ Release validation failed")
        exit(1)
    
    # Prepare release package
    publisher.prepare_release(version)
    
    # Publish to GitHub Pages if requested
    if github_pages:
        publisher.publish_github_pages()
    
    click.echo(f"\\n🎉 Book published successfully! Version: {version}")


if __name__ == "__main__":
    main()
'''