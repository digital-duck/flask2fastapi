#!/usr/bin/env python3
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
            outfile.write("# From Flask to FastAPI: A Journey of Async Programming\n\n")
            outfile.write(f"*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
            outfile.write("---\n\n")
            
            # Add table of contents
            outfile.write("# Table of Contents\n\n")
            self._write_table_of_contents(outfile)
            outfile.write("\n---\n\n")
            
            # Combine all chapters
            for chapter_dir in sorted(self.chapters_dir.iterdir()):
                if self._is_chapter_dir(chapter_dir):
                    readme_file = chapter_dir / "README.md"
                    if readme_file.exists():
                        outfile.write(f"\n\n# {chapter_dir.name.replace('_', ' ').title()}\n\n")
                        
                        content = readme_file.read_text(encoding='utf-8')
                        # Process content to fix relative links
                        processed_content = self._process_chapter_content(content, chapter_dir)
                        outfile.write(processed_content)
                        outfile.write("\n\n---\n\n")
        
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
                    first_line = readme_file.read_text().split('\n')[0]
                    chapter_title = first_line.replace('# ', '')
                    chapter_num = chapter_dir.name.split('_')[0]
                    outfile.write(f"{chapter_num}. {chapter_title}\n")
    
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
    
    click.echo("\n📚 Build completed!")


if __name__ == "__main__":
    main()
