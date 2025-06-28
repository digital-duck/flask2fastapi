#!/usr/bin/env python3
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
    
    click.echo(f"\n🎉 Book published successfully! Version: {version}")


if __name__ == "__main__":
    main()
