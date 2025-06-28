#!/usr/bin/env python3
"""
Book Project Structure Generator - Main Script
Creates the complete directory structure for "From Flask to FastAPI: A Journey of Async Programming"

Usage:
    python create_book_structure.py [OPTIONS]
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import click

from structure_generator import DirectoryStructureGenerator
from file_generator import InitialFileGenerator


class BookStructureGenerator:
    """Main coordinator for book project structure generation"""
    
    def __init__(self, base_path: str = "."):
        self.base_path = Path(base_path).expanduser().resolve()
        self.book_root = self.base_path / "async-book"
        
        # Initialize generators
        self.dir_generator = DirectoryStructureGenerator(self.book_root)
        self.file_generator = InitialFileGenerator(self.book_root)
    
    def create_structure(self, include_files: bool = True, verbose: bool = False):
        """Create the complete book structure"""
        
        if verbose:
            click.echo(f"📍 Base path: {self.base_path}")
            click.echo(f"📁 Creating book structure at: {self.book_root}")
        
        # Check if directory already exists
        if self.book_root.exists():
            if not click.confirm(
                f"Directory '{self.book_root}' already exists. Continue?",
                default=False
            ):
                click.echo("Operation cancelled.")
                return False
        
        try:
            # Step 1: Create directory structure
            click.echo("🏗️  Creating directory structure...")
            self.dir_generator.create_directories(verbose=verbose)
            
            # Step 2: Create initial files (optional)
            if include_files:
                click.echo("📝 Creating initial files...")
                self.file_generator.create_initial_files(verbose=verbose)
            
            # Success message
            self._print_success_message()
            return True
            
        except Exception as e:
            click.echo(f"❌ Error creating book structure: {e}", err=True)
            return False
    
    def _print_success_message(self):
        """Print success message with next steps"""
        click.echo()
        click.echo("🎉 Book project structure created successfully!")
        click.echo()
        click.echo("📖 Next steps:")
        click.echo(f"   1. cd {self.book_root}")
        click.echo("   2. pip install -r requirements.txt")
        click.echo("   3. Start writing chapters in the chapters/ directory")
        click.echo("   4. Add code examples to src/")
        click.echo("   5. Build the book: python tools/build_scripts/build_book.py")
        click.echo()
        click.echo("📂 Available commands:")
        click.echo("   make help          - Show all available commands")
        click.echo("   make examples      - Run Flask and FastAPI examples")
        click.echo("   make async-demo    - Run async patterns demo")
        click.echo("   make validate      - Validate book content")
        click.echo("   make build         - Build the complete book")


@click.command()
@click.option(
    '--path', '-p',
    default='.',
    help='Base path where to create the "async-book" directory',
    type=click.Path(exists=True, file_okay=False, dir_okay=True)
)
@click.option(
    '--no-files',
    is_flag=True,
    help='Create only directory structure, skip initial files'
)
@click.option(
    '--verbose', '-v',
    is_flag=True,
    help='Enable verbose output'
)
@click.option(
    '--dry-run',
    is_flag=True,
    help='Show what would be created without actually creating it'
)
@click.version_option(version='1.0.0', prog_name='Book Structure Generator')
def main(path, no_files, verbose, dry_run):
    """
    Generate complete project structure for "From Flask to FastAPI: A Journey of Async Programming" book.
    
    This script creates a comprehensive directory structure with optional initial files
    for authoring a professional book on Flask to FastAPI migration.
    
    Examples:
    
        python create_book_structure.py
        
        python create_book_structure.py --path ~/books
        
        python create_book_structure.py --no-files --verbose
        
        python create_book_structure.py --dry-run
    """
    
    if dry_run:
        click.echo("🔍 Dry run mode - showing what would be created:")
        click.echo()
        
        # Show directory structure
        dir_gen = DirectoryStructureGenerator(Path(path) / "async-book")
        dir_gen.show_structure()
        
        if not no_files:
            click.echo("\n📝 Initial files would be created:")
            file_gen = InitialFileGenerator(Path(path) / "async-book")
            file_gen.show_files()
        
        click.echo(f"\n📍 Location: {Path(path).expanduser().resolve() / 'async-book'}")
        return
    
    # Create generator and run
    generator = BookStructureGenerator(path)
    
    success = generator.create_structure(
        include_files=not no_files,
        verbose=verbose
    )
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()