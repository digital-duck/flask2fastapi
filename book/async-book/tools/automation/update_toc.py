#!/usr/bin/env python3
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
                first_line = chapter_readme.read_text().split('\n')[0]
                chapter_title = first_line.replace('# ', '')
                toc_lines.append(f"- [{chapter_title}](chapters/{chapter_dir.name}/README.md)")
    
    toc_content = "\n".join(toc_lines)
    
    # Replace existing TOC
    pattern = r"## 📚 Book Structure.*?(?=##|\Z)"
    new_content = re.sub(pattern, toc_content + "\n\n", content, flags=re.DOTALL)
    
    # Write updated README
    readme_path.write_text(new_content)
    print("✅ Table of contents updated")


if __name__ == "__main__":
    book_root = Path(".")
    update_table_of_contents(book_root)
