# Makefile for "From Flask to FastAPI" book project

.PHONY: help install build validate clean serve examples lint test setup-dev book-stats

help:  ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $1, $2}'

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
	@echo "Activate with: source venv/bin/activate (Linux/Mac) or venv\\Scripts\\activate (Windows)"
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
