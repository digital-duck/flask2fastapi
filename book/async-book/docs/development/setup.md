# Development Setup Guide

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
   source venv/bin/activate  # On Windows: venv\Scripts\activate
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
