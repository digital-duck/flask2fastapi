# Setup script for team members
#!/bin/bash
# setup_fastapi.sh

echo "Setting up FastAPI development environment..."

# Create virtual environment
conda create -n fastapi python=3.11
conda activate fastapi
# python -m venv fastapi
# source fastapi/bin/activate  # On Windows: fastapi\Scripts\activate

# Install development dependencies
pip install fastapi uvicorn[standard] aiohttp aioboto3 pytest pytest-asyncio httpx

# Install development tools
pip install black isort mypy pre-commit

# Setup pre-commit hooks
pre-commit install

echo "Environment ready! Start with: uvicorn main:app --reload"
