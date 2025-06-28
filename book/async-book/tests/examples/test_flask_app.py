"""
Example test file for Flask application
"""

import pytest
from src.flask_examples.basic_app import app, db, User


@pytest.fixture
def client():
    """Test client fixture"""
    app.config['TESTING'] = True
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'
    
    with app.test_client() as client:
        with app.app_context():
            db.create_all()
            yield client


def test_home_endpoint(client):
    """Test home endpoint"""
    response = client.get('/')
    assert response.status_code == 200
    assert b'Flask Application' in response.data


def test_get_users_empty(client):
    """Test getting users when none exist"""
    response = client.get('/users')
    assert response.status_code == 200
    data = response.get_json()
    assert data['count'] == 0
    assert data['users'] == []
