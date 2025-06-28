"""
Basic Flask Application Example
Chapter 1-3: Demonstrates traditional Flask patterns before migration

This example shows a typical Flask application with:
- Synchronous request handling
- SQLAlchemy ORM (sync)
- Traditional route definitions
- Manual validation and serialization

Usage:
    python src/flask_examples/basic_app.py
"""

from flask import Flask, jsonify, request
from flask_sqlalchemy import SQLAlchemy
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///example.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)


class User(db.Model):
    """User model for demonstration"""
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(80), nullable=False)
    email = db.Column(db.String(120), nullable=False, unique=True)
    created_at = db.Column(db.DateTime, default=db.func.current_timestamp())

    def to_dict(self):
        """Convert model to dictionary"""
        return {
            'id': self.id,
            'name': self.name,
            'email': self.email,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }


# Routes
@app.route('/')
def home():
    """Home endpoint"""
    return jsonify({
        "message": "Flask Application", 
        "version": "1.0",
        "framework": "Flask (Synchronous)"
    })


@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": time.time(),
        "database": "connected"
    })


@app.route('/users', methods=['GET'])
def get_users():
    """Get all users - synchronous operation"""
    start_time = time.time()
    
    # Simulate database query delay
    time.sleep(0.1)
    
    users = User.query.all()
    result = [user.to_dict() for user in users]
    
    processing_time = time.time() - start_time
    
    logger.info(f"Retrieved {len(result)} users in {processing_time:.3f}s")
    
    return jsonify({
        "users": result,
        "count": len(result),
        "processing_time": processing_time,
        "note": "This is a synchronous operation - blocks other requests"
    })


@app.route('/users', methods=['POST'])
def create_user():
    """Create new user - synchronous operation"""
    data = request.get_json()
    
    # Manual validation
    if not data or 'name' not in data or 'email' not in data:
        return jsonify({"error": "Name and email required"}), 400
    
    if '@' not in data['email']:
        return jsonify({"error": "Invalid email format"}), 400
    
    # Check if user exists
    existing_user = User.query.filter_by(email=data['email']).first()
    if existing_user:
        return jsonify({"error": "User with this email already exists"}), 409
    
    # Simulate validation delay
    time.sleep(0.05)
    
    try:
        user = User(name=data['name'], email=data['email'])
        db.session.add(user)
        db.session.commit()
        
        logger.info(f"Created user: {user.email}")
        return jsonify(user.to_dict()), 201
        
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error creating user: {e}")
        return jsonify({"error": "Failed to create user"}), 500


@app.route('/users/<int:user_id>', methods=['GET'])
def get_user(user_id):
    """Get single user - synchronous operation"""
    # Simulate database lookup delay
    time.sleep(0.05)
    
    user = User.query.get(user_id)
    if not user:
        return jsonify({"error": "User not found"}), 404
    
    return jsonify(user.to_dict())


@app.route('/slow-operation', methods=['GET'])
def slow_operation():
    """Simulate slow operation that blocks other requests"""
    logger.info("Starting slow operation...")
    
    # This blocks the entire thread - other requests must wait
    time.sleep(2)
    
    logger.info("Slow operation completed")
    return jsonify({
        "message": "Slow operation completed",
        "note": "This blocked all other requests for 2 seconds"
    })


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({"error": "Internal server error"}), 500


if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        logger.info("Database tables created")
    
    print("\n" + "="*50)
    print("Starting Flask Application")
    print("="*50)
    print("📍 URL: http://localhost:5000")
    print("📖 Endpoints:")
    print("   GET  /              - Home")
    print("   GET  /health        - Health check")
    print("   GET  /users         - List users")
    print("   POST /users         - Create user")
    print("   GET  /users/<id>    - Get user by ID")
    print("   GET  /slow-operation - Slow blocking operation")
    print("\n⚠️  Note: This is a synchronous application")
    print("   - Requests are processed sequentially")
    print("   - Slow operations block other requests")
    print("   - No async/await patterns")
    print("="*50)
    
    app.run(debug=True, port=5000, threaded=True)
