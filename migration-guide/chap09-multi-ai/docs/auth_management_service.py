# File: src/security/auth_management_service.py
"""
Comprehensive authentication and API key management service.
Implements JWT tokens, API key lifecycle, MFA, and session security.
"""

import asyncio
import hashlib
import secrets
import base64
import pyotp
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any, Union, Tuple
from enum import Enum
from dataclasses import dataclass
import jwt
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from pydantic import BaseModel, Field, EmailStr
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import Column, Integer, String, DateTime, Boolean, JSON, Text
from sqlalchemy.dialects.postgresql import UUID
import uuid
from passlib.context import CryptContext
import redis.asyncio as redis

# Authentication Types
class AuthMethod(str, Enum):
    API_KEY = "api_key"
    JWT = "jwt"
    OAUTH2 = "oauth2"
    MFA = "mfa"

class TokenType(str, Enum):
    ACCESS = "access"
    REFRESH = "refresh"
    ID = "id"

class MFAMethod(str, Enum):
    TOTP = "totp"
    SMS = "sms"
    EMAIL = "email"
    HARDWARE = "hardware"

class KeyStatus(str, Enum):
    ACTIVE = "active"
    EXPIRED = "expired"
    REVOKED = "revoked"
    PENDING = "pending"

# Data Models
@dataclass
class APIKeyScope:
    """Defines permissions for an API key"""
    resource: str
    actions: List[str]
    constraints: Dict[str, Any] = None

@dataclass
class JWTClaims:
    """JWT token claims"""
    sub: str  # Subject (user ID)
    iss: str  # Issuer
    aud: str  # Audience
    exp: int  # Expiration time
    nbf: int  # Not before
    iat: int  # Issued at
    jti: str  # JWT ID
    scope: List[str] = None
    roles: List[str] = None
    permissions: List[str] = None

# Database Models
class APIKey(Base):
    __tablename__ = "api_keys"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    key_id = Column(String(32), unique=True, nullable=False, index=True)
    key_hash = Column(String(128), nullable=False)
    name = Column(String(100), nullable=False)
    description = Column(Text)
    user_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    scopes = Column(JSON, nullable=False)
    status = Column(String(20), default=KeyStatus.ACTIVE, index=True)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    expires_at = Column(DateTime(timezone=True))
    last_used_at = Column(DateTime(timezone=True))
    usage_count = Column(Integer, default=0)
    rate_limit = Column(Integer, default=1000)  # requests per hour
    ip_whitelist = Column(JSON)  # List of allowed IPs
    metadata = Column(JSON)

class RefreshToken(Base):
    __tablename__ = "refresh_tokens"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    token_hash = Column(String(128), unique=True, nullable=False)
    user_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    device_id = Column(String(100))
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    expires_at = Column(DateTime(timezone=True), nullable=False)
    is_active = Column(Boolean, default=True)
    metadata = Column(JSON)

class UserMFA(Base):
    __tablename__ = "user_mfa"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    method = Column(String(20), nullable=False)
    secret = Column(String(255))  # Encrypted
    backup_codes = Column(JSON)  # Encrypted list
    is_enabled = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    last_used_at = Column(DateTime(timezone=True))

class UserSession(Base):
    __tablename__ = "user_sessions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(String(128), unique=True, nullable=False)
    user_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    ip_address = Column(String(45))
    user_agent = Column(Text)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    last_activity = Column(DateTime(timezone=True), default=datetime.utcnow)
    expires_at = Column(DateTime(timezone=True), nullable=False)
    is_active = Column(Boolean, default=True)
    metadata = Column(JSON)

# Cryptographic Utilities
class CryptoManager:
    """Handles cryptographic operations"""
    
    def __init__(self, secret_key: str):
        self.secret_key = secret_key.encode()
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
    
    def hash_password(self, password: str) -> str:
        """Hash a password using bcrypt"""
        return self.pwd_context.hash(password)
    
    def verify_password(self, password: str, hashed: str) -> bool:
        """Verify a password against its hash"""
        return self.pwd_context.verify(password, hashed)
    
    def generate_api_key(self) -> Tuple[str, str]:
        """Generate API key and its hash"""
        # Generate key ID (visible part)
        key_id = secrets.token_urlsafe(16)
        
        # Generate secret part
        secret = secrets.token_urlsafe(32)
        
        # Full key format: keyid_secret
        full_key = f"{key_id}_{secret}"
        
        # Hash for storage
        key_hash = hashlib.sha256(full_key.encode()).hexdigest()
        
        return full_key, key_hash
    
    def hash_api_key(self, api_key: str) -> str:
        """Hash an API key for storage"""
        return hashlib.sha256(api_key.encode()).hexdigest()
    
    def generate_session_id(self) -> str:
        """Generate a secure session ID"""
        return secrets.token_urlsafe(32)
    
    def encrypt_data(self, data: str) -> str:
        """Encrypt sensitive data (simplified - use proper encryption in production)"""
        # In production, use proper encryption like Fernet
        return base64.b64encode(data.encode()).decode()
    
    def decrypt_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data"""
        return base64.b64decode(encrypted_data.encode()).decode()

# JWT Token Manager
class JWTManager:
    """Manages JWT token operations"""
    
    def __init__(self, private_key: str, public_key: str, 
                 issuer: str, audience: str):
        self.private_key = private_key
        self.public_key = public_key
        self.issuer = issuer
        self.audience = audience
        self.algorithm = "RS256"
    
    def create_access_token(self, user_id: str, scopes: List[str] = None,
                          roles: List[str] = None,
                          expires_delta: timedelta = None) -> str:
        """Create a JWT access token"""
        if expires_delta is None:
            expires_delta = timedelta(minutes=30)
        
        now = datetime.now(timezone.utc)
        exp = now + expires_delta
        
        claims = JWTClaims(
            sub=user_id,
            iss=self.issuer,
            aud=self.audience,
            exp=int(exp.timestamp()),
            nbf=int(now.timestamp()),
            iat=int(now.timestamp()),
            jti=str(uuid.uuid4()),
            scope=scopes or [],
            roles=roles or []
        )
        
        return jwt.encode(
            claims.__dict__,
            self.private_key,
            algorithm=self.algorithm
        )
    
    def create_refresh_token(self, user_id: str, 
                           expires_delta: timedelta = None) -> str:
        """Create a refresh token"""
        if expires_delta is None:
            expires_delta = timedelta(days=30)
        
        now = datetime.now(timezone.utc)
        exp = now + expires_delta
        
        claims = {
            "sub": user_id,
            "iss": self.issuer,
            "aud": self.audience,
            "exp": int(exp.timestamp()),
            "iat": int(now.timestamp()),
            "jti": str(uuid.uuid4()),
            "type": "refresh"
        }
        
        return jwt.encode(claims, self.private_key, algorithm=self.algorithm)
    
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify and decode a JWT token"""
        try:
            payload = jwt.decode(
                token,
                self.public_key,
                algorithms=[self.algorithm],
                audience=self.audience,
                issuer=self.issuer
            )
            return payload
        except jwt.InvalidTokenError:
            return None
    
    def extract_claims(self, token: str) -> Optional[JWTClaims]:
        """Extract claims from a JWT token"""
        payload = self.verify_token(token)
        if not payload:
            return None
        
        return JWTClaims(**payload)

# API Key Manager
class APIKeyManager:
    """Manages API key lifecycle"""
    
    def __init__(self, db_session: AsyncSession, 
                 crypto_manager: CryptoManager,
                 redis_client: redis.Redis):
        self.db_session = db_session
        self.crypto = crypto_manager
        self.redis = redis_client
    
    async def create_api_key(self, user_id: str, name: str,
                           scopes: List[APIKeyScope],
                           description: str = None,
                           expires_in_days: int = None,
                           rate_limit: int = 1000,
                           ip_whitelist: List[str] = None) -> Tuple[str, str]:
        """Create a new API key"""
        # Generate key and hash
        full_key, key_hash = self.crypto.generate_api_key()
        key_id = full_key.split('_')[0]
        
        # Set expiration
        expires_at = None
        if expires_in_days:
            expires_at = datetime.now(timezone.utc) + timedelta(days=expires_in_days)
        
        # Create database record
        api_key = APIKey(
            key_id=key_id,
            key_hash=key_hash,
            name=name,
            description=description,
            user_id=user_id,
            scopes=[scope.__dict__ for scope in scopes],
            expires_at=expires_at,
            rate_limit=rate_limit,
            ip_whitelist=ip_whitelist
        )
        
        self.db_session.add(api_key)
        await self.db_session.commit()
        
        return full_key, str(api_key.id)
    
    async def verify_api_key(self, api_key: str, 
                           ip_address: str = None) -> Optional[APIKey]:
        """Verify an API key and return key info"""
        # Hash the provided key
        key_hash = self.crypto.hash_api_key(api_key)
        
        # Check cache first
        cached_key = await self.redis.get(f"api_key:{key_hash}")
        if cached_key:
            # Parse cached key data
            import json
            key_data = json.loads(cached_key)
            if key_data.get('status') != KeyStatus.ACTIVE:
                return None
        
        # Query database
        result = await self.db_session.execute(
            select(APIKey).where(
                APIKey.key_hash == key_hash,
                APIKey.status == KeyStatus.ACTIVE
            )
        )
        key_record = result.scalar_one_or_none()
        
        if not key_record:
            return None
        
        # Check expiration
        if key_record.expires_at and key_record.expires_at < datetime.now(timezone.utc):
            key_record.status = KeyStatus.EXPIRED
            await self.db_session.commit()
            return None
        
        # Check IP whitelist
        if key_record.ip_whitelist and ip_address:
            if ip_address not in key_record.ip_whitelist:
                return None
        
        # Update usage statistics
        key_record.last_used_at = datetime.now(timezone.utc)
        key_record.usage_count += 1
        await self.db_session.commit()
        
        # Cache key data
        await self.redis.setex(
            f"api_key:{key_hash}",
            300,  # 5 minutes
            json.dumps({
                'id': str(key_record.id),
                'user_id': str(key_record.user_id),
                'scopes': key_record.scopes,
                'status': key_record.status
            })
        )
        
        return key_record
    
    async def revoke_api_key(self, key_id: str, user_id: str = None) -> bool:
        """Revoke an API key"""
        query = select(APIKey).where(APIKey.id == key_id)
        if user_id:
            query = query.where(APIKey.user_id == user_id)
        
        result = await self.db_session.execute(query)
        key_record = result.scalar_one_or_none()
        
        if not key_record:
            return False
        
        key_record.status = KeyStatus.REVOKED
        await self.db_session.commit()
        
        # Remove from cache
        await self.redis.delete(f"api_key:{key_record.key_hash}")
        
        return True
    
    async def rotate_api_key(self, key_id: str, user_id: str) -> Optional[Tuple[str, str]]:
        """Rotate an API key (create new, mark old as expired)"""
        # Get existing key
        result = await self.db_session.execute(
            select(APIKey).where(
                APIKey.id == key_id,
                APIKey.user_id == user_id,
                APIKey.status == KeyStatus.ACTIVE
            )
        )
        old_key = result.scalar_one_or_none()
        
        if not old_key:
            return None
        
        # Create new key with same properties
        scopes = [APIKeyScope(**scope) for scope in old_key.scopes]
        new_key, new_key_id = await self.create_api_key(
            user_id=user_id,
            name=old_key.name,
            scopes=scopes,
            description=old_key.description,
            rate_limit=old_key.rate_limit,
            ip_whitelist=old_key.ip_whitelist
        )
        
        # Mark old key as expired
        old_key.status = KeyStatus.EXPIRED
        await self.db_session.commit()
        
        return new_key, new_key_id

# Multi-Factor Authentication Manager
class MFAManager:
    """Manages multi-factor authentication"""
    
    def __init__(self, db_session: AsyncSession, 
                 crypto_manager: CryptoManager):
        self.db_session = db_session
        self.crypto = crypto_manager
    
    async def setup_totp(self, user_id: str) -> Tuple[str, List[str]]:
        """Setup TOTP for a user"""
        # Generate secret
        secret = pyotp.random_base32()
        
        # Generate backup codes
        backup_codes = [secrets.token_hex(8) for _ in range(10)]
        
        # Encrypt sensitive data
        encrypted_secret = self.crypto.encrypt_data(secret)
        encrypted_codes = [self.crypto.encrypt_data(code) for code in backup_codes]
        
        # Store in database
        mfa_record = UserMFA(
            user_id=user_id,
            method=MFAMethod.TOTP,
            secret=encrypted_secret,
            backup_codes=encrypted_codes,
            is_enabled=False  # User must verify first
        )
        
        self.db_session.add(mfa_record)
        await self.db_session.commit()
        
        return secret, backup_codes
    
    async def verify_totp_setup(self, user_id: str, token: str) -> bool:
        """Verify TOTP setup and enable MFA"""
        result = await self.db_session.execute(
            select(UserMFA).where(
                UserMFA.user_id == user_id,
                UserMFA.method == MFAMethod.TOTP,
                UserMFA.is_enabled == False
            )
        )
        mfa_record = result.scalar_one_or_none()
        
        if not mfa_record:
            return False
        
        # Decrypt secret
        secret = self.crypto.decrypt_data(mfa_record.secret)
        
        # Verify token
        totp = pyotp.TOTP(secret)
        if totp.verify(token, valid_window=1):
            mfa_record.is_enabled = True
            await self.db_session.commit()
            return True
        
        return False
    
    async def verify_totp_token(self, user_id: str, token: str) -> bool:
        """Verify a TOTP token"""
        result = await self.db_session.execute(
            select(UserMFA).where(
                UserMFA.user_id == user_id,
                UserMFA.method == MFAMethod.TOTP,
                UserMFA.is_enabled == True
            )
        )
        mfa_record = result.scalar_one_or_none()
        
        if not mfa_record:
            return False
        
        # Decrypt secret
        secret = self.crypto.decrypt_data(mfa_record.secret)
        
        # Verify token
        totp = pyotp.TOTP(secret)
        if totp.verify(token, valid_window=1):
            mfa_record.last_used_at = datetime.now(timezone.utc)
            await self.db_session.commit()
            return True
        
        return False
    
    async def verify_backup_code(self, user_id: str, code: str) -> bool:
        """Verify and consume a backup code"""
        result = await self.db_session.execute(
            select(UserMFA).where(
                UserMFA.user_id == user_id,
                UserMFA.method == MFAMethod.TOTP,
                UserMFA.is_enabled == True
            )
        )
        mfa_record = result.scalar_one_or_none()
        
        if not mfa_record or not mfa_record.backup_codes:
            return False
        
        # Check if code matches any backup code
        for i, encrypted_code in enumerate(mfa_record.backup_codes):
            decrypted_code = self.crypto.decrypt_data(encrypted_code)
            if secrets.compare_digest(code, decrypted_code):
                # Remove used backup code
                mfa_record.backup_codes.pop(i)
                await self.db_session.commit()
                return True
        
        return False

# Session Manager
class SessionManager:
    """Manages user sessions"""
    
    def __init__(self, db_session: AsyncSession, 
                 crypto_manager: CryptoManager,
                 redis_client: redis.Redis):
        self.db_session = db_session
        self.crypto = crypto_manager
        self.redis = redis_client
        self.session_timeout = timedelta(hours=24)
    
    async def create_session(self, user_id: str, ip_address: str,
                           user_agent: str, metadata: Dict[str, Any] = None) -> str:
        """Create a new user session"""
        session_id = self.crypto.generate_session_id()
        expires_at = datetime.now(timezone.utc) + self.session_timeout
        
        # Create database record
        session = UserSession(
            session_id=session_id,
            user_id=user_id,
            ip_address=ip_address,
            user_agent=user_agent,
            expires_at=expires_at,
            metadata=metadata or {}
        )
        
        self.db_session.add(session)
        await self.db_session.commit()
        
        # Cache session data
        session_data = {
            'user_id': user_id,
            'ip_address': ip_address,
            'created_at': session.created_at.isoformat(),
            'expires_at': expires_at.isoformat()
        }
        
        await self.redis.setex(
            f"session:{session_id}",
            int(self.session_timeout.total_seconds()),
            json.dumps(session_data)
        )
        
        return session_id
    
    async def verify_session(self, session_id: str, 
                           ip_address: str = None) -> Optional[Dict[str, Any]]:
        """Verify a session and return session data"""
        # Check cache first
        cached_session = await self.redis.get(f"session:{session_id}")
        if cached_session:
            session_data = json.loads(cached_session)
            
            # Check IP address if provided
            if ip_address and session_data.get('ip_address') != ip_address:
                return None
            
            return session_data
        
        # Check database
        result = await self.db_session.execute(
            select(UserSession).where(
                UserSession.session_id == session_id,
                UserSession.is_active == True,
                UserSession.expires_at > datetime.now(timezone.utc)
            )
        )
        session = result.scalar_one_or_none()
        
        if not session:
            return None
        
        # Check IP address
        if ip_address and session.ip_address != ip_address:
            return None
        
        # Update last activity
        session.last_activity = datetime.now(timezone.utc)
        await self.db_session.commit()
        
        return {
            'user_id': str(session.user_id),
            'ip_address': session.ip_address,
            'created_at': session.created_at.isoformat(),
            'expires_at': session.expires_at.isoformat()
        }
    
    async def revoke_session(self, session_id: str) -> bool:
        """Revoke a session"""
        # Remove from cache
        await self.redis.delete(f"session:{session_id}")
        
        # Update database
        result = await self.db_session.execute(
            select(UserSession).where(UserSession.session_id == session_id)
        )
        session = result.scalar_one_or_none()
        
        if session:
            session.is_active = False
            await self.db_session.commit()
            return True
        
        return False
    
    async def revoke_all_user_sessions(self, user_id: str) -> int:
        """Revoke all sessions for a user"""
        # Get all active sessions
        result = await self.db_session.execute(
            select(UserSession).where(
                UserSession.user_id == user_id,
                UserSession.is_active == True
            )
        )
        sessions = result.scalars().all()
        
        # Revoke each session
        count = 0
        for session in sessions:
            await self.redis.delete(f"session:{session.session_id}")
            session.is_active = False
            count += 1
        
        await self.db_session.commit()
        return count

# Main Authentication Service
class AuthenticationService:
    """Main authentication service coordinator"""
    
    def __init__(self, db_session: AsyncSession, redis_client: redis.Redis,
                 secret_key: str, jwt_private_key: str, jwt_public_key: str,
                 issuer: str, audience: str):
        self.crypto = CryptoManager(secret_key)
        self.jwt_manager = JWTManager(jwt_private_key, jwt_public_key, issuer, audience)
        self.api_key_manager = APIKeyManager(db_session, self.crypto, redis_client)
        self.mfa_manager = MFAManager(db_session, self.crypto)
        self.session_manager = SessionManager(db_session, self.crypto, redis_client)
    
    async def authenticate_request(self, request_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Authenticate a request using various methods"""
        auth_header = request_data.get('authorization')
        api_key = request_data.get('api_key')
        session_id = request_data.get('session_id')
        ip_address = request_data.get('ip_address')
        
        # Try API key authentication
        if api_key:
            key_record = await self.api_key_manager.verify_api_key(api_key, ip_address)
            if key_record:
                return {
                    'method': AuthMethod.API_KEY,
                    'user_id': str(key_record.user_id),
                    'scopes': key_record.scopes,
                    'key_id': str(key_record.id)
                }
        
        # Try JWT authentication
        if auth_header and auth_header.startswith('Bearer '):
            token = auth_header[7:]
            claims = self.jwt_manager.extract_claims(token)
            if claims:
                return {
                    'method': AuthMethod.JWT,
                    'user_id': claims.sub,
                    'scopes': claims.scope,
                    'roles': claims.roles,
                    'token_id': claims.jti
                }
        
        # Try session authentication
        if session_id:
            session_data = await self.session_manager.verify_session(session_id, ip_address)
            if session_data:
                return {
                    'method': 'session',
                    'user_id': session_data['user_id'],
                    'session_id': session_id
                }
        
        return None