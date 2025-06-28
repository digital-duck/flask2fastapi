# File: src/security/unified_security_service.py
"""
Unified security service that integrates all security components.
Provides a single interface for comprehensive security management.
"""

import asyncio
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Union, Callable
from enum import Enum
from dataclasses import dataclass
from contextlib import asynccontextmanager
from fastapi import Request, Response, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import structlog

# Import security services
from .input_validation_service import InputValidationService, ThreatDetector
from .access_control_service import AccessControlService, RBACManager
from .privacy_compliance_service import PrivacyComplianceService, ConsentManager
from .audit_logging_service import AuditLoggingService, AuditEvent, EventType, SeverityLevel, ActionType, Actor, Resource
from .auth_management_service import AuthenticationService, AuthMethod

# Security Context
@dataclass
class SecurityContext:
    """Complete security context for a request"""
    request_id: str
    user_id: Optional[str] = None
    auth_method: Optional[AuthMethod] = None
    permissions: List[str] = None
    roles: List[str] = None
    risk_score: int = 0
    threat_level: str = "low"
    compliance_tags: List[str] = None
    session_id: Optional[str] = None
    api_key_id: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    validated_input: Dict[str, Any] = None
    privacy_context: Dict[str, Any] = None

class SecurityDecision(str, Enum):
    ALLOW = "allow"
    DENY = "deny"
    CHALLENGE = "challenge"
    MONITOR = "monitor"

# Security Policy
class SecurityPolicy(BaseModel):
    """Defines security policy for the application"""
    
    # Authentication requirements
    require_authentication: bool = True
    allowed_auth_methods: List[AuthMethod] = [AuthMethod.JWT, AuthMethod.API_KEY]
    mfa_required_for: List[str] = []  # List of roles requiring MFA
    
    # Authorization settings
    default_access: str = "deny"
    role_hierarchy: Dict[str, List[str]] = {}
    
    # Input validation
    max_request_size: int = 10 * 1024 * 1024  # 10MB
    validate_all_inputs: bool = True
    sanitize_outputs: bool = True
    
    # Rate limiting
    global_rate_limit: int = 1000  # requests per hour
    per_user_rate_limit: int = 100
    per_endpoint_rate_limit: Dict[str, int] = {}
    
    # Privacy and compliance
    data_retention_days: int = 90
    require_consent: List[str] = ["marketing", "analytics"]
    geographic_restrictions: List[str] = []
    
    # Audit logging
    log_all_requests: bool = True
    log_sensitive_data: bool = False
    audit_retention_days: int = 365
    
    # Security thresholds
    max_risk_score: int = 70
    threat_response_actions: Dict[str, str] = {
        "high": "challenge",
        "critical": "deny"
    }

# Security Metrics
@dataclass
class SecurityMetrics:
    """Security metrics for monitoring"""
    total_requests: int = 0
    blocked_requests: int = 0
    threat_detections: int = 0
    auth_failures: int = 0
    policy_violations: int = 0
    average_risk_score: float = 0.0
    last_updated: datetime = None

# Main Unified Security Service
class UnifiedSecurityService:
    """Unified security service coordinating all security components"""
    
    def __init__(self, 
                 input_validator: InputValidationService,
                 access_controller: AccessControlService,
                 privacy_manager: PrivacyComplianceService,
                 audit_logger: AuditLoggingService,
                 auth_service: AuthenticationService,
                 policy: SecurityPolicy):
        
        self.input_validator = input_validator
        self.access_controller = access_controller
        self.privacy_manager = privacy_manager
        self.audit_logger = audit_logger
        self.auth_service = auth_service
        self.policy = policy
        
        self.logger = structlog.get_logger(__name__)
        self.metrics = SecurityMetrics()
        self._security_cache = {}
        self._event_handlers = {}
    
    async def start(self):
        """Start the security service"""
        await self.audit_logger.start()
        self.logger.info("Unified security service started")
    
    async def stop(self):
        """Stop the security service"""
        await self.audit_logger.stop()
        self.logger.info("Unified security service stopped")
    
    async def process_request(self, request: Request) -> SecurityContext:
        """Process a request through the complete security pipeline"""
        start_time = time.time()
        
        # Initialize security context
        context = SecurityContext(
            request_id=self._generate_request_id(),
            ip_address=self._get_client_ip(request),
            user_agent=request.headers.get("user-agent")
        )
        
        try:
            # Step 1: Input Validation and Threat Detection
            await self._validate_input(request, context)
            
            # Step 2: Authentication
            await self._authenticate_request(request, context)
            
            # Step 3: Authorization and Access Control
            await self._authorize_request(request, context)
            
            # Step 4: Privacy and Compliance Checks
            await self._check_privacy_compliance(request, context)
            
            # Step 5: Risk Assessment
            await self._assess_risk(context)
            
            # Step 6: Security Decision
            decision = await self._make_security_decision(context)
            
            # Step 7: Audit Logging
            await self._log_security_event(request, context, decision)
            
            # Step 8: Update Metrics
            self._update_metrics(context, decision)
            
            # Handle security decision
            if decision == SecurityDecision.DENY:
                raise HTTPException(status_code=403, detail="Access denied")
            elif decision == SecurityDecision.CHALLENGE:
                raise HTTPException(status_code=401, detail="Additional authentication required")
            
            processing_time = time.time() - start_time
            self.logger.info("Request processed", 
                           request_id=context.request_id,
                           processing_time=processing_time,
                           risk_score=context.risk_score)
            
            return context
            
        except HTTPException:
            raise
        except Exception as e:
            self.logger.error("Security processing error", 
                            request_id=context.request_id,
                            error=str(e))
            # Log security error
            await self._log_security_error(request, context, e)
            raise HTTPException(status_code=500, detail="Security processing error")
    
    async def _validate_input(self, request: Request, context: SecurityContext):
        """Validate request input and detect threats"""
        # Extract request data
        request_data = await self._extract_request_data(request)
        
        # Validate input
        validation_result = await self.input_validator.validate_request(
            request_data, request.url.path
        )
        
        if not validation_result.is_valid:
            context.threat_level = "high"
            context.risk_score += 30
            raise HTTPException(status_code=400, detail="Invalid input")
        
        context.validated_input = validation_result.sanitized_data
        context.threat_level = validation_result.threat_level
        
        # Add threat score to risk
        if validation_result.threat_level == "medium":
            context.risk_score += 15
        elif validation_result.threat_level == "high":
            context.risk_score += 30
    
    async def _authenticate_request(self, request: Request, context: SecurityContext):
        """Authenticate the request"""
        if not self.policy.require_authentication:
            return
        
        # Extract authentication data
        auth_data = {
            'authorization': request.headers.get('authorization'),
            'api_key': request.headers.get('x-api-key'),
            'session_id': request.cookies.get('session_id'),
            'ip_address': context.ip_address
        }
        
        # Authenticate
        auth_result = await self.auth_service.authenticate_request(auth_data)
        
        if not auth_result:
            context.risk_score += 50
            raise HTTPException(status_code=401, detail="Authentication required")
        
        # Update context
        context.user_id = auth_result['user_id']
        context.auth_method = auth_result['method']
        context.permissions = auth_result.get('scopes', [])
        context.roles = auth_result.get('roles', [])
        context.session_id = auth_result.get('session_id')
        context.api_key_id = auth_result.get('key_id')
        
        # Check if MFA is required
        if any(role in self.policy.mfa_required_for for role in (context.roles or [])):
            # Check if MFA was used
            if not auth_result.get('mfa_verified'):
                raise HTTPException(status_code=401, detail="MFA required")
    
    async def _authorize_request(self, request: Request, context: SecurityContext):
        """Authorize the request"""
        if not context.user_id:
            return  # Anonymous access
        
        # Check rate limits
        await self._check_rate_limits(context)
        
        # Check permissions
        required_permission = self._get_required_permission(request.method, request.url.path)
        
        if required_permission:
            has_permission = await self.access_controller.check_permission(
                context.user_id, required_permission, context.roles
            )
            
            if not has_permission:
                context.risk_score += 25
                raise HTTPException(status_code=403, detail="Insufficient permissions")
    
    async def _check_privacy_compliance(self, request: Request, context: SecurityContext):
        """Check privacy and compliance requirements"""
        if not context.user_id:
            return
        
        # Check data processing consent
        if request.method in ['POST', 'PUT', 'PATCH']:
            processing_purposes = self._extract_processing_purposes(request)
            
            for purpose in processing_purposes:
                if purpose in self.policy.require_consent:
                    has_consent = await self.privacy_manager.check_consent(
                        context.user_id, purpose
                    )
                    
                    if not has_consent:
                        raise HTTPException(
                            status_code=451, 
                            detail=f"Consent required for {purpose}"
                        )
        
        # Geographic restrictions
        if self.policy.geographic_restrictions:
            user_location = await self._get_user_location(context.ip_address)
            if user_location in self.policy.geographic_restrictions:
                raise HTTPException(
                    status_code=451, 
                    detail="Access restricted in your location"
                )
        
        # Set compliance context
        context.compliance_tags = self._determine_compliance_tags(request, context)
    
    async def _assess_risk(self, context: SecurityContext):
        """Assess overall risk score"""
        # Time-based risk (off-hours access)
        current_hour = datetime.now().hour
        if current_hour < 6 or current_hour > 22:
            context.risk_score += 5
        
        # Geographic risk
        if await self._is_suspicious_location(context.ip_address, context.user_id):
            context.risk_score += 15
        
        # Behavioral risk
        if await self._detect_anomalous_behavior(context):
            context.risk_score += 20
        
        # Authentication method risk
        if context.auth_method == AuthMethod.API_KEY:
            context.risk_score += 5  # API keys are slightly riskier
    
    async def _make_security_decision(self, context: SecurityContext) -> SecurityDecision:
        """Make final security decision"""
        # Check risk threshold
        if context.risk_score >= self.policy.max_risk_score:
            if context.risk_score >= 90:
                return SecurityDecision.DENY
            else:
                return SecurityDecision.CHALLENGE
        
        # Check threat level
        threat_action = self.policy.threat_response_actions.get(context.threat_level)
        if threat_action == "deny":
            return SecurityDecision.DENY
        elif threat_action == "challenge":
            return SecurityDecision.CHALLENGE
        
        return SecurityDecision.ALLOW
    
    async def _log_security_event(self, request: Request, 
                                context: SecurityContext, 
                                decision: SecurityDecision):
        """Log security event"""
        actor = Actor(
            type="user" if context.user_id else "anonymous",
            identifier=context.user_id or "anonymous",
            ip_address=context.ip_address,
            user_agent=context.user_agent,
            session_id=context.session_id
        )
        
        resource = Resource(
            type="endpoint",
            identifier=f"{request.method} {request.url.path}",
            attributes={
                "query_params": dict(request.query_params),
                "headers": dict(request.headers)
            }
        )
        
        event = AuditEvent(
            event_type=EventType.SECURITY,
            severity=self._determine_severity(decision, context.risk_score),
            action=ActionType.ACCESS_DENIED if decision == SecurityDecision.DENY else ActionType.READ,
            actor=actor,
            resource=resource,
            outcome="success" if decision == SecurityDecision.ALLOW else "denied",
            context={
                "security_decision": decision.value,
                "risk_score": context.risk_score,
                "threat_level": context.threat_level,
                "auth_method": context.auth_method.value if context.auth_method else None
            },
            correlation_id=context.request_id,
            compliance_tags=context.compliance_tags or []
        )
        
        await self.audit_logger.log_event(event)
    
    async def _log_security_error(self, request: Request, 
                                context: SecurityContext, 
                                error: Exception):
        """Log security processing errors"""
        actor = Actor(
            type="system",
            identifier="security_service"
        )
        
        resource = Resource(
            type="security_processing",
            identifier=context.request_id
        )
        
        event = AuditEvent(
            event_type=EventType.SYSTEM,
            severity=SeverityLevel.HIGH,
            action=ActionType.SYSTEM,
            actor=actor,
            resource=resource,
            outcome="error",
            context={
                "error_type": type(error).__name__,
                "error_message": str(error),
                "request_path": str(request.url.path)
            },
            correlation_id=context.request_id
        )
        
        await self.audit_logger.log_event(event)
    
    # Helper methods
    def _generate_request_id(self) -> str:
        """Generate unique request ID"""
        import uuid
        return str(uuid.uuid4())
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP address"""
        # Check for forwarded headers
        forwarded_for = request.headers.get('x-forwarded-for')
        if forwarded_for:
            return forwarded_for.split(',')[0].strip()
        
        real_ip = request.headers.get('x-real-ip')
        if real_ip:
            return real_ip
        
        return getattr(request.client, 'host', 'unknown')
    
    async def _extract_request_data(self, request: Request) -> Dict[str, Any]:
        """Extract request data for validation"""
        data = {
            'method': request.method,
            'path': request.url.path,
            'query_params': dict(request.query_params),
            'headers': dict(request.headers),
            'cookies': dict(request.cookies)
        }
        
        # Extract body for POST/PUT requests
        if request.method in ['POST', 'PUT', 'PATCH']:
            content_type = request.headers.get('content-type', '')
            if 'application/json' in content_type:
                try:
                    data['body'] = await request.json()
                except:
                    pass
            elif 'application/x-www-form-urlencoded' in content_type:
                try:
                    form_data = await request.form()
                    data['body'] = dict(form_data)
                except:
                    pass
        
        return data
    
    async def _check_rate_limits(self, context: SecurityContext):
        """Check rate limiting"""
        # Implement rate limiting logic
        pass
    
    def _get_required_permission(self, method: str, path: str) -> Optional[str]:
        """Determine required permission for endpoint"""
        # Map HTTP methods to permissions
        method_permissions = {
            'GET': 'read',
            'POST': 'create',
            'PUT': 'update',
            'PATCH': 'update',
            'DELETE': 'delete'
        }
        
        base_permission = method_permissions.get(method, 'read')
        
        # Extract resource from path
        resource = path.split('/')[1] if '/' in path else 'default'
        
        return f"{resource}:{base_permission}"
    
    def _extract_processing_purposes(self, request: Request) -> List[str]:
        """Extract data processing purposes from request"""
        # This would be implemented based on your application logic
        purposes = []
        
        # Example: check for marketing consent in registration
        if '/register' in request.url.path:
            purposes.append('marketing')
        
        # Example: check for analytics consent
        if request.headers.get('x-analytics-enabled'):
            purposes.append('analytics')
        
        return purposes
    
    async def _get_user_location(self, ip_address: str) -> Optional[str]:
        """Get user location from IP address"""
        # Implement IP geolocation
        return None
    
    def _determine_compliance_tags(self, request: Request, 
                                 context: SecurityContext) -> List[str]:
        """Determine compliance tags for the request"""
        tags = []
        
        # Add GDPR tag for EU users
        if context.user_id:
            tags.append('gdpr')
        
        # Add financial tag for financial endpoints
        if '/payment' in request.url.path or '/billing' in request.url.path:
            tags.append('financial')
            tags.append('pci_dss')
        
        # Add health tag for health data
        if '/health' in request.url.path or '/medical' in request.url.path:
            tags.append('hipaa')
        
        return tags
    
    async def _is_suspicious_location(self, ip_address: str, 
                                    user_id: str) -> bool:
        """Check if location is suspicious for user"""
        # Implement location anomaly detection
        return False
    
    async def _detect_anomalous_behavior(self, context: SecurityContext) -> bool:
        """Detect anomalous user behavior"""
        # Implement behavioral anomaly detection
        return False
    
    def _determine_severity(self, decision: SecurityDecision, 
                          risk_score: int) -> SeverityLevel:
        """Determine event severity"""
        if decision == SecurityDecision.DENY:
            return SeverityLevel.HIGH
        elif decision == SecurityDecision.CHALLENGE:
            return SeverityLevel.MEDIUM
        elif risk_score > 50:
            return SeverityLevel.MEDIUM
        else:
            return SeverityLevel.LOW
    
    def _update_metrics(self, context: SecurityContext, 
                       decision: SecurityDecision):
        """Update security metrics"""
        self.metrics.total_requests += 1
        
        if decision == SecurityDecision.DENY:
            self.metrics.blocked_requests += 1
        
        if context.threat_level in ['high', 'critical']:
            self.metrics.threat_detections += 1
        
        # Update average risk score
        total_score = (self.metrics.average_risk_score * 
                      (self.metrics.total_requests - 1) + context.risk_score)
        self.metrics.average_risk_score = total_score / self.metrics.total_requests
        
        self.metrics.last_updated = datetime.now(timezone.utc)
    
    def get_metrics(self) -> SecurityMetrics:
        """Get current security metrics"""
        return self.metrics

# FastAPI Middleware Integration
class SecurityMiddleware:
    """FastAPI middleware for security processing"""
    
    def __init__(self, security_service: UnifiedSecurityService):
        self.security_service = security_service
    
    async def __call__(self, request: Request, call_next):
        """Process request through security pipeline"""
        try:
            # Process security
            context = await self.security_service.process_request(request)
            
            # Add security context to request state
            request.state.security_context = context
            
            # Process request
            response = await call_next(request)
            
            # Add security headers
            self._add_security_headers(response)
            
            return response
            
        except HTTPException as e:
            # Return error response
            from fastapi.responses import JSONResponse
            return JSONResponse(
                status_code=e.status_code,
                content={"detail": e.detail}
            )

    def _add_security_headers(self, response: Response):
        """Add security headers to response"""
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Content-Security-Policy"] = "default-src 'self'"

# FastAPI Dependency
class SecurityDependency:
    """FastAPI dependency for accessing security context"""
    
    def __init__(self, required_permissions: List[str] = None,
                 required_roles: List[str] = None):
        self.required_permissions = required_permissions or []
        self.required_roles = required_roles or []
    
    async def __call__(self, request: Request) -> SecurityContext:
        """Get security context from request"""
        context = getattr(request.state, 'security_context', None)
        
        if not context:
            raise HTTPException(status_code=500, detail="Security context not found")
        
        # Check additional permissions
        if self.required_permissions:
            user_permissions = context.permissions or []
            if not any(perm in user_permissions for perm in self.required_permissions):
                raise HTTPException(status_code=403, detail="Insufficient permissions")
        
        # Check additional roles
        if self.required_roles:
            user_roles = context.roles or []
            if not any(role in user_roles for role in self.required_roles):
                raise HTTPException(status_code=403, detail="Insufficient roles")
        
        return context

# Production Configuration
class ProductionSecurityConfig:
    """Production security configuration"""
    
    @staticmethod
    def create_security_policy() -> SecurityPolicy:
        """Create production security policy"""
        return SecurityPolicy(
            require_authentication=True,
            allowed_auth_methods=[AuthMethod.JWT, AuthMethod.API_KEY],
            mfa_required_for=["admin", "finance"],
            default_access="deny",
            max_request_size=5 * 1024 * 1024,  # 5MB
            validate_all_inputs=True,
            sanitize_outputs=True,
            global_rate_limit=10000,
            per_user_rate_limit=1000,
            data_retention_days=90,
            require_consent=["marketing", "analytics", "profiling"],
            log_all_requests=True,
            log_sensitive_data=False,
            audit_retention_days=2555,  # 7 years
            max_risk_score=60,
            threat_response_actions={
                "medium": "monitor",
                "high": "challenge",
                "critical": "deny"
            }
        )
    
    @staticmethod
    async def create_unified_service(db_session, redis_client, 
                                   config: Dict[str, Any]) -> UnifiedSecurityService:
        """Create configured unified security service"""
        from .input_validation_service import InputValidationService
        from .access_control_service import AccessControlService
        from .privacy_compliance_service import PrivacyComplianceService
        from .audit_logging_service import AuditLoggingService
        from .auth_management_service import AuthenticationService
        
        # Create component services
        input_validator = InputValidationService()
        access_controller = AccessControlService(db_session)
        privacy_manager = PrivacyComplianceService(db_session)
        audit_logger = AuditLoggingService(db_session)
        auth_service = AuthenticationService(
            db_session=db_session,
            redis_client=redis_client,
            secret_key=config['secret_key'],
            jwt_private_key=config['jwt_private_key'],
            jwt_public_key=config['jwt_public_key'],
            issuer=config['jwt_issuer'],
            audience=config['jwt_audience']
        )
        
        # Create unified service
        policy = ProductionSecurityConfig.create_security_policy()
        service = UnifiedSecurityService(
            input_validator=input_validator,
            access_controller=access_controller,
            privacy_manager=privacy_manager,
            audit_logger=audit_logger,
            auth_service=auth_service,
            policy=policy
        )
        
        await service.start()
        return service

# Usage Examples
async def example_fastapi_integration():
    """Example of FastAPI integration"""
    from fastapi import FastAPI, Depends
    
    app = FastAPI()
    
    # Create security service (in production, use dependency injection)
    # security_service = await ProductionSecurityConfig.create_unified_service(...)
    
    # Add middleware
    # app.add_middleware(SecurityMiddleware, security_service=security_service)
    
    # Create dependencies
    require_auth = SecurityDependency()
    require_admin = SecurityDependency(required_roles=["admin"])
    
    @app.get("/protected")
    async def protected_endpoint(context: SecurityContext = Depends(require_auth)):
        return {"message": "Protected data", "user_id": context.user_id}
    
    @app.post("/admin")
    async def admin_endpoint(context: SecurityContext = Depends(require_admin)):
        return {"message": "Admin operation", "user_id": context.user_id}

# Performance Monitoring
class SecurityPerformanceMonitor:
    """Monitor security service performance"""
    
    def __init__(self):
        self.metrics = {
            'request_processing_times': [],
            'threat_detection_times': [],
            'auth_processing_times': [],
            'total_requests_processed': 0,
            'average_processing_time': 0.0
        }
    
    def record_processing_time(self, processing_time: float):
        """Record request processing time"""
        self.metrics['request_processing_times'].append(processing_time)
        self.metrics['total_requests_processed'] += 1
        
        # Calculate rolling average
        recent_times = self.metrics['request_processing_times'][-100:]
        self.metrics['average_processing_time'] = sum(recent_times) / len(recent_times)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        times = self.metrics['request_processing_times']
        
        if not times:
            return self.metrics
        
        return {
            **self.metrics,
            'min_processing_time': min(times),
            'max_processing_time': max(times),
            'p95_processing_time': sorted(times)[int(0.95 * len(times))],
            'p99_processing_time': sorted(times)[int(0.99 * len(times))]
        }