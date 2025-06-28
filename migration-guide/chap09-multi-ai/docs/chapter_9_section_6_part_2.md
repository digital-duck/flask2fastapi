class RateLimitAction(str, Enum):
    ALLOW = "allow"
    THROTTLE = "throttle"
    BLOCK = "block"
    CAPTCHA = "captcha"
    QUARANTINE = "quarantine"

@dataclass
class RateLimitRule:
    """Rate limiting rule definition"""
    rule_id: str
    name: str
    scope: RateLimitScope
    limit: int
    window: RateLimitWindow
    window_size: int = 1  # Number of time units (e.g., 5 minutes = window_size=5, window=MINUTE)
    action: RateLimitAction = RateLimitAction.BLOCK
    burst_allowance: int = 0  # Additional requests allowed in burst
    reset_on_success: bool = False
    priority: int = 1  # Higher priority rules are evaluated first
    conditions: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    created_at: float = field(default_factory=time.time)

@dataclass
class RateLimitViolation:
    """Rate limit violation record"""
    rule_id: str
    scope: RateLimitScope
    identifier: str
    current_count: int
    limit: int
    window_start: float
    window_end: float
    action_taken: RateLimitAction
    timestamp: float = field(default_factory=time.time)
    additional_data: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RateLimitStatus:
    """Current rate limit status for a request"""
    allowed: bool
    remaining: int
    reset_time: float
    retry_after: Optional[float] = None
    violations: List[RateLimitViolation] = field(default_factory=list)
    applied_rules: List[str] = field(default_factory=list)

class AdvancedRateLimiter:
    """Advanced multi-tier rate limiting system"""
    
    def __init__(self, redis_client: Optional[redis.Redis] = None):
        self.redis_client = redis_client or redis.Redis(
            host='localhost', port=6379, db=2, decode_responses=True
        )
        
        # Rate limiting rules
        self.rules: Dict[str, RateLimitRule] = {}
        
        # In-memory cache for frequently accessed counters
        self.memory_cache: Dict[str, Tuple[int, float]] = {}  # key -> (count, window_start)
        self.cache_ttl = 300  # 5 minutes
        
        # Violation tracking
        self.violations: deque = deque(maxlen=10000)  # Keep last 10k violations
        
        # Statistics
        self.stats = {
            'total_requests': 0,
            'blocked_requests': 0,
            'throttled_requests': 0,
            'violations_by_rule': defaultdict(int),
            'violations_by_scope': defaultdict(int),
            'avg_check_time': 0.0
        }
        
        # Initialize default rules
        self._initialize_default_rules()
    
    def _initialize_default_rules(self):
        """Initialize default rate limiting rules"""
        
        default_rules = [
            # Per-user limits
            RateLimitRule(
                rule_id="USER_PER_MINUTE",
                name="User requests per minute",
                scope=RateLimitScope.USER,
                limit=60,
                window=RateLimitWindow.MINUTE,
                action=RateLimitAction.THROTTLE,
                burst_allowance=10
            ),
            
            RateLimitRule(
                rule_id="USER_PER_HOUR",
                name="User requests per hour",
                scope=RateLimitScope.USER,
                limit=1000,
                window=RateLimitWindow.HOUR,
                action=RateLimitAction.BLOCK
            ),
            
            RateLimitRule(
                rule_id="USER_PER_DAY",
                name="User requests per day",
                scope=RateLimitScope.USER,
                limit=10000,
                window=RateLimitWindow.DAY,
                action=RateLimitAction.BLOCK
            ),
            
            # Per-IP limits (more restrictive)
            RateLimitRule(
                rule_id="IP_PER_MINUTE",
                name="IP requests per minute",
                scope=RateLimitScope.IP,
                limit=30,
                window=RateLimitWindow.MINUTE,
                action=RateLimitAction.BLOCK,
                priority=2
            ),
            
            RateLimitRule(
                rule_id="IP_PER_HOUR",
                name="IP requests per hour",
                scope=RateLimitScope.IP,
                limit=500,
                window=RateLimitWindow.HOUR,
                action=RateLimitAction.BLOCK,
                priority=2
            ),
            
            # API key limits
            RateLimitRule(
                rule_id="API_KEY_PER_MINUTE",
                name="API key requests per minute",
                scope=RateLimitScope.API_KEY,
                limit=100,
                window=RateLimitWindow.MINUTE,
                action=RateLimitAction.THROTTLE
            ),
            
            # Global system limits
            RateLimitRule(
                rule_id="GLOBAL_PER_SECOND",
                name="Global system requests per second",
                scope=RateLimitScope.GLOBAL,
                limit=1000,
                window=RateLimitWindow.SECOND,
                action=RateLimitAction.THROTTLE,
                priority=10
            ),
            
            # Endpoint-specific limits
            RateLimitRule(
                rule_id="AI_GENERATION_PER_MINUTE",
                name="AI generation endpoint per minute per user",
                scope=RateLimitScope.USER,
                limit=20,
                window=RateLimitWindow.MINUTE,
                action=RateLimitAction.BLOCK,
                conditions={'endpoint': 'ai_generation'}
            ),
        ]
        
        for rule in default_rules:
            self.rules[rule.rule_id] = rule
        
        logger.info(
            "Default rate limiting rules initialized",
            rule_count=len(default_rules)
        )
    
    async def check_rate_limits(
        self,
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        api_key: Optional[str] = None,
        endpoint: Optional[str] = None,
        organization_id: Optional[str] = None,
        additional_context: Optional[Dict[str, Any]] = None
    ) -> RateLimitStatus:
        """Check rate limits against all applicable rules"""
        
        start_time = time.time()
        
        # Build context for rule evaluation
        context = {
            'endpoint': endpoint,
            **(additional_context or {})
        }
        
        # Identify applicable rules
        applicable_rules = self._get_applicable_rules(context)
        
        # Sort rules by priority (higher priority first)
        applicable_rules.sort(key=lambda r: r.priority, reverse=True)
        
        status = RateLimitStatus(
            allowed=True,
            remaining=float('inf'),
            reset_time=time.time()
        )
        
        violations = []
        
        # Check each applicable rule
        for rule in applicable_rules:
            identifier = self._get_identifier(rule.scope, user_id, ip_address, api_key, organization_id)
            
            if identifier is None:
                continue  # Skip rules where we don't have the required identifier
            
            rule_status = await self._check_single_rule(rule, identifier, context)
            status.applied_rules.append(rule.rule_id)
            
            # Update overall status based on most restrictive rule
            if not rule_status.allowed:
                status.allowed = False
                violations.extend(rule_status.violations)
                
                # Set retry_after to the earliest reset time
                if status.retry_after is None or rule_status.retry_after < status.retry_after:
                    status.retry_after = rule_status.retry_after
            
            # Track the most restrictive remaining count
            if rule_status.remaining < status.remaining:
                status.remaining = rule_status.remaining
                status.reset_time = rule_status.reset_time
        
        status.violations = violations
        
        # Update statistics
        check_time = time.time() - start_time
        self._update_stats(status, check_time)
        
        logger.debug(
            "Rate limit check completed",
            allowed=status.allowed,
            remaining=status.remaining,
            violations_count=len(violations),
            rules_checked=len(status.applied_rules),
            check_time=check_time
        )
        
        return status
    
    async def _check_single_rule(
        self,
        rule: RateLimitRule,
        identifier: str,
        context: Dict[str, Any]
    ) -> RateLimitStatus:
        """Check a single rate limiting rule"""
        
        # Generate cache key
        cache_key = f"rate_limit:{rule.rule_id}:{identifier}"
        
        # Get current window boundaries
        window_start, window_end = self._get_window_boundaries(rule.window, rule.window_size)
        
        # Get current count from Redis
        current_count = await self._get_current_count(cache_key, window_start, window_end)
        
        # Calculate remaining requests
        effective_limit = rule.limit + rule.burst_allowance
        remaining = max(0, effective_limit - current_count)
        
        # Determine if request should be allowed
        allowed = current_count < effective_limit
        
        status = RateLimitStatus(
            allowed=allowed,
            remaining=remaining,
            reset_time=window_end,
            retry_after=window_end - time.time() if not allowed else None
        )
        
        # Record violation if limit exceeded
        if not allowed:
            violation = RateLimitViolation(
                rule_id=rule.rule_id,
                scope=rule.scope,
                identifier=identifier,
                current_count=current_count,
                limit=effective_limit,
                window_start=window_start,
                window_end=window_end,
                action_taken=rule.action,
                additional_data={'context': context}
            )
            
            status.violations.append(violation)
            self.violations.append(violation)
            
            # Update violation statistics
            self.stats['violations_by_rule'][rule.rule_id] += 1
            self.stats['violations_by_scope'][rule.scope.value] += 1
        
        return status
    
    async def increment_counter(
        self,
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        api_key: Optional[str] = None,
        endpoint: Optional[str] = None,
        organization_id: Optional[str] = None,
        additional_context: Optional[Dict[str, Any]] = None
    ):
        """Increment counters for all applicable rules"""
        
        context = {
            'endpoint': endpoint,
            **(additional_context or {})
        }
        
        applicable_rules = self._get_applicable_rules(context)
        
        # Increment counters for each applicable rule
        for rule in applicable_rules:
            identifier = self._get_identifier(rule.scope, user_id, ip_address, api_key, organization_id)
            
            if identifier is None:
                continue
            
            cache_key = f"rate_limit:{rule.rule_id}:{identifier}"
            window_start, window_end = self._get_window_boundaries(rule.window, rule.window_size)
            
            await self._increment_counter(cache_key, window_start, window_end)
    
    def _get_applicable_rules(self, context: Dict[str, Any]) -> List[RateLimitRule]:
        """Get rules applicable to the current context"""
        
        applicable_rules = []
        
        for rule in self.rules.values():
            if not rule.enabled:
                continue
            
            # Check if rule conditions match context
            if self._rule_matches_context(rule, context):
                applicable_rules.append(rule)
        
        return applicable_rules
    
    def _rule_matches_context(self, rule: RateLimitRule, context: Dict[str, Any]) -> bool:
        """Check if rule conditions match the current context"""
        
        for condition_key, condition_value in rule.conditions.items():
            context_value = context.get(condition_key)
            
            if isinstance(condition_value, list):
                if context_value not in condition_value:
                    return False
            elif context_value != condition_value:
                return False
        
        return True
    
    def _get_identifier(
        self,
        scope: RateLimitScope,
        user_id: Optional[str],
        ip_address: Optional[str],
        api_key: Optional[str],
        organization_id: Optional[str]
    ) -> Optional[str]:
        """Get identifier for the specified scope"""
        
        if scope == RateLimitScope.USER:
            return user_id
        elif scope == RateLimitScope.IP:
            return ip_address
        elif scope == RateLimitScope.API_KEY:
            return api_key
        elif scope == RateLimitScope.ORGANIZATION:
            return organization_id
        elif scope == RateLimitScope.GLOBAL:
            return "global"
        elif scope == RateLimitScope.ENDPOINT:
            return f"endpoint:{ip_address or user_id}"  # Fallback for endpoint scope
        
        return None
    
    def _get_window_boundaries(self, window: RateLimitWindow, window_size: int) -> Tuple[float, float]:
        """Get current window boundaries"""
        
        current_time = time.time()
        
        if window == RateLimitWindow.SECOND:
            window_start = int(current_time / window_size) * window_size
            window_end = window_start + window_size
        elif window == RateLimitWindow.MINUTE:
            window_start = int(current_time / (60 * window_size)) * (60 * window_size)
            window_end = window_start + (60 * window_size)
        elif window == RateLimitWindow.HOUR:
            window_start = int(current_time / (3600 * window_size)) * (3600 * window_size)
            window_end = window_start + (3600 * window_size)
        elif window == RateLimitWindow.DAY:
            window_start = int(current_time / (86400 * window_size)) * (86400 * window_size)
            window_end = window_start + (86400 * window_size)
        elif window == RateLimitWindow.MONTH:
            # Simplified monthly calculation (30 days)
            window_start = int(current_time / (2592000 * window_size)) * (2592000 * window_size)
            window_end = window_start + (2592000 * window_size)
        else:
            # Default to minute
            window_start = int(current_time / 60) * 60
            window_end = window_start + 60
        
        return window_start, window_end
    
    async def _get_current_count(self, cache_key: str, window_start: float, window_end: float) -> int:
        """Get current count for the window"""
        
        try:
            # Try Redis first
            count = await self.redis_client.get(cache_key)
            if count is not None:
                return int(count)
            
            # Fallback to memory cache
            if cache_key in self.memory_cache:
                cached_count, cached_window_start = self.memory_cache[cache_key]
                if cached_window_start == window_start:
                    return cached_count
            
            return 0
            
        except Exception as e:
            logger.warning(
                "Failed to get rate limit count from Redis",
                cache_key=cache_key,
                error=str(e)
            )
            # Fallback to memory cache
            if cache_key in self.memory_cache:
                cached_count, cached_window_start = self.memory_cache[cache_key]
                if cached_window_start == window_start:
                    return cached_count
            return 0
    
    async def _increment_counter(self, cache_key: str, window_start: float, window_end: float):
        """Increment counter for the current window"""
        
        try:
            # Redis increment with expiration
            pipe = self.redis_client.pipeline()
            pipe.incr(cache_key)
            pipe.expireat(cache_key, int(window_end))
            await pipe.execute()
            
            # Update memory cache
            current_count = await self.redis_client.get(cache_key)
            if current_count:
                self.memory_cache[cache_key] = (int(current_count), window_start)
            
        except Exception as e:
            logger.warning(
                "Failed to increment rate limit counter in Redis",
                cache_key=cache_key,
                error=str(e)
            )
            # Fallback to memory cache
            if cache_key in self.memory_cache:
                count, cached_window_start = self.memory_cache[cache_key]
                if cached_window_start == window_start:
                    self.memory_cache[cache_key] = (count + 1, window_start)
                else:
                    self.memory_cache[cache_key] = (1, window_start)
            else:
                self.memory_cache[cache_key] = (1, window_start)
    
    def _update_stats(self, status: RateLimitStatus, check_time: float):
        """Update rate limiting statistics"""
        
        self.stats['total_requests'] += 1
        
        if not status.allowed:
            self.stats['blocked_requests'] += 1
        
        # Check if any violations resulted in throttling
        if any(v.action_taken == RateLimitAction.THROTTLE for v in status.violations):
            self.stats['throttled_requests'] += 1
        
        # Update average check time
        current_avg = self.stats['avg_check_time']
        total_requests = self.stats['total_requests']
        
        self.stats['avg_check_time'] = (
            (current_avg * (total_requests - 1) + check_time) / total_requests
        )
    
    def add_rule(self, rule: RateLimitRule) -> bool:
        """Add new rate limiting rule"""
        
        if rule.rule_id in self.rules:
            logger.warning(
                "Rate limiting rule already exists",
                rule_id=rule.rule_id
            )
            return False
        
        self.rules[rule.rule_id] = rule
        
        logger.info(
            "Rate limiting rule added",
            rule_id=rule.rule_id,
            scope=rule.scope.value,
            limit=rule.limit,
            window=rule.window.value
        )
        
        return True
    
    def update_rule(self, rule_id: str, updates: Dict[str, Any]) -> bool:
        """Update existing rate limiting rule"""
        
        if rule_id not in self.rules:
            return False
        
        rule = self.rules[rule_id]
        
        for key, value in updates.items():
            if hasattr(rule, key):
                setattr(rule, key, value)
        
        logger.info(
            "Rate limiting rule updated",
            rule_id=rule_id,
            updates=list(updates.keys())
        )
        
        return True
    
    def remove_rule(self, rule_id: str) -> bool:
        """Remove rate limiting rule"""
        
        if rule_id not in self.rules:
            return False
        
        del self.rules[rule_id]
        
        logger.info("Rate limiting rule removed", rule_id=rule_id)
        
        return True
    
    async def reset_limits(
        self,
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        api_key: Optional[str] = None,
        rule_id: Optional[str] = None
    ) -> int:
        """Reset rate limits for specified identifier or rule"""
        
        reset_count = 0
        
        if rule_id and rule_id in self.rules:
            # Reset specific rule for all identifiers
            pattern = f"rate_limit:{rule_id}:*"
            keys = await self.redis_client.keys(pattern)
            
            if keys:
                await self.redis_client.delete(*keys)
                reset_count = len(keys)
        else:
            # Reset all rules for specific identifiers
            for rule in self.rules.values():
                identifier = self._get_identifier(rule.scope, user_id, ip_address, api_key, None)
                
                if identifier:
                    cache_key = f"rate_limit:{rule.rule_id}:{identifier}"
                    
                    try:
                        deleted = await self.redis_client.delete(cache_key)
                        if deleted:
                            reset_count += 1
                        
                        # Also clear from memory cache
                        if cache_key in self.memory_cache:
                            del self.memory_cache[cache_key]
                            
                    except Exception as e:
                        logger.warning(
                            "Failed to reset rate limit",
                            cache_key=cache_key,
                            error=str(e)
                        )
        
        logger.info(
            "Rate limits reset",
            reset_count=reset_count,
            user_id=user_id,
            ip_address=ip_address,
            api_key=api_key,
            rule_id=rule_id
        )
        
        return reset_count
    
    def get_recent_violations(
        self,
        limit: int = 100,
        scope: Optional[RateLimitScope] = None,
        rule_id: Optional[str] = None
    ) -> List[RateLimitViolation]:
        """Get recent rate limit violations"""
        
        violations = list(self.violations)
        
        # Filter by scope if specified
        if scope:
            violations = [v for v in violations if v.scope == scope]
        
        # Filter by rule if specified
        if rule_id:
            violations = [v for v in violations if v.rule_id == rule_id]
        
        # Sort by timestamp (most recent first) and limit
        violations.sort(key=lambda v: v.timestamp, reverse=True)
        
        return violations[:limit]
    
    def get_rate_limit_stats(self) -> Dict[str, Any]:
        """Get comprehensive rate limiting statistics"""
        
        total_requests = self.stats['total_requests']
        blocked_requests = self.stats['blocked_requests']
        
        return {
            'requests': {
                'total': total_requests,
                'blocked': blocked_requests,
                'throttled': self.stats['throttled_requests'],
                'allowed': total_requests - blocked_requests,
                'block_rate': (blocked_requests / max(1, total_requests)) * 100
            },
            'violations': {
                'by_rule': dict(self.stats['violations_by_rule']),
                'by_scope': dict(self.stats['violations_by_scope']),
                'total_violations': sum(self.stats['violations_by_rule'].values()),
                'recent_violations': len(self.violations)
            },
            'performance': {
                'avg_check_time': self.stats['avg_check_time'],
                'cache_size': len(self.memory_cache)
            },
            'rules': {
                'total_rules': len(self.rules),
                'enabled_rules': len([r for r in self.rules.values() if r.enabled]),
                'rules_by_scope': {
                    scope.value: len([r for r in self.rules.values() if r.scope == scope])
                    for scope in RateLimitScope
                }
            }
        }
```

## Role-Based Access Control (RBAC)

### Comprehensive Access Control System

```python
# security/access_control.py
import time
import json
import hashlib
from typing import Dict, Any, Optional, List, Set, Union
from dataclasses import dataclass, field
from enum import Enum
import structlog
from functools import wraps

logger = structlog.get_logger()

class Permission(str, Enum):
    # AI Service Permissions
    AI_GENERATE = "ai:generate"
    AI_STREAM = "ai:stream"
    AI_BATCH = "ai:batch"
    AI_COLLABORATE = "ai:collaborate"
    
    # Administrative Permissions
    ADMIN_USERS = "admin:users"
    ADMIN_ROLES = "admin:roles"
    ADMIN_POLICIES = "admin:policies"
    ADMIN_MONITORING = "admin:monitoring"
    ADMIN_SYSTEM = "admin:system"
    
    # Data Permissions
    DATA_READ = "data:read"
    DATA_WRITE = "data:write"
    DATA_DELETE = "data:delete"
    DATA_EXPORT = "data:export"
    
    # API Permissions
    API_READ = "api:read"
    API_WRITE = "api:write"
    API_ADMIN = "api:admin"
    
    # Resource Permissions
    RESOURCE_CREATE = "resource:create"
    RESOURCE_UPDATE = "resource:update"
    RESOURCE_DELETE = "resource:delete"
    RESOURCE_SHARE = "resource:share"

class ResourceType(str, Enum):
    AI_MODEL = "ai_model"
    AI_SESSION = "ai_session"
    USER_DATA = "user_data"
    ORGANIZATION = "organization"
    API_KEY = "api_key"
    SYSTEM_CONFIG = "system_config"

@dataclass
class Role:
    """Role definition with permissions and constraints"""
    role_id: str
    name: str
    description: str
    permissions: Set[Permission] = field(default_factory=set)
    resource_constraints: Dict[ResourceType, Dict[str, Any]] = field(default_factory=dict)
    inherits_from: List[str] = field(default_factory=list)
    is_system_role: bool = False
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    
    def has_permission(self, permission: Permission) -> bool:
        """Check if role has specific permission"""
        return permission in self.permissions
    
    def add_permission(self, permission: Permission):
        """Add permission to role"""
        self.permissions.add(permission)
        self.updated_at = time.time()
    
    def remove_permission(self, permission: Permission):
        """Remove permission from role"""
        self.permissions.discard(permission)
        self.updated_at = time.time()

@dataclass
class User:
    """User with roles and additional permissions"""
    user_id: str
    username: str
    email: str
    roles: Set[str] = field(default_factory=set)
    additional_permissions: Set[Permission] = field(default_factory=set)
    denied_permissions: Set[Permission] = field(default_factory=set)
    organization_id: Optional[str] = None
    is_active: bool = True
    is_verified: bool = False
    last_login: Optional[float] = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AccessRequest:
    """Access control request"""
    user_id: str
    permission: Permission
    resource_type: Optional[ResourceType] = None
    resource_id: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

@dataclass
class AccessResult:
    """Access control result"""
    granted: bool
    user_id: str
    permission: Permission
    resource_type: Optional[ResourceType] = None
    resource_id: Optional[str] = None
    reason: str = ""
    matched_roles: List[str] = field(default_factory=list)
    applied_constraints: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

class AccessControlManager:
    """Comprehensive role-based access control system"""
    
    def __init__(self):
        self.roles: Dict[str, Role] = {}
        self.users: Dict[str, User] = {}
        
        # Access control cache for performance
        self.permission_cache: Dict[str, Dict[Permission, bool]] = {}
        self.cache_ttl = 300  # 5 minutes
        self.cache_timestamps: Dict[str, float] = {}
        
        # Audit trail
        self.access_log: List[AccessResult] = []
        self.max_log_size = 10000
        
        # Statistics
        self.stats = {
            'total_checks': 0,
            'granted_access': 0,
            'denied_access': 0,
            'cache_hits': 0,
            'avg_check_time': 0.0
        }
        
        # Initialize default roles
        self._initialize_default_roles()
    
    def _initialize_default_roles(self):
        """Initialize default system roles"""
        
        # Super Admin Role
        super_admin = Role(
            role_id="super_admin",
            name="Super Administrator",
            description="Full system access",
            permissions=set(Permission),  # All permissions
            is_system_role=True
        )
        
        # Admin Role
        admin = Role(
            role_id="admin",
            name="Administrator",
            description="Administrative access",
            permissions={
                Permission.AI_GENERATE,
                Permission.AI_STREAM,
                Permission.AI_BATCH,
                Permission.AI_COLLABORATE,
                Permission.ADMIN_USERS,
                Permission.ADMIN_ROLES,
                Permission.ADMIN_POLICIES,
                Permission.ADMIN_MONITORING,
                Permission.DATA_READ,
                Permission.DATA_WRITE,
                Permission.DATA_EXPORT,
                Permission.API_READ,
                Permission.API_WRITE,
                Permission.RESOURCE_CREATE,
                Permission.RESOURCE_UPDATE,
                Permission.RESOURCE_DELETE,
                Permission.RESOURCE_SHARE
            },
            is_system_role=True
        )
        
        # User Role
        user = Role(
            role_id="user",
            name="Standard User",
            description="Standard user access",
            permissions={
                Permission.AI_GENERATE,
                Permission.AI_STREAM,
                Permission.AI_COLLABORATE,
                Permission.DATA_READ,
                Permission.API_READ,
                Permission.RESOURCE_CREATE,
                Permission.RESOURCE_UPDATE
            },
            is_system_role=True
        )
        
        # Read-only Role
        readonly = Role(
            role_id="readonly",
            name="Read-only User",
            description="Read-only access",
            permissions={
                Permission.DATA_READ,
                Permission.API_READ
            },
            is_system_role=True
        )
        
        # API-only Role
        api_only = Role(
            role_id="api_only",
            name="API-only Access",
            description="API access only",
            permissions={
                Permission.AI_GENERATE,
                Permission.API_READ,
                Permission.API_WRITE
            },
            is_system_role=True
        )
        
        # Store default roles
        self.roles.update({
            "super_admin": super_admin,
            "admin": admin,
            "user": user,
            "readonly": readonly,
            "api_only": api_only
        })
        
        logger.info(
            "Default roles initialized",
            role_count=len(self.roles)
        )
    
    async def check_permission(
        self,
        user_id: str,
        permission: Permission,
        resource_type: Optional[ResourceType] = None,
        resource_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> AccessResult:
        """Check if user has permission for specific action"""
        
        start_time = time.time()
        context = context or {}
        
        # Create access request
        request = AccessRequest(
            user_id=user_id,
            permission=permission,
            resource_type=resource_type,
            resource_id=resource_id,
            context=context
        )
        
        # Check cache first
        cache_key = self._get_cache_key(user_id, permission, resource_type, resource_id)
        cached_result = self._get_from_cache(cache_key)
        
        if cached_result is not None:
            self.stats['cache_hits'] += 1
            self.stats['total_checks'] += 1
            return cached_result
        
        # Perform access check
        result = await self._perform_access_check(request)
        
        # Cache result
        self._store_in_cache(cache_key, result)
        
        # Update statistics
        check_time = time.time() - start_time
        self._update_access_stats(result, check_time)
        
        # Log access attempt
        self._log_access_attempt(result)
        
        logger.debug(
            "Permission check completed",
            user_id=user_id,
            permission=permission.value,
            granted=result.granted,
            reason=result.reason,
            check_time=check_time
        )
        
        return result
    
    async def _perform_access_check(self, request: AccessRequest) -> AccessResult:
        """Perform the actual access control check"""
        
        user = self.users.get(request.user_id)
        if not user:
            return AccessResult(
                granted=False,
                user_id=request.user_id,
                permission=request.permission,
                resource_type=request.resource_type,
                resource_id=request.resource_id,
                reason="User not found"
            )
        
        if not user.is_active:
            return AccessResult(
                granted=False,
                user_id=request.user_id,
                permission=request.permission,
                resource_type=request.resource_type,
                resource_id=request.resource_id,
                reason="User account is inactive"
            )
        
        # Check denied permissions first
        if request.permission in user.denied_permissions:
            return AccessResult(
                granted=False,
                user_id=request.user_id,
                permission=request.permission,
                resource_type=request.resource_type,
                resource_id=request.resource_id,
                reason="Permission explicitly denied"
            )
        
        # Check additional permissions
        if request.permission in user.additional_permissions:
            return AccessResult(
                granted=True,
                user_id=request.user_id,
                permission=request.permission,
                resource_type=request.resource_type,
                resource_id=request.resource_id,
                reason="Additional permission granted",
                matched_roles=["additional_permissions"]
            )
        
        # Check role-based permissions
        granted_roles = []
        all_permissions = set()
        
        for role_id in user.roles:
            role = self.roles.get(role_id)
            if not role:
                continue
            
            # Get all permissions from role and inherited roles
            role_permissions = self._get_effective_permissions(role)
            all_permissions.update(role_permissions)
            
            if request.permission in role_permissions:
                granted_roles.append(role_id)
        
        if request.permission in all_permissions:
            # Check resource constraints
            constraint_check = await self._check_resource_constraints(
                user, request, granted_roles
            )
            
            if constraint_check['allowed']:
                return AccessResult(
                    granted=True,
                    user_id=request.user_id,
                    permission=request.permission,
                    resource_type=request.resource_type,
                    resource_id=request.resource_id,
                    reason="Permission granted via roles",
                    matched_roles=granted_roles,
                    applied_constraints=constraint_check['constraints']
                )
            else:
                return AccessResult(
                    granted=False,
                    user_id=request.user_id,
                    permission=request.permission,
                    resource_type=request.resource_type,
                    resource_id=request.resource_id,
                    reason=f"Resource constraint violation: {constraint_check['reason']}",
                    matched_roles=granted_roles
                )
        
        return AccessResult(
            granted=False,
            user_id=request.user_id,
            permission=request.permission,
            resource_type=request.resource_type,
            resource_id=request.resource_id,
            reason="Permission not granted by any role"
        )
    
    def _get_effective_permissions(self, role: Role) -> Set[Permission]:
        """Get all effective permissions for a role including inherited"""
        
        permissions = set(role.permissions)
        
        # Add permissions from inherited roles
        for parent_role_id in role.inherits_from:
            parent_role = self.roles.get(parent_role_id)
            if parent_role:
                permissions.update(self._get_effective_permissions(parent_role))
        
        return permissions
    
    async def _check_resource_constraints(
        self,
        user: User,
        request: AccessRequest,
        granted_roles: List[str]
    ) -> Dict[str, Any]:
        """Check resource-level constraints"""
        
        if not request.resource_type:
            return {'allowed': True, 'constraints': {}, 'reason': ''}
        
        applied_constraints = {}
        
        # Check constraints from all granted roles
        for role_id in granted_roles:
            role = self.roles.get(role_id)
            if not role:
                continue
            
            resource_constraints = role.resource_constraints.get(request.resource_type, {})
            
            for constraint_type, constraint_value in resource_constraints.items():
                if constraint_type == 'owner_only' and constraint_value:
                    # Check if user owns the resource
                    if not await self._check_resource_ownership(user.user_id, request.resource_type, request.resource_id):
                        return {
                            'allowed': False,
                            'constraints': applied_constraints,
                            'reason': 'Resource access restricted to owner'
                        }
                    applied_constraints['owner_only'] = True
                
                elif constraint_type == 'organization_only' and constraint_value:
                    # Check if resource belongs to user's organization
                    if not await self._check_organization_access(user, request.resource_type, request.resource_id):
                        return {
                            'allowed': False,
                            'constraints': applied_constraints,
                            'reason': 'Resource access restricted to organization'
                        }
                    applied_constraints['organization_only'] = True
                
                elif constraint_type == 'time_based':
                    # Check time-based access constraints
                    if not self._check_time_constraints(constraint_value):
                        return {
                            'allowed': False,
                            'constraints': applied_constraints,
                            'reason': 'Access not allowed at current time'
                        }
                    applied_constraints['time_based'] = constraint_value
        
        return {'allowed': True, 'constraints': applied_constraints, 'reason': ''}
    
    async def _check_resource_ownership(
        self,
        user_id: str,
        resource_type: ResourceType,
        resource_id: Optional[str]
    ) -> bool:
        """Check if user owns the specified resource"""
        
        # This would integrate with your resource management system
        # For now, we'll return a simplified check
        
        if not resource_id:
            return True  # No specific resource to check
        
        # Simplified ownership check (in production, query your database)
        if resource_type == ResourceType.AI_SESSION:
            # Check if AI session belongs to user
            return resource_id.startswith(user_id)
        elif resource_type == ResourceType.USER_DATA:
            # Check if user data belongs to user
            return resource_id == user_id or resource_id.startswith(f"{user_id}_")
        
        return True  # Default allow for other resource types
    
    async def _check_organization_access(
        self,
        user: User,
        resource_type: ResourceType,
        resource_id: Optional[str]
    ) -> bool:
        """Check if resource belongs to user's organization"""
        
        if not user.organization_id or not resource_id:
            return True
        
        # This would integrate with your organization management system
        # For now, we'll return a simplified check
        
        # Simplified organization check
        return resource_id.find(user.organization_id) != -1
    
    def _check_time_constraints(self, time_constraint: Dict[str, Any]) -> bool:
        """Check time-based access constraints"""
        
        current_time = time.time()
        current_hour = time.localtime(current_time).tm_hour
        current_day = time.localtime(current_time).tm_wday  # 0=Monday, 6=Sunday
        
        # Check allowed hours
        if 'allowed_hours' in time_constraint:
            allowed_hours = time_constraint['allowed_hours']
            if current_hour not in allowed_hours:
                return False
        
        # Check allowed days
        if 'allowed_days' in time_constraint:
            allowed_days = time_constraint['allowed_days']
            if current_day not in allowed_days:
                return False
        
        # Check date range
        if 'start_date' in time_constraint and 'end_date' in time_constraint:
            start_date = time_constraint['start_date']
            end_date = time_constraint['end_date']
            if not (start_date <= current_time <= end_date):
                return False
        
        return True
    
    def _get_cache_key(
        self,
        user_id: str,
        permission: Permission,
        resource_type: Optional[ResourceType],
        resource_id: Optional[str]
    ) -> str:
        """Generate cache key for permission check"""
        
        key_parts = [user_id, permission.value]
        
        if resource_type:
            key_parts.append(resource_type.value)
        if resource_id:
            key_parts.append(resource_id)
        
        combined_key = ":".join(key_parts)
        return hashlib.md5(combined_key.encode()).hexdigest()
    
    def _get_from_cache(self, cache_key: str) -> Optional[AccessResult]:
        """Get permission result from cache"""
        
        if cache_key not in self.permission_cache:
            return None
        
        cache_time = self.cache_timestamps.get(cache_key, 0)
        if time.time() - cache_time > self.cache_ttl:
            # Cache expired
            del self.permission_cache[cache_key]
            del self.cache_timestamps[cache_key]
            return None
        
        cached_permissions = self.permission_cache[cache_key]
        # For simplicity, we'll return the first cached result
        # In a real implementation, you'd cache the full AccessResult
        return None  # Simplified for this example
    
    def _store_in_cache(self, cache_key: str, result: AccessResult):
        """Store permission result in cache"""
        
        if cache_key not in self.permission_cache:
            self.permission_cache[cache_key] = {}
        
        self.permission_cache[cache_key][result.permission] = result.granted
        self.cache_timestamps[cache_key] = time.time()
        
        # Clean up old cache entries periodically
        if len(self.permission_cache) > 10000:
            self._cleanup_cache()
    
    def _cleanup_cache(self):
        """Clean up expired cache entries"""
        
        current_time = time.time()
        expired_keys = []
        
        for cache_key, timestamp in self.cache_timestamps.items():
            if current_time - timestamp > self.cache_ttl:
                expired_keys.append(cache_key)
        
        for key in expired_keys:
            if key in self.permission_cache:
                del self.permission_cache[key]
            if key in self.cache_timestamps:
                del self.cache_timestamps[key]
    
    def _update_access_stats(self, result: AccessResult, check_time: float):
        """Update access control statistics"""
        
        self.stats['total_checks'] += 1
        
        if result.granted:
            self.stats['granted_access'] += 1
        else:
            self.stats['denied_access'] += 1
        
        # Update average check time
        current_avg = self.stats['avg_check_time']
        total_checks = self.stats['total_checks']
        
        self.stats['avg_check_time'] = (
            (current_avg * (total_checks - 1) + check_time) / total_checks
        )
    
    def _log_access_attempt(self, result: AccessResult):
        """Log access attempt for audit trail"""
        
        self.access_log.append(result)
        
        # Limit log size
        if len(self.access_log) > self.max_log_size:
            self.access_log = self.access_log[-self.max_log_size:]
    
    def create_role(
        self,
        role_id: str,
        name: str,
        description: str,
        permissions: Optional[Set[Permission]] = None,
        inherits_from: Optional[List[str]] = None
    ) -> Role:
        """Create new role"""
        
        if role_id in self.roles:
            raise ValueError(f"Role {role_id} already exists")
        
        role = Role(
            role_id=role_id,
            name=name,
            description=description,
            permissions=permissions or set(),
            inherits_from=inherits_from or []
        )
        
        self.roles[role_id] = role
        
        logger.info(
            "Role created",
            role_id=role_id,
            name=name,
            permission_count=len(role.permissions)
        )
        
        return role
    
    def create_user(
        self,
        user_id: str,
        username: str,
        email: str,
        roles: Optional[Set[str]] = None,
        organization_id: Optional[str] = None
    ) -> User:
        """Create new user"""
        
        if user_id in self.users:
            raise ValueError(f"User {user_id} already exists")
        
        user = User(
            user_id=user_id,
            username=username,
            email=email,
            roles=roles or set(),
            organization_id=organization_id
        )
        
        self.users[user_id] = user
        
        logger.info(
            "User created",
            user_id=user_id,
            username=username,
            roles=list(user.roles),
            organization_id=organization_id
        )
        
        return user
    
    def assign_role(self, user_id: str, role_id: str) -> bool:
        """Assign role to user"""
        
        user = self.users.get(user_id)
        if not user:
            return False
        
        if role_id not in self.roles:
            return False
        
        user.roles.add(role_id)
        user.updated_at = time.time()
        
        # Invalidate cache for this user
        self._invalidate_user_cache(user_id)
        
        logger.info(
            "Role assigned",
            user_id=user_id,
            role_id=role_id
        )
        
        return True
    
    def revoke_role(self, user_id: str, role_id: str) -> bool:
        """Revoke role from user"""
        
        user = self.users.get(user_id)
        if not user:
            return False
        
        user.roles.discard(role_id)
        user.updated_at = time.time()
        
        # Invalidate cache for this user
        self._invalidate_user_cache(user_id)
        
        logger.info(
            "Role revoked",
            user_id=user_id,
            role_id=role_id
        )
        
        return True
    
    def grant_permission(self, user_id: str, permission: Permission) -> bool:
        """Grant additional permission to user"""
        
        user = self.users.get(user_id)
        if not user:
            return False
        
        user.additional_permissions.add(permission)
        user.denied_permissions.discard(permission)  # Remove from denied if present
        user.updated_at = time.time()
        
        # Invalidate cache for this user
        self._invalidate_user_cache(user_id)
        
        logger.info(
            "Permission granted",
            user_id=user_id,
            permission=permission.value
        )
        
        return True
    
    def deny_permission(self, user_id: str, permission: Permission) -> bool:
        """Explicitly deny permission to user"""
        
        user = self.users.get(user_id)
        if not user:
            return False
        
        user.denied_permissions.add(permission)
        user.additional_permissions.discard(permission)  # Remove from additional if present
        user.updated_at = time.time()
        
        # Invalidate cache for this user
        self._invalidate_user_cache(user_id)
        
        logger.info(
            "Permission denied",
            user_id=user_id,
            permission=permission.value
        )
        
        return True
    
    def _invalidate_user_cache(self, user_id: str):
        """Invalidate cache entries for specific user"""
        
        keys_to_remove = []
        
        for cache_key in self.permission_cache.keys():
            # Check if cache key belongs to this user
            # This is a simplified check - in production you'd have a more robust mapping
            if user_id in cache_key:
                keys_to_remove.append(cache_key)
        
        for key in keys_to_remove:
            if key in self.permission_cache:
                del self.permission_cache[key]
            if key in self.cache_timestamps:
                del self.cache_timestamps[key]
    
    def get_user_permissions(self, user_id: str) -> Set[Permission]:
        """Get all effective permissions for a user"""
        
        user = self.users.get(user_id)
        if not user:
            return set()
        
        all_permissions = set(user.additional_permissions)
        
        # Add permissions from roles
        for role_id in user.roles:
            role = self.roles.get(role_id)
            if role:
                all_permissions.update(self._get_effective_permissions(role))
        
        # Remove denied permissions
        all_permissions -= user.denied_permissions
        
        return all_permissions
    
    def get_access_stats(self) -> Dict[str, Any]:
        """Get access control statistics"""
        
        total_checks = self.stats['total_checks']
        
        return {
            'access_checks': {
                'total': total_checks,
                'granted': self.stats['granted_access'],
                'denied': self.stats['denied_access'],
                'grant_rate': (self.stats['granted_access'] / max(1, total_checks)) * 100
            },
            'cache_performance': {
                'cache_hits': self.stats['cache_hits'],
                'cache_hit_rate': (self.stats['cache_hits'] / max(1, total_checks)) * 100,
                'cache_size': len(self.permission_cache)
            },
            'performance': {
                'avg_check_time': self.stats['avg_check_time']
            },
            'system_overview': {
                'total_users': len(self.users),
                'total_roles': len(self.roles),
                'active_users': len([u for u in self.users.values() if u.is_active]),
                'system_roles': len([r for r in self.roles.values() if r.is_system_role])
            }
        }
    
    def get_recent_access_attempts(
        self,
        limit: int = 100,
        user_id: Optional[str] = None,
        granted_only: Optional[bool] = None
    ) -> List[AccessResult]:
        """Get recent access attempts for audit"""
        
        attempts = list(self.access_log)
        
        # Filter by user if specified
        if user_id:
            attempts = [a for a in attempts if a.user_id == user_id]
        
        # Filter by granted/denied if specified
        if granted_only is not None:
            attempts = [a for a in attempts if a.granted == granted_only]
        
        # Sort by timestamp (most recent first) and limit
        attempts.sort(key=lambda a: a.timestamp, reverse=True)
        
        return attempts[:limit]

# Decorator for permission checking
def require_permission(permission: Permission, resource_type: Optional[ResourceType] = None):
    """Decorator to require specific permission for function access"""
    
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract user_id from function arguments or context
            # This is a simplified example - in practice you'd get this from your auth system
            user_id = kwargs.get('user_id') or (args[0] if args else None)
            resource_id = kwargs.get('resource_id')
            
            if not user_id:
                raise PermissionError("User ID not provided")
            
            # Get access control manager from context (simplified)
            # In practice, this would be dependency injected
            acm = kwargs.get('access_control_manager')
            if not acm:
                raise RuntimeError("Access control manager not available")
            
            # Check permission
            result = await acm.check_permission(
                user_id=user_id,
                permission=permission,
                resource_type=resource_type,
                resource_id=resource_id
            )
            
            if not result.granted:
                raise PermissionError(f"Access denied: {result.reason}")
            
            return await func(*args, **kwargs)
        
        return wrapper
    return decorator
```

This comprehensive rate limiting and access control system provides:

### **Key Features:**

1. **Multi-Tier Rate Limiting**: User, IP, API key, endpoint, and global rate limits
2. **Flexible Time Windows**: Second, minute, hour, day, and month-based windows
3. **Burst Allowance**: Additional requests allowed in burst scenarios
4. **Role-Based Access Control**: Hierarchical roles with permission inheritance
5. **Resource Constraints**: Fine-grained access control based on resource ownership
6. **Caching System**: High-performance permission checking with TTL-based cache
7. **Audit Trail**: Comprehensive logging of all access attempts
8. **Real-time Statistics**: Detailed metrics and monitoring capabilities

The next part will cover **Data Privacy and Compliance** to complete the security framework.# Chapter 9, Section 6.2: Rate Limiting and Access Control

## Overview

This section implements comprehensive rate limiting, access control, and abuse prevention mechanisms to protect AI services from malicious usage while ensuring legitimate users have appropriate access. We'll cover multi-tier rate limiting, role-based access control, and advanced threat detection.

## Advanced Rate Limiting System

### Multi-Tier Rate Limiting Engine

```python
# security/rate_limiting.py
import asyncio
import time
import json
import hashlib
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import structlog
from collections import defaultdict, deque
import redis.asyncio as redis

logger = structlog.get_logger()

class RateLimitScope(str, Enum):
    USER = "user"
    IP = "ip"
    API_KEY = "api_key"
    ENDPOINT = "endpoint"
    GLOBAL = "global"
    ORGANIZATION = "organization"

class RateLimitWindow(str, Enum):
    SECOND = "second"
    MINUTE = "minute"
    HOUR = "hour"
    DAY = "day"
    MONTH = "month"

class RateLimitAction(str, Enum):
    ALLOW = "allow"
    THROTTLE = "throttle"
    BLOCK = "block"
    CAPT