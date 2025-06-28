# File: src/security/audit_logging_service.py
"""
Comprehensive audit logging service for security event tracking and compliance.
Implements structured logging with real-time monitoring and compliance reporting.
"""

import json
import asyncio
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from contextlib import asynccontextmanager
import structlog
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import Column, Integer, String, DateTime, JSON, Text, Index
from sqlalchemy.dialects.postgresql import UUID
import uuid

# Event Classification
class EventType(str, Enum):
    SECURITY = "security"
    ACCESS = "access"
    DATA = "data"
    SYSTEM = "system"
    COMPLIANCE = "compliance"

class SeverityLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ActionType(str, Enum):
    CREATE = "create"
    READ = "read"
    UPDATE = "update"
    DELETE = "delete"
    LOGIN = "login"
    LOGOUT = "logout"
    ACCESS_DENIED = "access_denied"
    PERMISSION_ESCALATION = "permission_escalation"
    DATA_EXPORT = "data_export"
    POLICY_VIOLATION = "policy_violation"

# Core Data Models
@dataclass
class Actor:
    """Represents the entity performing an action"""
    type: str  # user, system, service, anonymous
    identifier: str  # user_id, service_name, etc.
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    session_id: Optional[str] = None

@dataclass
class Resource:
    """Represents the resource being accessed"""
    type: str  # endpoint, database, file, etc.
    identifier: str  # specific resource ID
    attributes: Optional[Dict[str, Any]] = None

@dataclass
class AuditEvent:
    """Core audit event structure"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    event_type: EventType
    severity: SeverityLevel
    action: ActionType
    actor: Actor
    resource: Resource
    outcome: str  # success, failure, partial
    context: Dict[str, Any] = Field(default_factory=dict)
    risk_score: int = Field(default=0, ge=0, le=100)
    correlation_id: Optional[str] = None
    compliance_tags: List[str] = Field(default_factory=list)

# Database Model
class AuditLog(Base):
    __tablename__ = "audit_logs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    timestamp = Column(DateTime(timezone=True), nullable=False, index=True)
    event_type = Column(String(50), nullable=False, index=True)
    severity = Column(String(20), nullable=False, index=True)
    action = Column(String(50), nullable=False, index=True)
    actor_data = Column(JSON, nullable=False)
    resource_data = Column(JSON, nullable=False)
    outcome = Column(String(20), nullable=False)
    context = Column(JSON)
    risk_score = Column(Integer, default=0)
    correlation_id = Column(String(100), index=True)
    compliance_tags = Column(JSON)
    
    # Indexes for common queries
    __table_args__ = (
        Index('idx_audit_timestamp_severity', 'timestamp', 'severity'),
        Index('idx_audit_actor_action', 'actor_data', 'action'),
        Index('idx_audit_compliance', 'compliance_tags'),
    )

# Risk Assessment Engine
class RiskAssessmentEngine:
    """Calculates risk scores for audit events"""
    
    SEVERITY_WEIGHTS = {
        SeverityLevel.LOW: 10,
        SeverityLevel.MEDIUM: 30,
        SeverityLevel.HIGH: 60,
        SeverityLevel.CRITICAL: 100
    }
    
    ACTION_WEIGHTS = {
        ActionType.LOGIN: 5,
        ActionType.READ: 10,
        ActionType.CREATE: 20,
        ActionType.UPDATE: 25,
        ActionType.DELETE: 40,
        ActionType.ACCESS_DENIED: 50,
        ActionType.PERMISSION_ESCALATION: 80,
        ActionType.DATA_EXPORT: 60,
        ActionType.POLICY_VIOLATION: 70
    }
    
    @classmethod
    def calculate_risk_score(cls, event: AuditEvent) -> int:
        """Calculate risk score based on event characteristics"""
        base_score = cls.SEVERITY_WEIGHTS.get(event.severity, 0)
        action_score = cls.ACTION_WEIGHTS.get(event.action, 0)
        
        # Additional risk factors
        context_multiplier = 1.0
        
        # Failed operations increase risk
        if event.outcome == "failure":
            context_multiplier += 0.3
        
        # Multiple failed attempts (from context)
        failed_attempts = event.context.get("failed_attempts", 0)
        if failed_attempts > 3:
            context_multiplier += 0.5
        
        # Off-hours access
        if cls._is_off_hours(event.timestamp):
            context_multiplier += 0.2
        
        # Suspicious IP patterns
        if event.context.get("suspicious_ip", False):
            context_multiplier += 0.4
        
        final_score = int((base_score + action_score) * context_multiplier)
        return min(final_score, 100)  # Cap at 100
    
    @staticmethod
    def _is_off_hours(timestamp: datetime) -> bool:
        """Check if timestamp is during off-hours (9 PM - 6 AM)"""
        hour = timestamp.hour
        return hour >= 21 or hour <= 6

# Audit Logging Service
class AuditLoggingService:
    """Main audit logging service"""
    
    def __init__(self, db_session: AsyncSession, 
                 log_processor: Optional['LogProcessor'] = None):
        self.db_session = db_session
        self.log_processor = log_processor or LogProcessor()
        self.logger = structlog.get_logger(__name__)
        self._event_queue = asyncio.Queue()
        self._processing_task = None
    
    async def start(self):
        """Start the audit logging service"""
        self._processing_task = asyncio.create_task(self._process_events())
        self.logger.info("Audit logging service started")
    
    async def stop(self):
        """Stop the audit logging service"""
        if self._processing_task:
            self._processing_task.cancel()
            try:
                await self._processing_task
            except asyncio.CancelledError:
                pass
        self.logger.info("Audit logging service stopped")
    
    async def log_event(self, event: AuditEvent) -> str:
        """Log an audit event"""
        # Calculate risk score
        event.risk_score = RiskAssessmentEngine.calculate_risk_score(event)
        
        # Add to processing queue
        await self._event_queue.put(event)
        
        # Log to structured logger immediately for real-time monitoring
        await self._log_structured_event(event)
        
        return event.id
    
    async def _log_structured_event(self, event: AuditEvent):
        """Log event to structured logger"""
        log_data = {
            "event_id": event.id,
            "timestamp": event.timestamp.isoformat(),
            "event_type": event.event_type,
            "severity": event.severity,
            "action": event.action,
            "actor": asdict(event.actor),
            "resource": asdict(event.resource),
            "outcome": event.outcome,
            "risk_score": event.risk_score,
            "context": event.context,
            "correlation_id": event.correlation_id,
            "compliance_tags": event.compliance_tags
        }
        
        if event.severity in [SeverityLevel.HIGH, SeverityLevel.CRITICAL]:
            self.logger.warning("High severity audit event", **log_data)
        else:
            self.logger.info("Audit event", **log_data)
    
    async def _process_events(self):
        """Background task to process audit events"""
        while True:
            try:
                event = await self._event_queue.get()
                await self._store_event(event)
                await self.log_processor.process_event(event)
                self._event_queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("Error processing audit event", error=str(e))
    
    async def _store_event(self, event: AuditEvent):
        """Store event in database"""
        audit_log = AuditLog(
            id=uuid.UUID(event.id),
            timestamp=event.timestamp,
            event_type=event.event_type,
            severity=event.severity,
            action=event.action,
            actor_data=asdict(event.actor),
            resource_data=asdict(event.resource),
            outcome=event.outcome,
            context=event.context,
            risk_score=event.risk_score,
            correlation_id=event.correlation_id,
            compliance_tags=event.compliance_tags
        )
        
        self.db_session.add(audit_log)
        await self.db_session.commit()
    
    async def search_events(self, 
                          start_time: Optional[datetime] = None,
                          end_time: Optional[datetime] = None,
                          event_type: Optional[EventType] = None,
                          severity: Optional[SeverityLevel] = None,
                          actor_id: Optional[str] = None,
                          limit: int = 100) -> List[AuditEvent]:
        """Search audit events with filters"""
        query = select(AuditLog)
        
        if start_time:
            query = query.where(AuditLog.timestamp >= start_time)
        if end_time:
            query = query.where(AuditLog.timestamp <= end_time)
        if event_type:
            query = query.where(AuditLog.event_type == event_type)
        if severity:
            query = query.where(AuditLog.severity == severity)
        if actor_id:
            query = query.where(AuditLog.actor_data['identifier'].astext == actor_id)
        
        query = query.order_by(AuditLog.timestamp.desc()).limit(limit)
        
        result = await self.db_session.execute(query)
        logs = result.scalars().all()
        
        # Convert back to AuditEvent objects
        events = []
        for log in logs:
            actor = Actor(**log.actor_data)
            resource = Resource(**log.resource_data)
            event = AuditEvent(
                id=str(log.id),
                timestamp=log.timestamp,
                event_type=EventType(log.event_type),
                severity=SeverityLevel(log.severity),
                action=ActionType(log.action),
                actor=actor,
                resource=resource,
                outcome=log.outcome,
                context=log.context or {},
                risk_score=log.risk_score,
                correlation_id=log.correlation_id,
                compliance_tags=log.compliance_tags or []
            )
            events.append(event)
        
        return events

# Log Processing and Analysis
class LogProcessor:
    """Processes audit events for real-time analysis"""
    
    def __init__(self):
        self.alert_manager = AlertManager()
        self.pattern_detector = PatternDetector()
    
    async def process_event(self, event: AuditEvent):
        """Process a single audit event"""
        # Check for immediate alerts
        if event.severity in [SeverityLevel.HIGH, SeverityLevel.CRITICAL]:
            await self.alert_manager.send_alert(event)
        
        # Pattern detection
        await self.pattern_detector.analyze_event(event)
        
        # Custom processing based on event type
        if event.event_type == EventType.SECURITY:
            await self._process_security_event(event)
        elif event.event_type == EventType.COMPLIANCE:
            await self._process_compliance_event(event)
    
    async def _process_security_event(self, event: AuditEvent):
        """Process security-specific events"""
        # Implement security-specific processing
        pass
    
    async def _process_compliance_event(self, event: AuditEvent):
        """Process compliance-specific events"""
        # Implement compliance-specific processing
        pass

# Alert Management
class AlertManager:
    """Manages security alerts"""
    
    async def send_alert(self, event: AuditEvent):
        """Send alert for high-severity events"""
        # Implement alert sending logic (email, Slack, etc.)
        pass

# Pattern Detection
class PatternDetector:
    """Detects suspicious patterns in audit events"""
    
    async def analyze_event(self, event: AuditEvent):
        """Analyze event for suspicious patterns"""
        # Implement pattern detection logic
        pass

# Compliance Reporting
class ComplianceReporter:
    """Generates compliance reports"""
    
    def __init__(self, audit_service: AuditLoggingService):
        self.audit_service = audit_service
    
    async def generate_sox_report(self, start_date: datetime, 
                                end_date: datetime) -> Dict[str, Any]:
        """Generate SOX compliance report"""
        events = await self.audit_service.search_events(
            start_time=start_date,
            end_time=end_date
        )
        
        # Filter for financial data access
        financial_events = [
            e for e in events 
            if 'financial' in e.compliance_tags
        ]
        
        return {
            "report_type": "SOX",
            "period": f"{start_date.date()} to {end_date.date()}",
            "total_events": len(financial_events),
            "events_by_severity": self._count_by_severity(financial_events),
            "events_by_action": self._count_by_action(financial_events),
            "high_risk_events": [
                e for e in financial_events if e.risk_score >= 70
            ]
        }
    
    async def generate_gdpr_report(self, start_date: datetime,
                                 end_date: datetime) -> Dict[str, Any]:
        """Generate GDPR compliance report"""
        events = await self.audit_service.search_events(
            start_time=start_date,
            end_time=end_date
        )
        
        # Filter for personal data processing
        gdpr_events = [
            e for e in events 
            if 'gdpr' in e.compliance_tags or 'personal_data' in e.compliance_tags
        ]
        
        return {
            "report_type": "GDPR",
            "period": f"{start_date.date()} to {end_date.date()}",
            "data_processing_events": len(gdpr_events),
            "data_subjects_affected": self._count_unique_subjects(gdpr_events),
            "processing_purposes": self._extract_purposes(gdpr_events),
            "data_exports": [
                e for e in gdpr_events if e.action == ActionType.DATA_EXPORT
            ]
        }
    
    def _count_by_severity(self, events: List[AuditEvent]) -> Dict[str, int]:
        """Count events by severity level"""
        counts = {}
        for event in events:
            severity = event.severity.value
            counts[severity] = counts.get(severity, 0) + 1
        return counts
    
    def _count_by_action(self, events: List[AuditEvent]) -> Dict[str, int]:
        """Count events by action type"""
        counts = {}
        for event in events:
            action = event.action.value
            counts[action] = counts.get(action, 0) + 1
        return counts
    
    def _count_unique_subjects(self, events: List[AuditEvent]) -> int:
        """Count unique data subjects in GDPR events"""
        subjects = set()
        for event in events:
            subject_id = event.context.get('data_subject_id')
            if subject_id:
                subjects.add(subject_id)
        return len(subjects)
    
    def _extract_purposes(self, events: List[AuditEvent]) -> List[str]:
        """Extract processing purposes from GDPR events"""
        purposes = set()
        for event in events:
            purpose = event.context.get('processing_purpose')
            if purpose:
                purposes.add(purpose)
        return list(purposes)

# Usage Context Manager
@asynccontextmanager
async def audit_context(audit_service: AuditLoggingService, 
                       correlation_id: str):
    """Context manager for correlated audit events"""
    try:
        yield AuditContext(audit_service, correlation_id)
    finally:
        pass

class AuditContext:
    """Context for creating correlated audit events"""
    
    def __init__(self, audit_service: AuditLoggingService, 
                 correlation_id: str):
        self.audit_service = audit_service
        self.correlation_id = correlation_id
    
    async def log(self, event_type: EventType, severity: SeverityLevel,
                  action: ActionType, actor: Actor, resource: Resource,
                  outcome: str = "success", context: Dict[str, Any] = None,
                  compliance_tags: List[str] = None) -> str:
        """Log an audit event with correlation ID"""
        event = AuditEvent(
            event_type=event_type,
            severity=severity,
            action=action,
            actor=actor,
            resource=resource,
            outcome=outcome,
            context=context or {},
            correlation_id=self.correlation_id,
            compliance_tags=compliance_tags or []
        )
        
        return await self.audit_service.log_event(event)