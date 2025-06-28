# Chapter 9, Section 6.1: Input Validation and Security Controls

## Overview

AI applications face unique security challenges including prompt injection attacks, data leakage, and malicious input exploitation. This section implements comprehensive input validation, sanitization, and security controls to protect AI systems from various attack vectors while maintaining functionality.

## Input Validation Framework

### Core Security Validators

```python
# security/input_validation.py
import re
import html
import json
import hashlib
import time
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import structlog
from urllib.parse import urlparse
import base64

logger = structlog.get_logger()

class SecurityRiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ValidationResult(str, Enum):
    PASS = "pass"
    WARN = "warn"
    BLOCK = "block"

@dataclass
class SecurityFinding:
    """Security validation finding"""
    rule_id: str
    rule_name: str
    risk_level: SecurityRiskLevel
    description: str
    matched_content: Optional[str] = None
    suggested_action: str = ""
    confidence: float = 1.0
    
@dataclass
class ValidationReport:
    """Input validation report"""
    input_hash: str
    timestamp: float = field(default_factory=time.time)
    overall_result: ValidationResult = ValidationResult.PASS
    risk_score: float = 0.0
    findings: List[SecurityFinding] = field(default_factory=list)
    sanitized_input: Optional[str] = None
    processing_time: float = 0.0
    
    def add_finding(self, finding: SecurityFinding):
        """Add security finding and update overall assessment"""
        self.findings.append(finding)
        
        # Update risk score
        risk_weights = {
            SecurityRiskLevel.LOW: 1,
            SecurityRiskLevel.MEDIUM: 3,
            SecurityRiskLevel.HIGH: 7,
            SecurityRiskLevel.CRITICAL: 10
        }
        
        self.risk_score += risk_weights[finding.risk_level] * finding.confidence
        
        # Update overall result based on highest risk
        if finding.risk_level == SecurityRiskLevel.CRITICAL:
            self.overall_result = ValidationResult.BLOCK
        elif finding.risk_level == SecurityRiskLevel.HIGH and self.overall_result != ValidationResult.BLOCK:
            self.overall_result = ValidationResult.WARN
    
    def should_block(self) -> bool:
        """Determine if input should be blocked"""
        return self.overall_result == ValidationResult.BLOCK or self.risk_score >= 50

class AISecurityValidator:
    """Comprehensive AI input security validator"""
    
    def __init__(self, strict_mode: bool = False):
        self.strict_mode = strict_mode
        
        # Prompt injection patterns
        self.injection_patterns = [
            # Direct instruction injection
            (r'\b(ignore|forget|disregard)\s+(previous|prior|all|above)\s+(instructions?|prompts?|rules?)', 
             SecurityRiskLevel.HIGH, "Direct instruction override attempt"),
            
            # Role manipulation
            (r'\b(you are now|act as|pretend to be|roleplay as)\s+(?:a\s+)?(?:different|new|another)\s+\w+',
             SecurityRiskLevel.HIGH, "Role manipulation attempt"),
            
            # System prompt extraction
            (r'\b(show|tell|reveal|display)\s+(?:me\s+)?(?:your\s+)?(system\s+prompt|instructions|rules|guidelines)',
             SecurityRiskLevel.CRITICAL, "System prompt extraction attempt"),
            
            # Output manipulation
            (r'\b(respond\s+with|output|print|say)\s+(?:only\s+)?["\'][^"\']*["\']',
             SecurityRiskLevel.MEDIUM, "Output manipulation attempt"),
            
            # Jailbreak attempts
            (r'\b(jailbreak|bypass|circumvent|override)\s+(?:your\s+)?(safety|security|filters?|restrictions?)',
             SecurityRiskLevel.CRITICAL, "Jailbreak attempt detected"),
            
            # Developer mode activation
            (r'\b(developer\s+mode|admin\s+mode|debug\s+mode|maintenance\s+mode)',
             SecurityRiskLevel.HIGH, "Unauthorized mode activation attempt"),
            
            # Prompt continuation attacks
            (r'(\.\.\.|continue|and then|after that)\s+(?:ignore|bypass|override)',
             SecurityRiskLevel.MEDIUM, "Prompt continuation attack"),
            
            # Unicode/encoding attacks
            (r'[^\x00-\x7F]+.*(?:ignore|bypass|system)',
             SecurityRiskLevel.MEDIUM, "Potential encoding-based attack")
        ]
        
        # PII patterns
        self.pii_patterns = [
            # Social Security Numbers
            (r'\b\d{3}-?\d{2}-?\d{4}\b', SecurityRiskLevel.HIGH, "Social Security Number detected"),
            
            # Credit card numbers
            (r'\b(?:\d{4}[-\s]?){3}\d{4}\b', SecurityRiskLevel.CRITICAL, "Credit card number detected"),
            
            # Email addresses
            (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', 
             SecurityRiskLevel.MEDIUM, "Email address detected"),
            
            # Phone numbers
            (r'\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b',
             SecurityRiskLevel.MEDIUM, "Phone number detected"),
            
            # IP addresses
            (r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b', SecurityRiskLevel.LOW, "IP address detected"),
            
            # API keys/tokens (generic patterns)
            (r'\b[A-Za-z0-9]{32,}\b', SecurityRiskLevel.HIGH, "Potential API key or token detected"),
        ]
        
        # Malicious content patterns
        self.malicious_patterns = [
            # SQL injection attempts
            (r'\b(union|select|insert|update|delete|drop|exec|execute)\s+.*\s+(from|into|where)',
             SecurityRiskLevel.HIGH, "SQL injection attempt"),
            
            # Script injection
            (r'<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>',
             SecurityRiskLevel.HIGH, "Script injection attempt"),
            
            # Command injection
            (r'[;&|`]\s*(rm|del|format|shutdown|reboot|wget|curl)',
             SecurityRiskLevel.CRITICAL, "Command injection attempt"),
            
            # Path traversal
            (r'\.\.[\\/]', SecurityRiskLevel.MEDIUM, "Path traversal attempt"),
        ]
        
        # Suspicious URLs
        self.suspicious_url_patterns = [
            (r'(?:https?://)?(?:[\w\-]+\.)*(?:bit\.ly|tinyurl|shorturl)',
             SecurityRiskLevel.MEDIUM, "Suspicious shortened URL"),
            
            (r'(?:https?://)?[\d.]+(?::\d+)?/', 
             SecurityRiskLevel.LOW, "Direct IP address URL"),
        ]
        
        # Statistics
        self.validation_stats = {
            'total_validations': 0,
            'blocked_inputs': 0,
            'warnings_issued': 0,
            'findings_by_type': {},
            'avg_processing_time': 0.0
        }
    
    async def validate_input(
        self,
        user_input: str,
        context: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None
    ) -> ValidationReport:
        """Comprehensive input validation"""
        
        start_time = time.time()
        context = context or {}
        
        # Create input hash for tracking
        input_hash = hashlib.sha256(user_input.encode()).hexdigest()[:16]
        
        report = ValidationReport(input_hash=input_hash)
        
        try:
            # 1. Basic sanitization
            sanitized_input = await self._basic_sanitization(user_input)
            report.sanitized_input = sanitized_input
            
            # 2. Length validation
            await self._validate_length(user_input, report)
            
            # 3. Encoding validation
            await self._validate_encoding(user_input, report)
            
            # 4. Prompt injection detection
            await self._detect_prompt_injection(user_input, report)
            
            # 5. PII detection
            await self._detect_pii(user_input, report, context.get('allow_pii', False))
            
            # 6. Malicious content detection
            await self._detect_malicious_content(user_input, report)
            
            # 7. URL validation
            await self._validate_urls(user_input, report)
            
            # 8. Context-specific validation
            if context:
                await self._validate_context_specific(user_input, report, context)
            
            # 9. Rate limiting check
            if user_id:
                await self._check_rate_limits(user_id, report)
            
            # Update processing time
            report.processing_time = time.time() - start_time
            
            # Update statistics
            self._update_stats(report)
            
            logger.info(
                "Input validation completed",
                input_hash=input_hash,
                result=report.overall_result.value,
                risk_score=report.risk_score,
                findings_count=len(report.findings),
                processing_time=report.processing_time
            )
            
            return report
            
        except Exception as e:
            logger.error(
                "Input validation failed",
                input_hash=input_hash,
                error=str(e)
            )
            
            # Return critical finding for validation failure
            report.add_finding(SecurityFinding(
                rule_id="VALIDATION_ERROR",
                rule_name="Validation System Error",
                risk_level=SecurityRiskLevel.CRITICAL,
                description=f"Validation system error: {str(e)}",
                suggested_action="Block input and investigate"
            ))
            
            return report
    
    async def _basic_sanitization(self, user_input: str) -> str:
        """Basic input sanitization"""
        
        # HTML escape
        sanitized = html.escape(user_input)
        
        # Remove null bytes
        sanitized = sanitized.replace('\x00', '')
        
        # Normalize whitespace
        sanitized = re.sub(r'\s+', ' ', sanitized).strip()
        
        # Remove potential ANSI escape sequences
        sanitized = re.sub(r'\x1b\[[0-9;]*m', '', sanitized)
        
        return sanitized
    
    async def _validate_length(self, user_input: str, report: ValidationReport):
        """Validate input length"""
        
        max_length = 50000 if not self.strict_mode else 10000
        min_length = 1
        
        if len(user_input) > max_length:
            report.add_finding(SecurityFinding(
                rule_id="LENGTH_MAX",
                rule_name="Maximum Length Exceeded",
                risk_level=SecurityRiskLevel.MEDIUM,
                description=f"Input length ({len(user_input)}) exceeds maximum ({max_length})",
                suggested_action="Truncate input or reject"
            ))
        
        if len(user_input) < min_length:
            report.add_finding(SecurityFinding(
                rule_id="LENGTH_MIN",
                rule_name="Minimum Length Not Met",
                risk_level=SecurityRiskLevel.LOW,
                description=f"Input length ({len(user_input)}) below minimum ({min_length})",
                suggested_action="Request longer input"
            ))
    
    async def _validate_encoding(self, user_input: str, report: ValidationReport):
        """Validate character encoding and detect anomalies"""
        
        try:
            # Check for valid UTF-8
            user_input.encode('utf-8')
        except UnicodeEncodeError as e:
            report.add_finding(SecurityFinding(
                rule_id="ENCODING_INVALID",
                rule_name="Invalid Character Encoding",
                risk_level=SecurityRiskLevel.MEDIUM,
                description=f"Invalid UTF-8 encoding: {str(e)}",
                suggested_action="Reject input with invalid encoding"
            ))
        
        # Check for suspicious character patterns
        non_printable_count = sum(1 for c in user_input if ord(c) < 32 and c not in '\t\n\r')
        if non_printable_count > 5:
            report.add_finding(SecurityFinding(
                rule_id="ENCODING_SUSPICIOUS",
                rule_name="Suspicious Character Pattern",
                risk_level=SecurityRiskLevel.MEDIUM,
                description=f"High number of non-printable characters: {non_printable_count}",
                suggested_action="Review input for potential encoding attack"
            ))
        
        # Check for homograph attacks (mixed scripts)
        scripts = set()
        for char in user_input:
            if char.isalpha():
                script = self._get_unicode_script(char)
                scripts.add(script)
        
        if len(scripts) > 3:  # Mixed scripts might indicate homograph attack
            report.add_finding(SecurityFinding(
                rule_id="ENCODING_MIXED_SCRIPTS",
                rule_name="Mixed Unicode Scripts",
                risk_level=SecurityRiskLevel.LOW,
                description=f"Multiple Unicode scripts detected: {len(scripts)}",
                suggested_action="Review for potential homograph attack"
            ))
    
    async def _detect_prompt_injection(self, user_input: str, report: ValidationReport):
        """Detect prompt injection attempts"""
        
        input_lower = user_input.lower()
        
        for pattern, risk_level, description in self.injection_patterns:
            matches = re.finditer(pattern, input_lower, re.IGNORECASE)
            
            for match in matches:
                confidence = self._calculate_injection_confidence(match, user_input)
                
                report.add_finding(SecurityFinding(
                    rule_id=f"INJECTION_{pattern[:20]}",
                    rule_name="Prompt Injection Attempt",
                    risk_level=risk_level,
                    description=description,
                    matched_content=match.group()[:100],
                    suggested_action="Block or sanitize input",
                    confidence=confidence
                ))
        
        # Advanced injection detection using heuristics
        await self._detect_advanced_injection(user_input, report)
    
    async def _detect_advanced_injection(self, user_input: str, report: ValidationReport):
        """Advanced prompt injection detection using heuristics"""
        
        # Check for instruction-like patterns
        instruction_indicators = [
            'you must', 'you should', 'you will', 'you need to',
            'do not', 'don\'t', 'never', 'always',
            'from now on', 'starting now', 'beginning now'
        ]
        
        instruction_count = sum(1 for indicator in instruction_indicators 
                              if indicator in user_input.lower())
        
        if instruction_count >= 3:
            report.add_finding(SecurityFinding(
                rule_id="INJECTION_HEURISTIC_INSTRUCTION",
                rule_name="High Instruction Density",
                risk_level=SecurityRiskLevel.MEDIUM,
                description=f"High density of instruction-like patterns: {instruction_count}",
                suggested_action="Review for potential instruction injection",
                confidence=min(1.0, instruction_count / 5)
            ))
        
        # Check for meta-conversation attempts
        meta_patterns = [
            'this conversation', 'our chat', 'this prompt', 'my message',
            'system message', 'ai model', 'language model'
        ]
        
        meta_count = sum(1 for pattern in meta_patterns 
                        if pattern in user_input.lower())
        
        if meta_count >= 2:
            report.add_finding(SecurityFinding(
                rule_id="INJECTION_HEURISTIC_META",
                rule_name="Meta-conversation Pattern",
                risk_level=SecurityRiskLevel.MEDIUM,
                description=f"Meta-conversation patterns detected: {meta_count}",
                suggested_action="Review for potential system manipulation",
                confidence=min(1.0, meta_count / 3)
            ))
    
    def _calculate_injection_confidence(self, match, full_input: str) -> float:
        """Calculate confidence score for injection attempt"""
        
        base_confidence = 0.7
        
        # Increase confidence if match is at beginning
        if match.start() < len(full_input) * 0.1:
            base_confidence += 0.2
        
        # Increase confidence if followed by specific keywords
        following_text = full_input[match.end():match.end()+100].lower()
        high_risk_keywords = ['system', 'admin', 'root', 'override', 'bypass']
        
        for keyword in high_risk_keywords:
            if keyword in following_text:
                base_confidence += 0.1
        
        return min(1.0, base_confidence)
    
    async def _detect_pii(self, user_input: str, report: ValidationReport, allow_pii: bool = False):
        """Detect personally identifiable information"""
        
        if allow_pii:
            return  # Skip PII detection if explicitly allowed
        
        for pattern, risk_level, description in self.pii_patterns:
            matches = re.finditer(pattern, user_input, re.IGNORECASE)
            
            for match in matches:
                # Additional validation for potential false positives
                if self._validate_pii_match(match.group(), description):
                    report.add_finding(SecurityFinding(
                        rule_id=f"PII_{description.split()[0].upper()}",
                        rule_name="PII Detection",
                        risk_level=risk_level,
                        description=description,
                        matched_content="[REDACTED]",  # Don't log actual PII
                        suggested_action="Remove or mask PII before processing"
                    ))
    
    def _validate_pii_match(self, match_text: str, description: str) -> bool:
        """Validate PII matches to reduce false positives"""
        
        if "Social Security" in description:
            # Validate SSN format and checksums
            digits = re.sub(r'\D', '', match_text)
            if len(digits) == 9:
                # Basic SSN validation (simplified)
                return not (digits.startswith('000') or digits[3:5] == '00' or digits[5:] == '0000')
        
        elif "Credit card" in description:
            # Luhn algorithm validation
            digits = re.sub(r'\D', '', match_text)
            return self._luhn_check(digits) if len(digits) >= 13 else False
        
        elif "Email" in description:
            # Basic email validation
            return '@' in match_text and '.' in match_text.split('@')[-1]
        
        elif "API key" in description:
            # Reduce false positives for generic patterns
            return len(match_text) >= 32 and any(c.isdigit() for c in match_text) and any(c.isalpha() for c in match_text)
        
        return True  # Default to true for other patterns
    
    def _luhn_check(self, card_number: str) -> bool:
        """Luhn algorithm for credit card validation"""
        
        def luhn_checksum(card_num):
            def digits_of(n):
                return [int(d) for d in str(n)]
            
            digits = digits_of(card_num)
            odd_digits = digits[-1::-2]
            even_digits = digits[-2::-2]
            checksum = sum(odd_digits)
            for d in even_digits:
                checksum += sum(digits_of(d*2))
            return checksum % 10
        
        return luhn_checksum(card_number) == 0
    
    async def _detect_malicious_content(self, user_input: str, report: ValidationReport):
        """Detect malicious content patterns"""
        
        for pattern, risk_level, description in self.malicious_patterns:
            matches = re.finditer(pattern, user_input, re.IGNORECASE)
            
            for match in matches:
                report.add_finding(SecurityFinding(
                    rule_id=f"MALICIOUS_{description.split()[0].upper()}",
                    rule_name="Malicious Content",
                    risk_level=risk_level,
                    description=description,
                    matched_content=match.group()[:50],
                    suggested_action="Block input immediately"
                ))
    
    async def _validate_urls(self, user_input: str, report: ValidationReport):
        """Validate and check URLs for security issues"""
        
        # Extract URLs
        url_pattern = r'https?://[^\s<>"\'|\\^`{}\[\]]+'
        urls = re.findall(url_pattern, user_input, re.IGNORECASE)
        
        for url in urls:
            await self._analyze_url(url, report)
    
    async def _analyze_url(self, url: str, report: ValidationReport):
        """Analyze individual URL for security issues"""
        
        try:
            parsed = urlparse(url)
            
            # Check against suspicious patterns
            for pattern, risk_level, description in self.suspicious_url_patterns:
                if re.match(pattern, url, re.IGNORECASE):
                    report.add_finding(SecurityFinding(
                        rule_id="URL_SUSPICIOUS",
                        rule_name="Suspicious URL",
                        risk_level=risk_level,
                        description=description,
                        matched_content=url[:100],
                        suggested_action="Review URL before processing"
                    ))
            
            # Check for localhost/internal addresses
            if parsed.hostname in ['localhost', '127.0.0.1', '0.0.0.0'] or \
               (parsed.hostname and parsed.hostname.startswith('192.168.')):
                report.add_finding(SecurityFinding(
                    rule_id="URL_INTERNAL",
                    rule_name="Internal/Localhost URL",
                    risk_level=SecurityRiskLevel.MEDIUM,
                    description="URL points to internal/localhost address",
                    matched_content=url[:100],
                    suggested_action="Block access to internal resources"
                ))
            
        except Exception as e:
            report.add_finding(SecurityFinding(
                rule_id="URL_MALFORMED",
                rule_name="Malformed URL",
                risk_level=SecurityRiskLevel.LOW,
                description=f"Malformed URL detected: {str(e)}",
                matched_content=url[:100],
                suggested_action="Validate URL format"
            ))
    
    async def _validate_context_specific(
        self, 
        user_input: str, 
        report: ValidationReport, 
        context: Dict[str, Any]
    ):
        """Context-specific validation rules"""
        
        context_type = context.get('type', 'general')
        
        if context_type == 'code_review':
            # Additional validation for code review contexts
            await self._validate_code_content(user_input, report)
        
        elif context_type == 'financial':
            # Enhanced financial data validation
            await self._validate_financial_content(user_input, report)
        
        elif context_type == 'medical':
            # Medical content validation
            await self._validate_medical_content(user_input, report)
    
    async def _validate_code_content(self, user_input: str, report: ValidationReport):
        """Validate code content for security issues"""
        
        # Check for potentially dangerous functions
        dangerous_functions = [
            'eval', 'exec', 'compile', '__import__',
            'subprocess', 'os.system', 'shell_exec'
        ]
        
        for func in dangerous_functions:
            if func in user_input:
                report.add_finding(SecurityFinding(
                    rule_id="CODE_DANGEROUS_FUNCTION",
                    rule_name="Dangerous Function Usage",
                    risk_level=SecurityRiskLevel.HIGH,
                    description=f"Potentially dangerous function detected: {func}",
                    suggested_action="Review code for security implications"
                ))
    
    async def _validate_financial_content(self, user_input: str, report: ValidationReport):
        """Enhanced validation for financial contexts"""
        
        # More strict PII detection for financial context
        await self._detect_pii(user_input, report, allow_pii=False)
        
        # Check for financial keywords that might indicate sensitive data
        financial_keywords = ['account number', 'routing number', 'balance', 'transaction']
        
        for keyword in financial_keywords:
            if keyword in user_input.lower():
                report.add_finding(SecurityFinding(
                    rule_id="FINANCIAL_SENSITIVE",
                    rule_name="Financial Sensitive Content",
                    risk_level=SecurityRiskLevel.HIGH,
                    description=f"Financial sensitive keyword detected: {keyword}",
                    suggested_action="Enhanced security review required"
                ))
    
    async def _validate_medical_content(self, user_input: str, report: ValidationReport):
        """Validation for medical/healthcare contexts"""
        
        # HIPAA-related sensitive information patterns
        medical_patterns = [
            (r'\b(patient|medical)\s+(?:id|number|record)', SecurityRiskLevel.CRITICAL, "Medical ID detected"),
            (r'\b(diagnosis|condition|medication|treatment)', SecurityRiskLevel.MEDIUM, "Medical information detected"),
        ]
        
        for pattern, risk_level, description in medical_patterns:
            if re.search(pattern, user_input, re.IGNORECASE):
                report.add_finding(SecurityFinding(
                    rule_id="MEDICAL_PHI",
                    rule_name="Protected Health Information",
                    risk_level=risk_level,
                    description=description,
                    suggested_action="HIPAA compliance review required"
                ))
    
    async def _check_rate_limits(self, user_id: str, report: ValidationReport):
        """Check user-specific rate limits"""
        
        # This would integrate with actual rate limiting service
        # For now, we'll add a placeholder finding for high-risk users
        
        # Check if user is on watchlist (simplified)
        if user_id in ['suspicious_user_1', 'test_attacker']:
            report.add_finding(SecurityFinding(
                rule_id="RATE_LIMIT_WATCHLIST",
                rule_name="Watchlist User",
                risk_level=SecurityRiskLevel.HIGH,
                description="User is on security watchlist",
                suggested_action="Enhanced monitoring required"
            ))
    
    def _get_unicode_script(self, char: str) -> str:
        """Get Unicode script for character (simplified)"""
        
        code_point = ord(char)
        
        if 0x0000 <= code_point <= 0x007F:
            return "Latin"
        elif 0x0400 <= code_point <= 0x04FF:
            return "Cyrillic"
        elif 0x0590 <= code_point <= 0x05FF:
            return "Hebrew"
        elif 0x0600 <= code_point <= 0x06FF:
            return "Arabic"
        elif 0x4E00 <= code_point <= 0x9FFF:
            return "CJK"
        else:
            return "Other"
    
    def _update_stats(self, report: ValidationReport):
        """Update validation statistics"""
        
        self.validation_stats['total_validations'] += 1
        
        if report.should_block():
            self.validation_stats['blocked_inputs'] += 1
        elif report.overall_result == ValidationResult.WARN:
            self.validation_stats['warnings_issued'] += 1
        
        # Update findings by type
        for finding in report.findings:
            rule_type = finding.rule_id.split('_')[0]
            self.validation_stats['findings_by_type'][rule_type] = \
                self.validation_stats['findings_by_type'].get(rule_type, 0) + 1
        
        # Update average processing time
        current_avg = self.validation_stats['avg_processing_time']
        total_validations = self.validation_stats['total_validations']
        
        self.validation_stats['avg_processing_time'] = (
            (current_avg * (total_validations - 1) + report.processing_time) / total_validations
        )
    
    def get_validation_stats(self) -> Dict[str, Any]:
        """Get validation statistics"""
        
        total = self.validation_stats['total_validations']
        
        return {
            'total_validations': total,
            'blocked_inputs': self.validation_stats['blocked_inputs'],
            'warnings_issued': self.validation_stats['warnings_issued'],
            'pass_rate': ((total - self.validation_stats['blocked_inputs']) / max(1, total)) * 100,
            'avg_processing_time': self.validation_stats['avg_processing_time'],
            'findings_by_type': dict(self.validation_stats['findings_by_type']),
            'security_metrics': {
                'injection_attempts_blocked': self.validation_stats['findings_by_type'].get('INJECTION', 0),
                'pii_instances_found': self.validation_stats['findings_by_type'].get('PII', 0),
                'malicious_content_blocked': self.validation_stats['findings_by_type'].get('MALICIOUS', 0)
            }
        }
```

## Content Sanitization Engine

### Advanced Sanitization System

```python
# security/content_sanitizer.py
import re
import html
import json
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
import structlog
from urllib.parse import quote, unquote

logger = structlog.get_logger()

@dataclass
class SanitizationRule:
    """Content sanitization rule"""
    rule_id: str
    name: str
    pattern: str
    replacement: str
    description: str
    enabled: bool = True

class ContentSanitizer:
    """Advanced content sanitization for AI inputs"""
    
    def __init__(self, aggressive_mode: bool = False):
        self.aggressive_mode = aggressive_mode
        
        # Default sanitization rules
        self.sanitization_rules = [
            # HTML/Script sanitization
            SanitizationRule(
                rule_id="HTML_TAGS",
                name="HTML Tag Removal",
                pattern=r'<[^>]*>',
                replacement='',
                description="Remove HTML tags"
            ),
            
            # JavaScript event handlers
            SanitizationRule(
                rule_id="JS_EVENTS",
                name="JavaScript Event Handler Removal",
                pattern=r'on\w+\s*=\s*["\'][^"\']*["\']',
                replacement='',
                description="Remove JavaScript event handlers"
            ),
            
            # SQL injection keywords
            SanitizationRule(
                rule_id="SQL_INJECTION",
                name="SQL Injection Keyword Sanitization",
                pattern=r'\b(union|select|insert|update|delete|drop|exec|execute)\s+(all\s+)?(select|from|where|into)',
                replacement='[SQL_REMOVED]',
                description="Sanitize SQL injection attempts"
            ),
            
            # Command injection
            SanitizationRule(
                rule_id="CMD_INJECTION",
                name="Command Injection Sanitization",
                pattern=r'[;&|`]\s*(rm|del|format|shutdown|reboot|wget|curl|nc|netcat)',
                replacement='[CMD_REMOVED]',
                description="Sanitize command injection attempts"
            ),
            
            # Prompt injection markers
            SanitizationRule(
                rule_id="PROMPT_INJECTION",
                name="Prompt Injection Marker Removal",
                pattern=r'\b(ignore|forget|disregard)\s+(previous|prior|all|above)\s+(instructions?|prompts?|rules?)',
                replacement='[INSTRUCTION_REMOVED]',
                description="Remove prompt injection attempts"
            ),
            
            # System prompt extraction
            SanitizationRule(
                rule_id="SYSTEM_PROMPT_EXTRACTION",
                name="System Prompt Extraction Removal",
                pattern=r'\b(show|tell|reveal|display)\s+(?:me\s+)?(?:your\s+)?(system\s+prompt|instructions|rules|guidelines)',
                replacement='[SYSTEM_QUERY_REMOVED]',
                description="Remove system prompt extraction attempts"
            ),
            
            # Excessive whitespace
            SanitizationRule(
                rule_id="WHITESPACE",
                name="Whitespace Normalization",
                pattern=r'\s+',
                replacement=' ',
                description="Normalize whitespace"
            ),
            
            # Non-printable characters
            SanitizationRule(
                rule_id="NON_PRINTABLE",
                name="Non-printable Character Removal",
                pattern=r'[^\x20-\x7E\n\r\t]',
                replacement='',
                description="Remove non-printable characters"
            ),
            
            # URL encoding attacks
            SanitizationRule(
                rule_id="URL_ENCODING",
                name="URL Encoding Normalization",
                pattern=r'%[0-9A-Fa-f]{2}',
                replacement=lambda m: unquote(m.group()),
                description="Decode URL encoding"
            ),
        ]
        
        # PII masking rules
        self.pii_masking_rules = [
            SanitizationRule(
                rule_id="MASK_SSN",
                name="Social Security Number Masking",
                pattern=r'\b\d{3}-?\d{2}-?\d{4}\b',
                replacement='XXX-XX-XXXX',
                description="Mask Social Security Numbers"
            ),
            
            SanitizationRule(
                rule_id="MASK_CREDIT_CARD",
                name="Credit Card Masking",
                pattern=r'\b(?:\d{4}[-\s]?){3}\d{4}\b',
                replacement='XXXX-XXXX-XXXX-XXXX',
                description="Mask credit card numbers"
            ),
            
            SanitizationRule(
                rule_id="MASK_EMAIL",
                name="Email Masking",
                pattern=r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
                replacement='[EMAIL_REDACTED]',
                description="Mask email addresses"
            ),
            
            SanitizationRule(
                rule_id="MASK_PHONE",
                name="Phone Number Masking",
                pattern=r'\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b',
                replacement='XXX-XXX-XXXX',
                description="Mask phone numbers"
            ),
        ]
        
        # Statistics
        self.sanitization_stats = {
            'total_sanitizations': 0,
            'rules_applied': {},
            'avg_processing_time': 0.0,
            'content_reduction_ratio': 0.0
        }
    
    async def sanitize_content(
        self,
        content: str,
        apply_pii_masking: bool = True,
        custom_rules: Optional[List[SanitizationRule]] = None,
        preserve_meaning: bool = True
    ) -> Tuple[str, Dict[str, Any]]:
        """Comprehensive content sanitization"""
        
        import time
        start_time = time.time()
        
        original_length = len(content)
        sanitized_content = content
        applied_rules = []
        
        try:
            # Apply core sanitization rules
            for rule in self.sanitization_rules:
                if rule.enabled:
                    before_length = len(sanitized_content)
                    sanitized_content = await self._apply_rule(sanitized_content, rule)
                    after_length = len(sanitized_content)
                    
                    if before_length != after_length:
                        applied_rules.append({
                            'rule_id': rule.rule_id,
                            'rule_name': rule.name,
                            'characters_affected': before_length - after_length
                        })
            
            # Apply PII masking if requested
            if apply_pii_masking:
                for rule in self.pii_masking_rules:
                    if rule.enabled:
                        before_length = len(sanitized_content)
                        sanitized_content = await self._apply_rule(sanitized_content, rule)
                        after_length = len(sanitized_content)
                        
                        if before_length != after_length:
                            applied_rules.append({
                                'rule_id': rule.rule_id,
                                'rule_name': rule.name,
                                'characters_affected': abs(before_length - after_length)
                            })
            
            # Apply custom rules if provided
            if custom_rules:
                for rule in custom_rules:
                    if rule.enabled:
                        before_length = len(sanitized_content)
                        sanitized_content = await self._apply_rule(sanitized_content, rule)
                        after_length = len(sanitized_content)
                        
                        if before_length != after_length:
                            applied_rules.append({
                                'rule_id': rule.rule_id,
                                'rule_name': rule.name,
                                'characters_affected': abs(before_length - after_length)
                            })
            
            # Final cleanup
            sanitized_content = sanitized_content.strip()
            
            # Update statistics
            processing_time = time.time() - start_time
            self._update_sanitization_stats(applied_rules, processing_time, original_length, len(sanitized_content))
            
            # Prepare summary
            summary = {
                'original_length': original_length,
                'sanitized_length': len(sanitized_content),
                'rules_applied': applied_rules,
                'processing_time': processing_time,
                'content_reduction_percent': ((original_length - len(sanitized_content)) / max(1, original_length)) * 100
            }
            
            logger.info(
                "Content sanitization completed",
                original_length=original_length,
                sanitized_length=len(sanitized_content),
                rules_applied_count=len(applied_rules),
                processing_time=processing_time
            )
            
            return sanitized_content, summary
            
        except Exception as e:
            logger.error(
                "Content sanitization failed",
                error=str(e),
                original_length=original_length
            )
            # Return original content if sanitization fails
            return content, {'error': str(e)}
    
    async def _apply_rule(self, content: str, rule: SanitizationRule) -> str:
        """Apply individual sanitization rule"""
        
        try:
            if callable(rule.replacement):
                # Handle lambda replacements
                sanitized = re.sub(rule.pattern, rule.replacement, content, flags=re.IGNORECASE)
            else:
                # Handle string replacements
                sanitized = re.sub(rule.pattern, rule.replacement, content, flags=re.IGNORECASE)
            
            return sanitized
            
        except Exception as e:
            logger.warning(
                "Failed to apply sanitization rule",
                rule_id=rule.rule_id,
                error=str(e)
            )
            return content
    
    def _update_sanitization_stats(
        self, 
        applied_rules: List[Dict[str, Any]], 
        processing_time: float,
        original_length: int,
        sanitized_length: int
    ):
        """Update sanitization statistics"""
        
        self.sanitization_stats['total_sanitizations'] += 1
        
        # Update rule application counts
        for rule_info in applied_rules:
            rule_id = rule_info['rule_id']
            self.sanitization_stats['rules_applied'][rule_id] = \
                self.sanitization_stats['rules_applied'].get(rule_id, 0) + 1
        
        # Update average processing time
        current_avg = self.sanitization_stats['avg_processing_time']
        total_sanitizations = self.sanitization_stats['total_sanitizations']
        
        self.sanitization_stats['avg_processing_time'] = (
            (current_avg * (total_sanitizations - 1) + processing_time) / total_sanitizations
        )
        
        # Update content reduction ratio
        if original_length > 0:
            reduction_ratio = (original_length - sanitized_length) / original_length
            current_reduction = self.sanitization_stats['content_reduction_ratio']
            
            self.sanitization_stats['content_reduction_ratio'] = (
                (current_reduction * (total_sanitizations - 1) + reduction_ratio) / total_sanitizations
            )
    
    async def create_custom_rule(
        self,
        rule_id: str,
        name: str,
        pattern: str,
        replacement: str,
        description: str
    ) -> SanitizationRule:
        """Create custom sanitization rule"""
        
        # Validate pattern
        try:
            re.compile(pattern)
        except re.error as e:
            raise ValueError(f"Invalid regex pattern: {e}")
        
        rule = SanitizationRule(
            rule_id=rule_id,
            name=name,
            pattern=pattern,
            replacement=replacement,
            description=description
        )
        
        logger.info(
            "Custom sanitization rule created",
            rule_id=rule_id,
            name=name
        )
        
        return rule
    
    def get_sanitization_stats(self) -> Dict[str, Any]:
        """Get sanitization statistics"""
        
        return {
            'total_sanitizations': self.sanitization_stats['total_sanitizations'],
            'avg_processing_time': self.sanitization_stats['avg_processing_time'],
            'content_reduction_ratio': self.sanitization_stats['content_reduction_ratio'],
            'rules_applied': dict(self.sanitization_stats['rules_applied']),
            'most_used_rules': sorted(
                self.sanitization_stats['rules_applied'].items(),
                key=lambda x: x[1],
                reverse=True
            )[:5],
            'total_rules_available': len(self.sanitization_rules) + len(self.pii_masking_rules)
        }
    
    async def test_sanitization_rule(
        self,
        rule: SanitizationRule,
        test_content: str
    ) -> Dict[str, Any]:
        """Test sanitization rule against content"""
        
        original_content = test_content
        
        try:
            sanitized_content = await self._apply_rule(test_content, rule)
            
            matches = re.findall(rule.pattern, original_content, re.IGNORECASE)
            
            return {
                'rule_id': rule.rule_id,
                'original_content': original_content,
                'sanitized_content': sanitized_content,
                'matches_found': len(matches),
                'matches': matches[:10],  # First 10 matches
                'content_changed': original_content != sanitized_content,
                'test_successful': True
            }
            
        except Exception as e:
            return {
                'rule_id': rule.rule_id,
                'test_successful': False,
                'error': str(e)
            }
    
    async def batch_sanitize(
        self,
        content_list: List[str],
        apply_pii_masking: bool = True,
        max_concurrent: int = 10
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """Batch sanitization for multiple content items"""
        
        import asyncio
        
        # Create semaphore to limit concurrent processing
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def sanitize_single(content: str) -> Tuple[str, Dict[str, Any]]:
            async with semaphore:
                return await self.sanitize_content(content, apply_pii_masking)
        
        # Process all content concurrently
        tasks = [sanitize_single(content) for content in content_list]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append((
                    content_list[i],  # Return original content on error
                    {'error': str(result)}
                ))
            else:
                processed_results.append(result)
        
        logger.info(
            "Batch sanitization completed",
            total_items=len(content_list),
            successful=len([r for r in processed_results if 'error' not in r[1]]),
            failed=len([r for r in processed_results if 'error' in r[1]])
        )
        
        return processed_results
```

## Security Configuration Management

### Security Policy Engine

```python
# security/security_policy.py
import json
import time
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field
from enum import Enum
import structlog

logger = structlog.get_logger()

class PolicyScope(str, Enum):
    GLOBAL = "global"
    USER = "user"
    CONTEXT = "context"
    ENDPOINT = "endpoint"

class PolicyAction(str, Enum):
    ALLOW = "allow"
    WARN = "warn"
    BLOCK = "block"
    LOG_ONLY = "log_only"

@dataclass
class SecurityPolicy:
    """Security policy definition"""
    policy_id: str
    name: str
    description: str
    scope: PolicyScope
    action: PolicyAction
    conditions: Dict[str, Any] = field(default_factory=dict)
    exceptions: List[str] = field(default_factory=list)
    enabled: bool = True
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    version: str = "1.0"
    
    def matches_conditions(self, context: Dict[str, Any]) -> bool:
        """Check if policy conditions match current context"""
        
        for condition_key, condition_value in self.conditions.items():
            context_value = context.get(condition_key)
            
            if isinstance(condition_value, dict):
                # Handle complex conditions
                operator = condition_value.get('operator', 'equals')
                value = condition_value.get('value')
                
                if operator == 'equals' and context_value != value:
                    return False
                elif operator == 'contains' and value not in str(context_value):
                    return False
                elif operator == 'greater_than' and (context_value is None or context_value <= value):
                    return False
                elif operator == 'less_than' and (context_value is None or context_value >= value):
                    return False
                elif operator == 'in' and context_value not in value:
                    return False
            else:
                # Simple equality check
                if context_value != condition_value:
                    return False
        
        return True

class SecurityPolicyEngine:
    """Security policy management and enforcement engine"""
    
    def __init__(self):
        self.policies: Dict[str, SecurityPolicy] = {}
        self.policy_history: List[Dict[str, Any]] = []
        
        # Default security policies
        self._initialize_default_policies()
        
        # Policy enforcement statistics
        self.enforcement_stats = {
            'total_evaluations': 0,
            'policy_hits': {},
            'actions_taken': {action.value: 0 for action in PolicyAction},
            'avg_evaluation_time': 0.0
        }
    
    def _initialize_default_policies(self):
        """Initialize default security policies"""
        
        default_policies = [
            SecurityPolicy(
                policy_id="PROMPT_INJECTION_BLOCK",
                name="Block Prompt Injection Attempts",
                description="Block inputs with high prompt injection risk",
                scope=PolicyScope.GLOBAL,
                action=PolicyAction.BLOCK,
                conditions={
                    'risk_score': {'operator': 'greater_than', 'value': 30},
                    'injection_findings': {'operator': 'greater_than', 'value': 0}
                }
            ),
            
            SecurityPolicy(
                policy_id="PII_WARNING",
                name="Warn on PII Detection",
                description="Issue warning when PII is detected in input",
                scope=PolicyScope.GLOBAL,
                action=PolicyAction.WARN,
                conditions={
                    'pii_findings': {'operator': 'greater_than', 'value': 0}
                }
            ),
            
            SecurityPolicy(
                policy_id="MALICIOUS_CONTENT_BLOCK",
                name="Block Malicious Content",
                description="Block inputs containing malicious content patterns",
                scope=PolicyScope.GLOBAL,
                action=PolicyAction.BLOCK,
                conditions={
                    'malicious_findings': {'operator': 'greater_than', 'value': 0}
                }
            ),
            
            SecurityPolicy(
                policy_id="FINANCIAL_STRICT",
                name="Strict Financial Context Policy",
                description="Enhanced security for financial contexts",
                scope=PolicyScope.CONTEXT,
                action=PolicyAction.BLOCK,
                conditions={
                    'context_type': 'financial',
                    'risk_score': {'operator': 'greater_than', 'value': 10}
                }
            ),
            
            SecurityPolicy(
                policy_id="HIGH_VOLUME_USER_LIMIT",
                name="High Volume User Rate Limiting",
                description="Enhanced monitoring for high-volume users",
                scope=PolicyScope.USER,
                action=PolicyAction.LOG_ONLY,
                conditions={
                    'daily_request_count': {'operator': 'greater_than', 'value': 1000}
                }
            ),
        ]
        
        for policy in default_policies:
            self.policies[policy.policy_id] = policy
        
        logger.info(
            "Default security policies initialized",
            policy_count=len(default_policies)
        )
    
    async def evaluate_policies(
        self,
        validation_report: 'ValidationReport',
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Evaluate all applicable security policies"""
        
        start_time = time.time()
        
        policy_results = {
            'applicable_policies': [],
            'triggered_policies': [],
            'final_action': PolicyAction.ALLOW,
            'risk_adjustments': {},
            'enforcement_details': []
        }
        
        # Build evaluation context
        eval_context = {
            'risk_score': validation_report.risk_score,
            'injection_findings': len([f for f in validation_report.findings if 'INJECTION' in f.rule_id]),
            'pii_findings': len([f for f in validation_report.findings if 'PII' in f.rule_id]),
            'malicious_findings': len([f for f in validation_report.findings if 'MALICIOUS' in f.rule_id]),
            **context
        }
        
        # Evaluate each policy
        for policy_id, policy in self.policies.items():
            if not policy.enabled:
                continue
            
            policy_results['applicable_policies'].append(policy_id)
            
            # Check if policy conditions match
            if policy.matches_conditions(eval_context):
                policy_results['triggered_policies'].append(policy_id)
                
                # Record policy hit
                self.enforcement_stats['policy_hits'][policy_id] = \
                    self.enforcement_stats['policy_hits'].get(policy_id, 0) + 1
                
                # Apply policy action
                enforcement_detail = {
                    'policy_id': policy_id,
                    'policy_name': policy.name,
                    'action': policy.action,
                    'conditions_matched': policy.conditions,
                    'timestamp': time.time()
                }
                
                policy_results['enforcement_details'].append(enforcement_detail)
                
                # Determine final action (most restrictive wins)
                if policy.action == PolicyAction.BLOCK:
                    policy_results['final_action'] = PolicyAction.BLOCK
                elif policy.action == PolicyAction.WARN and policy_results['final_action'] != PolicyAction.BLOCK:
                    policy_results['final_action'] = PolicyAction.WARN
                
                # Apply risk adjustments
                if policy.scope == PolicyScope.CONTEXT and context.get('context_type') == 'financial':
                    policy_results['risk_adjustments']['financial_context'] = 1.5
                
                # Update action statistics
                self.enforcement_stats['actions_taken'][policy.action.value] += 1
        
        # Update evaluation statistics
        evaluation_time = time.time() - start_time
        self.enforcement_stats['total_evaluations'] += 1
        
        current_avg = self.enforcement_stats['avg_evaluation_time']
        total_evals = self.enforcement_stats['total_evaluations']
        self.enforcement_stats['avg_evaluation_time'] = (
            (current_avg * (total_evals - 1) + evaluation_time) / total_evals
        )
        
        logger.info(
            "Security policy evaluation completed",
            applicable_policies=len(policy_results['applicable_policies']),
            triggered_policies=len(policy_results['triggered_policies']),
            final_action=policy_results['final_action'].value,
            evaluation_time=evaluation_time
        )
        
        return policy_results
    
    def create_policy(
        self,
        policy_id: str,
        name: str,
        description: str,
        scope: PolicyScope,
        action: PolicyAction,
        conditions: Dict[str, Any],
        exceptions: Optional[List[str]] = None
    ) -> SecurityPolicy:
        """Create new security policy"""
        
        if policy_id in self.policies:
            raise ValueError(f"Policy with ID {policy_id} already exists")
        
        policy = SecurityPolicy(
            policy_id=policy_id,
            name=name,
            description=description,
            scope=scope,
            action=action,
            conditions=conditions,
            exceptions=exceptions or []
        )
        
        self.policies[policy_id] = policy
        
        # Record policy creation
        self.policy_history.append({
            'action': 'created',
            'policy_id': policy_id,
            'timestamp': time.time(),
            'policy_data': policy.__dict__.copy()
        })
        
        logger.info(
            "Security policy created",
            policy_id=policy_id,
            name=name,
            scope=scope.value,
            action=action.value
        )
        
        return policy
    
    def update_policy(
        self,
        policy_id: str,
        updates: Dict[str, Any]
    ) -> bool:
        """Update existing security policy"""
        
        if policy_id not in self.policies:
            return False
        
        policy = self.policies[policy_id]
        old_policy_data = policy.__dict__.copy()
        
        # Apply updates
        for key, value in updates.items():
            if hasattr(policy, key):
                setattr(policy, key, value)
        
        policy.updated_at = time.time()
        
        # Record policy update
        self.policy_history.append({
            'action': 'updated',
            'policy_id': policy_id,
            'timestamp': time.time(),
            'old_data': old_policy_data,
            'new_data': policy.__dict__.copy(),
            'changes': updates
        })
        
        logger.info(
            "Security policy updated",
            policy_id=policy_id,
            changes=list(updates.keys())
        )
        
        return True
    
    def delete_policy(self, policy_id: str) -> bool:
        """Delete security policy"""
        
        if policy_id not in self.policies:
            return False
        
        policy_data = self.policies[policy_id].__dict__.copy()
        del self.policies[policy_id]
        
        # Record policy deletion
        self.policy_history.append({
            'action': 'deleted',
            'policy_id': policy_id,
            'timestamp': time.time(),
            'policy_data': policy_data
        })
        
        logger.info("Security policy deleted", policy_id=policy_id)
        
        return True
    
    def get_policy(self, policy_id: str) -> Optional[SecurityPolicy]:
        """Get security policy by ID"""
        return self.policies.get(policy_id)
    
    def list_policies(
        self,
        scope: Optional[PolicyScope] = None,
        enabled_only: bool = True
    ) -> List[SecurityPolicy]:
        """List security policies with optional filtering"""
        
        policies = list(self.policies.values())
        
        if scope:
            policies = [p for p in policies if p.scope == scope]
        
        if enabled_only:
            policies = [p for p in policies if p.enabled]
        
        return policies
    
    def get_enforcement_stats(self) -> Dict[str, Any]:
        """Get policy enforcement statistics"""
        
        return {
            'total_evaluations': self.enforcement_stats['total_evaluations'],
            'policy_hits': dict(self.enforcement_stats['policy_hits']),
            'actions_taken': dict(self.enforcement_stats['actions_taken']),
            'avg_evaluation_time': self.enforcement_stats['avg_evaluation_time'],
            'most_triggered_policies': sorted(
                self.enforcement_stats['policy_hits'].items(),
                key=lambda x: x[1],
                reverse=True
            )[:5],
            'total_policies': len(self.policies),
            'enabled_policies': len([p for p in self.policies.values() if p.enabled])
        }
    
    def export_policies(self) -> Dict[str, Any]:
        """Export all policies for backup/migration"""
        
        return {
            'policies': {
                policy_id: {
                    **policy.__dict__,
                    'scope': policy.scope.value,
                    'action': policy.action.value
                }
                for policy_id, policy in self.policies.items()
            },
            'export_timestamp': time.time(),
            'policy_count': len(self.policies)
        }
    
    def import_policies(
        self,
        policy_data: Dict[str, Any],
        overwrite: bool = False
    ) -> Dict[str, Any]:
        """Import policies from backup/migration data"""
        
        imported_count = 0
        skipped_count = 0
        errors = []
        
        for policy_id, policy_dict in policy_data.get('policies', {}).items():
            try:
                if policy_id in self.policies and not overwrite:
                    skipped_count += 1
                    continue
                
                # Convert string enums back to enum values
                policy_dict['scope'] = PolicyScope(policy_dict['scope'])
                policy_dict['action'] = PolicyAction(policy_dict['action'])
                
                # Create policy object
                policy = SecurityPolicy(**policy_dict)
                self.policies[policy_id] = policy
                
                imported_count += 1
                
            except Exception as e:
                errors.append(f"Failed to import policy {policy_id}: {str(e)}")
        
        result = {
            'imported_count': imported_count,
            'skipped_count': skipped_count,
            'errors': errors,
            'success': len(errors) == 0
        }
        
        logger.info(
            "Policy import completed",
            imported=imported_count,
            skipped=skipped_count,
            errors=len(errors)
        )
        
        return result
```

This comprehensive input validation and security controls system provides:

### **Key Security Features:**

1. **Multi-layered Validation**: Pattern-based detection for prompt injection, PII, malicious content
2. **Advanced Sanitization**: Content cleaning with customizable rules and PII masking
3. **Policy Engine**: Flexible security policy management with different scopes and actions
4. **Threat Detection**: Sophisticated detection of various attack vectors
5. **Compliance Support**: GDPR, HIPAA, and other regulatory compliance features

The next part will cover **Rate Limiting and Access Control** to complete the security framework.