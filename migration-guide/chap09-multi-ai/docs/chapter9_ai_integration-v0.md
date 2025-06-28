# Chapter 9: AI Service Integration

## Overview

This chapter explores integrating different AI services into your FastAPI application, comparing major cloud providers' AI offerings, and implementing robust patterns for AI service management. We'll migrate our MedAssistant application across AWS Bedrock, Azure OpenAI, and Google Cloud Vertex AI, documenting performance differences, cost implications, and architectural considerations.

**Key Topics Covered:**
- AI service provider comparison and selection
- FastAPI integration patterns for different AI services
- Performance optimization and cost management
- Fallback strategies and reliability patterns
- Multi-model orchestration and routing

---

## AI Service Landscape Overview

### Service Provider Comparison Matrix

| Feature | AWS Bedrock | Azure OpenAI | GCP Vertex AI |
|---------|-------------|--------------|---------------|
| **Best Models** | Claude 3.5 Sonnet | GPT-4 Turbo | Gemini Pro |
| **Pricing Model** | Pay-per-token | Pay-per-token + Reserved | Pay-per-prediction |
| **Rate Limits** | [PLACEHOLDER: Model-specific] | [PLACEHOLDER: Tier-based] | [PLACEHOLDER: Quota-based] |
| **Context Window** | Up to 200K tokens | Up to 128K tokens | Up to 32K tokens |
| **Multimodal** | Limited | Yes (Vision, Audio) | Yes (Vision, Video) |
| **Fine-tuning** | Limited | Yes | Yes |
| **Built-in RAG** | Knowledge Bases | Azure AI Search | Vertex Search |
| **Guardrails** | Native | Content filters | AI Safety filters |
| **Global Availability** | 8+ regions | 10+ regions | 12+ regions |

---

## Section 1: AWS Bedrock Integration

### 1.1 Bedrock Service Architecture

AWS Bedrock provides a unified API to access foundation models from multiple AI companies including Anthropic (Claude), Meta (Llama), and Amazon (Titan). It offers built-in capabilities for RAG through Knowledge Bases and provides enterprise-grade security and compliance.

**Key Advantages:**
- Multiple model providers through single API
- Built-in Knowledge Bases for RAG
- Enterprise security and compliance
- Fine-grained access controls via IAM

### 1.2 Bedrock FastAPI Service Implementation

```python
# services/aws_bedrock_service.py
import boto3
import json
import asyncio
from typing import Dict, Any, Optional, List
import structlog
from botocore.exceptions import ClientError
import aioboto3

from config.settings import settings

logger = structlog.get_logger()

class BedrockService:
    """AWS Bedrock service integration with async support"""
    
    def __init__(self):
        self.region_name = settings.aws_region
        self.session = aioboto3.Session()
        self.model_configs = {
            'claude-3-5-sonnet': {
                'model_id': 'anthropic.claude-3-5-sonnet-20241022-v2:0',
                'max_tokens': 4096,
                'temperature': 0.7
            },
            'claude-3-haiku': {
                'model_id': 'anthropic.claude-3-haiku-20240307-v1:0',
                'max_tokens': 4096,
                'temperature': 0.7
            },
            'llama-3-1': {
                'model_id': 'meta.llama3-1-70b-instruct-v1:0',
                'max_tokens': 2048,
                'temperature': 0.7
            }
        }
    
    async def generate_response(
        self,
        prompt: str,
        model_name: str = 'claude-3-5-sonnet',
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None
    ) -> Dict[str, Any]:
        """Generate response using Bedrock models"""
        
        if model_name not in self.model_configs:
            raise ValueError(f"Unsupported model: {model_name}")
        
        config = self.model_configs[model_name]
        model_id = config['model_id']
        
        # Build request payload based on model type
        if 'anthropic' in model_id:
            payload = self._build_anthropic_payload(
                prompt, system_prompt, max_tokens or config['max_tokens'], 
                temperature or config['temperature']
            )
        elif 'meta' in model_id:
            payload = self._build_llama_payload(
                prompt, max_tokens or config['max_tokens'],
                temperature or config['temperature']
            )
        else:
            raise ValueError(f"Unsupported model type: {model_id}")
        
        try:
            async with self.session.client(
                'bedrock-runtime',
                region_name=self.region_name
            ) as client:
                
                response = await client.invoke_model(
                    modelId=model_id,
                    body=json.dumps(payload),
                    contentType='application/json'
                )
                
                response_body = json.loads(response['body'].read())
                
                # Parse response based on model type
                if 'anthropic' in model_id:
                    return self._parse_anthropic_response(response_body)
                elif 'meta' in model_id:
                    return self._parse_llama_response(response_body)
                
        except ClientError as e:
            logger.error("Bedrock API error", error=str(e), model_id=model_id)
            raise
        except Exception as e:
            logger.error("Unexpected error", error=str(e), model_id=model_id)
            raise
    
    def _build_anthropic_payload(
        self, 
        prompt: str, 
        system_prompt: Optional[str], 
        max_tokens: int, 
        temperature: float
    ) -> Dict:
        """Build payload for Anthropic models"""
        messages = [{"role": "user", "content": prompt}]
        
        payload = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "anthropic_version": "bedrock-2023-05-31"
        }
        
        if system_prompt:
            payload["system"] = system_prompt
        
        return payload
    
    def _parse_anthropic_response(self, response_body: Dict) -> Dict[str, Any]:
        """Parse Anthropic model response"""
        content = response_body.get('content', [])
        if content and len(content) > 0:
            text = content[0].get('text', '')
        else:
            text = ''
        
        return {
            'text': text,
            'model': 'anthropic',
            'usage': response_body.get('usage', {}),
            'stop_reason': response_body.get('stop_reason'),
            'raw_response': response_body
        }
    
    async def query_knowledge_base(
        self,
        query: str,
        knowledge_base_id: str,
        max_results: int = 5
    ) -> Dict[str, Any]:
        """Query Bedrock Knowledge Base for RAG"""
        try:
            async with self.session.client(
                'bedrock-agent-runtime',
                region_name=self.region_name
            ) as kb_client:
                
                response = await kb_client.retrieve(
                    knowledgeBaseId=knowledge_base_id,
                    retrievalQuery={'text': query},
                    retrievalConfiguration={
                        'vectorSearchConfiguration': {
                            'numberOfResults': max_results
                        }
                    }
                )
                
                retrieved_results = response.get('retrievalResults', [])
                
                # Extract relevant information
                contexts = []
                for result in retrieved_results:
                    contexts.append({
                        'content': result.get('content', {}).get('text', ''),
                        'score': result.get('score', 0.0),
                        'location': result.get('location', {}),
                        'metadata': result.get('metadata', {})
                    })
                
                return {
                    'contexts': contexts,
                    'query': query,
                    'knowledge_base_id': knowledge_base_id,
                    'total_results': len(contexts)
                }
                
        except Exception as e:
            logger.error("Knowledge base query failed", error=str(e))
            return {
                'contexts': [],
                'query': query,
                'error': str(e)
            }
    
    async def health_check(self) -> Dict[str, Any]:
        """Check Bedrock service health"""
        try:
            # Simple model invocation test
            test_response = await self.generate_response(
                "Hello, this is a health check.", 
                model_name='claude-3-haiku'  # Use fastest model for health checks
            )
            
            return {
                'status': 'healthy',
                'service': 'aws_bedrock',
                'test_response_tokens': len(test_response.get('text', '').split()),
                'available_models': list(self.model_configs.keys())
            }
            
        except Exception as e:
            logger.error("Bedrock health check failed", error=str(e))
            return {
                'status': 'unhealthy',
                'service': 'aws_bedrock',
                'error': str(e)
            }

# Global instance for dependency injection
bedrock_service = BedrockService()

async def get_bedrock_service() -> BedrockService:
    """FastAPI dependency for Bedrock service"""
    return bedrock_service
```

### 1.3 Bedrock FastAPI Routes

```python
# routers/bedrock_chat.py
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
import structlog
import time

from services.aws_bedrock_service import BedrockService, get_bedrock_service

logger = structlog.get_logger()
router = APIRouter(prefix="/bedrock", tags=["AWS Bedrock"])

class BedrockChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=8000)
    model_name: str = Field(default="claude-3-5-sonnet")
    system_prompt: Optional[str] = Field(None, max_length=2000)
    use_rag: bool = Field(default=False)
    knowledge_base_id: Optional[str] = Field(None)
    max_tokens: Optional[int] = Field(default=None, ge=1, le=4096)
    temperature: Optional[float] = Field(default=None, ge=0.0, le=1.0)

class BedrockChatResponse(BaseModel):
    response: str
    model_used: str
    tokens_used: Dict[str, int]
    processing_time_ms: float
    rag_used: bool = False
    rag_sources: Optional[List[Dict]] = None

@router.post("/chat", response_model=BedrockChatResponse)
async def chat_with_bedrock(
    request: BedrockChatRequest,
    background_tasks: BackgroundTasks,
    bedrock: BedrockService = Depends(get_bedrock_service)
):
    """Chat endpoint using AWS Bedrock models"""
    start_time = time.time()
    
    try:
        if request.use_rag and request.knowledge_base_id:
            # Use RAG-enhanced generation
            rag_results = await bedrock.query_knowledge_base(
                query=request.message,
                knowledge_base_id=request.knowledge_base_id
            )
            
            if rag_results['contexts']:
                # Build context-enhanced prompt
                context_text = "\n\n".join([
                    f"Context {i+1}: {ctx['content']}"
                    for i, ctx in enumerate(rag_results['contexts'][:3])
                ])
                
                enhanced_prompt = f"""Based on the following context, please answer the question:

Context:
{context_text}

Question: {request.message}

Please provide a helpful and accurate answer based on the context provided."""
                
                result = await bedrock.generate_response(
                    prompt=enhanced_prompt,
                    model_name=request.model_name,
                    system_prompt=request.system_prompt,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature
                )
                rag_used = True
                rag_sources = rag_results['contexts']
            else:
                # Fallback to direct generation
                result = await bedrock.generate_response(
                    prompt=request.message,
                    model_name=request.model_name,
                    system_prompt=request.system_prompt,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature
                )
                rag_used = False
                rag_sources = None
        else:
            # Direct generation
            result = await bedrock.generate_response(
                prompt=request.message,
                model_name=request.model_name,
                system_prompt=request.system_prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature
            )
            rag_used = False
            rag_sources = None
        
        processing_time = (time.time() - start_time) * 1000
        
        # Log usage for analytics
        background_tasks.add_task(
            log_bedrock_usage,
            model_name=request.model_name,
            tokens=result.get('usage', {}),
            processing_time=processing_time,
            rag_used=rag_used
        )
        
        return BedrockChatResponse(
            response=result['text'],
            model_used=request.model_name,
            tokens_used=result.get('usage', {}),
            processing_time_ms=processing_time,
            rag_used=rag_used,
            rag_sources=rag_sources
        )
        
    except Exception as e:
        logger.error("Bedrock chat failed", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Bedrock service error: {str(e)}"
        )

@router.get("/models")
async def list_available_models(
    bedrock: BedrockService = Depends(get_bedrock_service)
):
    """List available Bedrock models"""
    return {
        "available_models": list(bedrock.model_configs.keys()),
        "model_details": bedrock.model_configs
    }

async def log_bedrock_usage(
    model_name: str,
    tokens: Dict,
    processing_time: float,
    rag_used: bool
):
    """Background task to log Bedrock usage"""
    logger.info(
        "Bedrock usage logged",
        model=model_name,
        input_tokens=tokens.get('prompt_tokens', 0),
        output_tokens=tokens.get('completion_tokens', 0),
        processing_time_ms=processing_time,
        rag_enabled=rag_used
    )
```

### 1.4 Bedrock Performance Characteristics

**Strengths:**
- Excellent model variety (Claude, Llama, Titan)
- Built-in Knowledge Bases for RAG
- Enterprise security and compliance
- Consistent API across models

**Considerations:**
- [PLACEHOLDER: Response time benchmarks]
- [PLACEHOLDER: Cost analysis per model]
- [PLACEHOLDER: Rate limiting behavior]

---

## Section 2: Azure OpenAI Integration

### 2.1 Azure OpenAI Service Architecture

Azure OpenAI Service provides access to OpenAI's models through Microsoft's cloud infrastructure with enterprise-grade security, compliance, and SLA guarantees. It offers unique capabilities like content filtering, custom fine-tuning, and integration with Azure's ecosystem.

**Key Advantages:**
- Access to latest OpenAI models (GPT-4, GPT-3.5)
- Enterprise security and compliance
- Custom fine-tuning capabilities
- Built-in content moderation
- Integration with Azure ecosystem

### 2.2 Azure OpenAI FastAPI Service Implementation

```python
# services/azure_openai_service.py
import asyncio
import json
from typing import Dict, Any, Optional, List
import structlog
import httpx
from openai import AsyncAzureOpenAI

from config.settings import settings

logger = structlog.get_logger()

class AzureOpenAIService:
    """Azure OpenAI service integration"""
    
    def __init__(self):
        self.client = AsyncAzureOpenAI(
            api_key=settings.azure_openai_api_key,
            api_version=settings.azure_openai_api_version,
            azure_endpoint=settings.azure_openai_endpoint
        )
        
        self.model_configs = {
            'gpt-4-turbo': {
                'deployment_name': 'gpt-4-turbo',
                'max_tokens': 4096,
                'temperature': 0.7
            },
            'gpt-4': {
                'deployment_name': 'gpt-4',
                'max_tokens': 8192,
                'temperature': 0.7
            },
            'gpt-35-turbo': {
                'deployment_name': 'gpt-35-turbo',
                'max_tokens': 4096,
                'temperature': 0.7
            }
        }
    
    async def generate_response(
        self,
        prompt: str,
        model_name: str = 'gpt-4-turbo',
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None
    ) -> Dict[str, Any]:
        """Generate response using Azure OpenAI models"""
        
        if model_name not in self.model_configs:
            raise ValueError(f"Unsupported model: {model_name}")
        
        config = self.model_configs[model_name]
        deployment_name = config['deployment_name']
        
        # Build messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = await self.client.chat.completions.create(
                model=deployment_name,
                messages=messages,
                max_tokens=max_tokens or config['max_tokens'],
                temperature=temperature or config['temperature']
            )
            
            choice = response.choices[0]
            
            return {
                'text': choice.message.content,
                'model': model_name,
                'usage': {
                    'prompt_tokens': response.usage.prompt_tokens,
                    'completion_tokens': response.usage.completion_tokens,
                    'total_tokens': response.usage.total_tokens
                },
                'finish_reason': choice.finish_reason,
                'raw_response': response.model_dump()
            }
                
        except Exception as e:
            logger.error("Azure OpenAI API error", error=str(e), model=model_name)
            raise
    
    async def moderate_content(self, text: str) -> Dict[str, Any]:
        """Check content using Azure OpenAI moderation"""
        try:
            response = await self.client.moderations.create(input=text)
            
            result = response.results[0]
            
            return {
                'flagged': result.flagged,
                'categories': result.categories.model_dump(),
                'category_scores': result.category_scores.model_dump()
            }
            
        except Exception as e:
            logger.error("Azure OpenAI moderation error", error=str(e))
            raise
    
    async def health_check(self) -> Dict[str, Any]:
        """Check Azure OpenAI service health"""
        try:
            # Simple completion test
            test_response = await self.generate_response(
                "Hello, this is a health check.",
                model_name='gpt-35-turbo'  # Use fastest model for health checks
            )
            
            return {
                'status': 'healthy',
                'service': 'azure_openai',
                'test_response_tokens': test_response.get('usage', {}).get('total_tokens', 0),
                'available_models': list(self.model_configs.keys())
            }
            
        except Exception as e:
            logger.error("Azure OpenAI health check failed", error=str(e))
            return {
                'status': 'unhealthy',
                'service': 'azure_openai',
                'error': str(e)
            }

# Global instance for dependency injection
azure_openai_service = AzureOpenAIService()

async def get_azure_openai_service() -> AzureOpenAIService:
    """FastAPI dependency for Azure OpenAI service"""
    return azure_openai_service
```

### 2.3 Azure OpenAI Performance Characteristics

**Strengths:**
- Latest OpenAI models with enterprise SLA
- Built-in content moderation
- Fine-tuning capabilities
- Strong integration with Azure services

**Considerations:**
- [PLACEHOLDER: Response time vs. OpenAI direct]
- [PLACEHOLDER: Regional availability limitations]
- [PLACEHOLDER: Cost comparison with other providers]

---

## Section 3: Google Cloud Vertex AI Integration

### 3.1 Vertex AI Service Architecture

Google Cloud Vertex AI provides access to Google's foundation models including Gemini, PaLM, and Codey, along with comprehensive MLOps capabilities. It offers unique multimodal capabilities and tight integration with Google's search and knowledge services.

**Key Advantages:**
- Google's latest models (Gemini Pro, PaLM)
- Strong multimodal capabilities
- Integrated MLOps platform
- Vertex AI Search for RAG
- Competitive pricing

### 3.2 Vertex AI Performance Characteristics

**Strengths:**
- Excellent multimodal capabilities
- Competitive pricing structure
- Strong integration with Google services
- Advanced MLOps features

**Considerations:**
- [PLACEHOLDER: Model availability by region]
- [PLACEHOLDER: Performance benchmarks]
- [PLACEHOLDER: Learning curve for Google-specific APIs]

---

## Section 4: Multi-Provider Orchestration

### 4.1 AI Service Router Pattern

```python
# services/ai_router_service.py
from typing import Dict, Any, Optional
import structlog
from enum import Enum

from services.aws_bedrock_service import BedrockService
from services.azure_openai_service import AzureOpenAIService

logger = structlog.get_logger()

class AIProvider(str, Enum):
    AWS_BEDROCK = "aws_bedrock"
    AZURE_OPENAI = "azure_openai"
    GCP_VERTEX = "gcp_vertex"

class AIServiceRouter:
    """Intelligent routing between AI service providers"""
    
    def __init__(self):
        self.providers = {
            AIProvider.AWS_BEDROCK: BedrockService(),
            AIProvider.AZURE_OPENAI: AzureOpenAIService(),
            # GCP Vertex AI service would be added here
        }
        
        self.provider_stats = {
            provider: {
                'total_requests': 0,
                'total_failures': 0,
                'avg_response_time': 0.0,
                'is_healthy': True
            }
            for provider in AIProvider
        }
    
    async def route_request(
        self,
        prompt: str,
        preferred_provider: Optional[AIProvider] = None,
        fallback_enabled: bool = True
    ) -> Dict[str, Any]:
        """Route AI request to optimal provider"""
        
        # Use preferred provider or select best available
        provider = preferred_provider or await self._select_best_provider()
        
        # Make request with fallback
        return await self._make_request_with_fallback(
            provider, prompt, fallback_enabled
        )
    
    async def _select_best_provider(self) -> AIProvider:
        """Select provider based on health and performance"""
        
        # Simple selection logic - can be enhanced
        healthy_providers = [
            provider for provider in AIProvider
            if self.provider_stats[provider]['is_healthy']
        ]
        
        if not healthy_providers:
            return AIProvider.AWS_BEDROCK  # Default fallback
        
        # Select provider with best response time
        best_provider = min(
            healthy_providers,
            key=lambda p: self.provider_stats[p]['avg_response_time']
        )
        
        return best_provider
    
    async def _make_request_with_fallback(
        self,
        primary_provider: AIProvider,
        prompt: str,
        fallback_enabled: bool
    ) -> Dict[str, Any]:
        """Make request with automatic fallback"""
        
        providers_to_try = [primary_provider]
        
        if fallback_enabled:
            # Add other providers as fallback
            fallback_providers = [p for p in AIProvider if p != primary_provider]
            providers_to_try.extend(fallback_providers)
        
        for provider in providers_to_try:
            try:
                service = self.providers[provider]
                result = await service.generate_response(prompt)
                
                result['provider_used'] = provider.value
                result['fallback_used'] = provider != primary_provider
                
                return result
                
            except Exception as e:
                logger.warning(
                    "Provider request failed",
                    provider=provider,
                    error=str(e)
                )
                continue
        
        raise Exception("All AI providers failed")

# Global router instance
ai_router = AIServiceRouter()

async def get_ai_router() -> AIServiceRouter:
    """FastAPI dependency for AI router"""
    return ai_router
```

### 4.2 Cost and Performance Analysis

**Placeholder sections for actual implementation data:**

#### 4.2.1 Response Time Comparison
- [PLACEHOLDER: Benchmark results across providers]
- [PLACEHOLDER: Performance under load]
- [PLACEHOLDER: Cold start times]

#### 4.2.2 Cost Analysis
- [PLACEHOLDER: Cost per 1K tokens by provider/model]
- [PLACEHOLDER: Monthly cost projections]
- [PLACEHOLDER: Cost optimization strategies]

#### 4.2.3 Reliability Metrics
- [PLACEHOLDER: Uptime statistics]
- [PLACEHOLDER: Error rates by provider]
- [PLACEHOLDER: Failover performance]

---

## Summary

This chapter provided a comprehensive approach to integrating multiple AI services into FastAPI applications. Key takeaways include:

**Implementation Strategy:**
- Vendor-specific service implementations with consistent APIs
- Intelligent routing with fallback capabilities
- Performance monitoring and cost optimization
- Enterprise-grade reliability patterns

**Provider Selection Criteria:**
- Model capabilities and availability
- Performance characteristics
- Cost structure and predictability
- Integration complexity and ecosystem

**Best Practices:**
- Abstract vendor-specific implementations
- Implement robust error handling and fallbacks
- Monitor usage and costs continuously
- Plan for vendor lock-in avoidance

The multi-provider approach provides flexibility and resilience while allowing optimization based on specific use cases and requirements.

---

*Next: Chapter 10 - Production Deployment and Scaling*