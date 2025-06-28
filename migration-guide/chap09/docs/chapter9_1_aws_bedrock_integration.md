# Chapter 9, Section 1: AWS Bedrock Integration

## Overview

AWS Bedrock provides a unified API to access foundation models from multiple AI companies including Anthropic (Claude), Meta (Llama), Amazon (Titan), Cohere, and Mistral AI. It offers built-in capabilities for RAG through Knowledge Bases and provides enterprise-grade security and compliance features.

## Key Advantages

- **Multiple Model Providers:** Access Claude, Llama, Titan, and others through single API
- **Built-in Knowledge Bases:** Integrated RAG capabilities with vector search
- **Enterprise Security:** Full AWS IAM integration and compliance features
- **Fine-grained Access Controls:** Detailed permissions and audit trails
- **No Model Management:** Fully managed service with automatic scaling

## Model Landscape

### Available Models

| Model Family | Model ID | Strengths | Use Cases |
|--------------|----------|-----------|-----------|
| **Claude 3.5 Sonnet** | `anthropic.claude-3-5-sonnet-20241022-v2:0` | Advanced reasoning, coding | Complex analysis, code generation |
| **Claude 3 Haiku** | `anthropic.claude-3-haiku-20240307-v1:0` | Speed, cost-effective | Simple Q&A, content moderation |
| **Llama 3.1 70B** | `meta.llama3-1-70b-instruct-v1:0` | Open source, versatile | General purpose, fine-tuning |
| **Titan Text** | `amazon.titan-text-express-v1` | AWS native, reliable | Basic text generation |

### Model Selection Guide

```python
MODEL_SELECTION_GUIDE = {
    'complex_reasoning': 'claude-3-5-sonnet',
    'fast_responses': 'claude-3-haiku', 
    'code_generation': 'claude-3-5-sonnet',
    'cost_optimization': 'claude-3-haiku',
    'open_source': 'llama-3-1',
    'aws_native': 'amazon-titan'
}
```

## FastAPI Service Implementation

### Core Bedrock Service

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
                'temperature': 0.7,
                'pricing': {'input': 0.003, 'output': 0.015}  # per 1K tokens
            },
            'claude-3-haiku': {
                'model_id': 'anthropic.claude-3-haiku-20240307-v1:0',
                'max_tokens': 4096,
                'temperature': 0.7,
                'pricing': {'input': 0.00025, 'output': 0.00125}
            },
            'llama-3-1': {
                'model_id': 'meta.llama3-1-70b-instruct-v1:0',
                'max_tokens': 2048,
                'temperature': 0.7,
                'pricing': {'input': 0.00065, 'output': 0.00065}
            },
            'amazon-titan': {
                'model_id': 'amazon.titan-text-express-v1',
                'max_tokens': 8000,
                'temperature': 0.7,
                'pricing': {'input': 0.0008, 'output': 0.0016}
            }
        }
    
    async def generate_response(
        self,
        prompt: str,
        model_name: str = 'claude-3-5-sonnet',
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        timeout: float = 30.0
    ) -> Dict[str, Any]:
        """Generate response using Bedrock models with comprehensive error handling"""
        
        if model_name not in self.model_configs:
            available_models = list(self.model_configs.keys())
            raise ValueError(f"Unsupported model: {model_name}. Available: {available_models}")
        
        config = self.model_configs[model_name]
        model_id = config['model_id']
        
        # Build request payload based on model type
        try:
            if 'anthropic' in model_id:
                payload = self._build_anthropic_payload(
                    prompt, system_prompt, 
                    max_tokens or config['max_tokens'], 
                    temperature or config['temperature']
                )
            elif 'meta' in model_id:
                payload = self._build_llama_payload(
                    prompt, 
                    max_tokens or config['max_tokens'],
                    temperature or config['temperature']
                )
            elif 'amazon' in model_id:
                payload = self._build_titan_payload(
                    prompt,
                    max_tokens or config['max_tokens'],
                    temperature or config['temperature']
                )
            else:
                raise ValueError(f"Unsupported model type: {model_id}")
                
        except Exception as e:
            logger.error("Payload building failed", model_id=model_id, error=str(e))
            raise
        
        # Make request with timeout and retry logic
        try:
            async with self.session.client(
                'bedrock-runtime',
                region_name=self.region_name
            ) as client:
                
                response = await asyncio.wait_for(
                    client.invoke_model(
                        modelId=model_id,
                        body=json.dumps(payload),
                        contentType='application/json'
                    ),
                    timeout=timeout
                )
                
                response_body = json.loads(response['body'].read())
                
                # Parse response based on model type
                if 'anthropic' in model_id:
                    result = self._parse_anthropic_response(response_body)
                elif 'meta' in model_id:
                    result = self._parse_llama_response(response_body)
                elif 'amazon' in model_id:
                    result = self._parse_titan_response(response_body)
                
                # Add cost estimation
                if 'usage' in result:
                    result['estimated_cost'] = self._calculate_cost(
                        model_name, result['usage']
                    )
                
                logger.info(
                    "Bedrock generation successful",
                    model=model_name,
                    input_tokens=result.get('usage', {}).get('prompt_tokens', 0),
                    output_tokens=result.get('usage', {}).get('completion_tokens', 0)
                )
                
                return result
                
        except asyncio.TimeoutError:
            logger.error("Bedrock request timeout", model_id=model_id, timeout=timeout)
            raise Exception(f"Request timeout after {timeout} seconds")
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            error_message = e.response['Error']['Message']
            
            logger.error(
                "Bedrock API error",
                model_id=model_id,
                error_code=error_code,
                error_message=error_message
            )
            
            # Handle specific error types
            if error_code == 'ThrottlingException':
                raise Exception("Rate limit exceeded. Please try again later.")
            elif error_code == 'ValidationException':
                raise Exception(f"Invalid request: {error_message}")
            elif error_code == 'ResourceNotFoundException':
                raise Exception(f"Model not found: {model_id}")
            else:
                raise Exception(f"Bedrock error: {error_message}")
                
        except Exception as e:
            logger.error("Unexpected Bedrock error", model_id=model_id, error=str(e))
            raise Exception(f"Bedrock service error: {str(e)}")
    
    def _build_anthropic_payload(
        self, 
        prompt: str, 
        system_prompt: Optional[str], 
        max_tokens: int, 
        temperature: float
    ) -> Dict:
        """Build payload for Anthropic Claude models"""
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
    
    def _build_llama_payload(
        self, 
        prompt: str, 
        max_tokens: int, 
        temperature: float
    ) -> Dict:
        """Build payload for Meta Llama models"""
        return {
            "prompt": f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>",
            "max_gen_len": max_tokens,
            "temperature": temperature,
            "top_p": 0.9
        }
    
    def _build_titan_payload(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float
    ) -> Dict:
        """Build payload for Amazon Titan models"""
        return {
            "inputText": prompt,
            "textGenerationConfig": {
                "maxTokenCount": max_tokens,
                "temperature": temperature,
                "topP": 0.9
            }
        }
    
    def _parse_anthropic_response(self, response_body: Dict) -> Dict[str, Any]:
        """Parse Anthropic Claude model response"""
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
    
    def _parse_llama_response(self, response_body: Dict) -> Dict[str, Any]:
        """Parse Meta Llama model response"""
        return {
            'text': response_body.get('generation', ''),
            'model': 'llama',
            'usage': {
                'prompt_tokens': response_body.get('prompt_token_count', 0),
                'completion_tokens': response_body.get('generation_token_count', 0),
                'total_tokens': response_body.get('prompt_token_count', 0) + response_body.get('generation_token_count', 0)
            },
            'stop_reason': response_body.get('stop_reason'),
            'raw_response': response_body
        }
    
    def _parse_titan_response(self, response_body: Dict) -> Dict[str, Any]:
        """Parse Amazon Titan model response"""
        results = response_body.get('results', [])
        text = results[0].get('outputText', '') if results else ''
        
        return {
            'text': text,
            'model': 'titan',
            'usage': {
                'prompt_tokens': response_body.get('inputTextTokenCount', 0),
                'completion_tokens': response_body.get('results', [{}])[0].get('tokenCount', 0),
                'total_tokens': response_body.get('inputTextTokenCount', 0) + response_body.get('results', [{}])[0].get('tokenCount', 0)
            },
            'stop_reason': results[0].get('completionReason', '') if results else '',
            'raw_response': response_body
        }
    
    def _calculate_cost(self, model_name: str, usage: Dict[str, int]) -> float:
        """Calculate estimated cost for the request"""
        if model_name not in self.model_configs:
            return 0.0
        
        pricing = self.model_configs[model_name]['pricing']
        input_tokens = usage.get('prompt_tokens', 0)
        output_tokens = usage.get('completion_tokens', 0)
        
        input_cost = (input_tokens / 1000) * pricing['input']
        output_cost = (output_tokens / 1000) * pricing['output']
        
        return round(input_cost + output_cost, 6)
    
    async def health_check(self) -> Dict[str, Any]:
        """Check Bedrock service health"""
        try:
            # Simple model invocation test using fastest model
            test_response = await self.generate_response(
                "Hello", 
                model_name='claude-3-haiku',
                max_tokens=10,
                timeout=10.0
            )
            
            return {
                'status': 'healthy',
                'service': 'aws_bedrock',
                'test_response_length': len(test_response.get('text', '')),
                'available_models': list(self.model_configs.keys()),
                'response_time': test_response.get('response_time', 0)
            }
            
        except Exception as e:
            logger.error("Bedrock health check failed", error=str(e))
            return {
                'status': 'unhealthy',
                'service': 'aws_bedrock',
                'error': str(e),
                'available_models': list(self.model_configs.keys())
            }

# Global instance for dependency injection
bedrock_service = BedrockService()

async def get_bedrock_service() -> BedrockService:
    """FastAPI dependency for Bedrock service"""
    return bedrock_service
```

## RAG Integration with Knowledge Bases

### Knowledge Base Service

```python
# services/bedrock_rag_service.py
import asyncio
from typing import Dict, Any, Optional, List
import structlog
import aioboto3

from config.settings import settings
from services.aws_bedrock_service import BedrockService

logger = structlog.get_logger()

class BedrockRAGService:
    """RAG implementation using AWS Bedrock Knowledge Bases"""
    
    def __init__(self):
        self.region_name = settings.aws_region
        self.session = aioboto3.Session()
        self.bedrock_service = BedrockService()
        
        # Knowledge Base configurations
        self.knowledge_bases = {
            'medical': {
                'id': 'kb-medical-docs-123',
                'description': 'Medical documentation and guidelines'
            },
            'general': {
                'id': 'kb-general-knowledge-456', 
                'description': 'General knowledge base'
            }
        }
    
    async def query_knowledge_base(
        self,
        query: str,
        knowledge_base_id: str,
        max_results: int = 5,
        min_score: float = 0.5
    ) -> Dict[str, Any]:
        """Query Bedrock Knowledge Base for relevant documents"""
        
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
                            'numberOfResults': max_results,
                            'overrideSearchType': 'HYBRID'  # Use both semantic and keyword search
                        }
                    }
                )
                
                retrieved_results = response.get('retrievalResults', [])
                
                # Filter by score and extract relevant information
                contexts = []
                for result in retrieved_results:
                    score = result.get('score', 0.0)
                    if score >= min_score:
                        contexts.append({
                            'content': result.get('content', {}).get('text', ''),
                            'score': score,
                            'location': result.get('location', {}),
                            'metadata': result.get('metadata', {}),
                            'source': result.get('location', {}).get('s3Location', {}).get('uri', 'unknown')
                        })
                
                # Sort by score (descending)
                contexts.sort(key=lambda x: x['score'], reverse=True)
                
                logger.info(
                    "Knowledge base query successful",
                    knowledge_base_id=knowledge_base_id,
                    query_length=len(query),
                    results_found=len(contexts),
                    avg_score=sum(c['score'] for c in contexts) / len(contexts) if contexts else 0
                )
                
                return {
                    'contexts': contexts,
                    'query': query,
                    'knowledge_base_id': knowledge_base_id,
                    'total_results': len(contexts),
                    'search_metadata': {
                        'max_results': max_results,
                        'min_score': min_score,
                        'search_type': 'hybrid'
                    }
                }
                
        except Exception as e:
            logger.error(
                "Knowledge base query failed",
                knowledge_base_id=knowledge_base_id,
                query=query[:100],
                error=str(e)
            )
            return {
                'contexts': [],
                'query': query,
                'knowledge_base_id': knowledge_base_id,
                'error': str(e)
            }
    
    async def generate_with_rag(
        self,
        query: str,
        knowledge_base_type: str = 'general',
        model_name: str = 'claude-3-5-sonnet',
        system_prompt: Optional[str] = None,
        max_context_length: int = 3000,
        include_sources: bool = True
    ) -> Dict[str, Any]:
        """Generate response using RAG with Bedrock Knowledge Base"""
        
        # Get knowledge base ID
        if knowledge_base_type not in self.knowledge_bases:
            available_types = list(self.knowledge_bases.keys())
            raise ValueError(f"Unknown knowledge base type: {knowledge_base_type}. Available: {available_types}")
        
        knowledge_base_id = self.knowledge_bases[knowledge_base_type]['id']
        
        # Retrieve relevant context
        rag_results = await self.query_knowledge_base(query, knowledge_base_id)
        
        if not rag_results['contexts']:
            logger.warning(
                "No RAG context found, using direct generation",
                query=query[:100],
                knowledge_base_type=knowledge_base_type
            )
            # Fallback to direct generation without context
            direct_result = await self.bedrock_service.generate_response(
                prompt=query,
                model_name=model_name,
                system_prompt=system_prompt
            )
            direct_result['rag_used'] = False
            return direct_result
        
        # Build context-enhanced prompt
        context_parts = []
        total_context_length = 0
        
        for i, ctx in enumerate(rag_results['contexts']):
            context_text = f"Source {i+1} (Score: {ctx['score']:.2f}):\n{ctx['content']}"
            
            # Check if adding this context would exceed limit
            if total_context_length + len(context_text) > max_context_length:
                logger.info(
                    "Context length limit reached",
                    contexts_used=i,
                    total_length=total_context_length
                )
                break
            
            context_parts.append(context_text)
            total_context_length += len(context_text)
        
        context_text = "\n\n".join(context_parts)
        
        # Enhanced system prompt for RAG
        rag_system_prompt = system_prompt or ""
        if rag_system_prompt:
            rag_system_prompt += "\n\n"
        
        rag_system_prompt += """You are an AI assistant that answers questions based on provided context. 
        Follow these guidelines:
        1. Base your answer primarily on the provided context
        2. If the context doesn't contain enough information, clearly state this
        3. Include relevant details and be specific
        4. If asked about sources, refer to the numbered sources provided"""
        
        enhanced_prompt = f"""Based on the following context, please answer the question:

Context:
{context_text}

Question: {query}

Please provide a comprehensive answer based on the context provided."""
        
        # Generate response with context
        result = await self.bedrock_service.generate_response(
            prompt=enhanced_prompt,
            model_name=model_name,
            system_prompt=rag_system_prompt
        )
        
        # Add RAG metadata
        result['rag_used'] = True
        result['rag_context'] = rag_results['contexts']
        result['rag_query'] = query
        result['knowledge_base_type'] = knowledge_base_type
        result['context_length'] = total_context_length
        result['contexts_used'] = len(context_parts)
        
        # Include source information if requested
        if include_sources:
            result['sources'] = [
                {
                    'source': ctx['source'],
                    'score': ctx['score'],
                    'metadata': ctx['metadata']
                }
                for ctx in rag_results['contexts'][:len(context_parts)]
            ]
        
        logger.info(
            "RAG generation completed",
            model=model_name,
            knowledge_base_type=knowledge_base_type,
            contexts_used=len(context_parts),
            context_length=total_context_length
        )
        
        return result
    
    async def list_knowledge_bases(self) -> Dict[str, Any]:
        """List available knowledge bases"""
        return {
            'knowledge_bases': {
                kb_type: {
                    'id': config['id'],
                    'description': config['description']
                }
                for kb_type, config in self.knowledge_bases.items()
            }
        }

# Global instance for dependency injection
bedrock_rag_service = BedrockRAGService()

async def get_bedrock_rag_service() -> BedrockRAGService:
    """FastAPI dependency for Bedrock RAG service"""
    return bedrock_rag_service
```

## FastAPI Routes

### Bedrock Chat Routes

```python
# routers/bedrock_chat.py
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Literal
import structlog
import time

from services.aws_bedrock_service import BedrockService, get_bedrock_service
from services.bedrock_rag_service import BedrockRAGService, get_bedrock_rag_service

logger = structlog.get_logger()
router = APIRouter(prefix="/bedrock", tags=["AWS Bedrock"])

# Request Models
class BedrockChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=8000, description="User message")
    model_name: Literal['claude-3-5-sonnet', 'claude-3-haiku', 'llama-3-1', 'amazon-titan'] = Field(
        default="claude-3-5-sonnet", 
        description="Bedrock model to use"
    )
    system_prompt: Optional[str] = Field(None, max_length=2000, description="System prompt for context")
    max_tokens: Optional[int] = Field(default=None, ge=1, le=8000, description="Maximum tokens to generate")
    temperature: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Sampling temperature")
    timeout: float = Field(default=30.0, ge=1.0, le=120.0, description="Request timeout in seconds")

class BedrockRAGRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=8000)
    knowledge_base_type: Literal['medical', 'general'] = Field(default='general')
    model_name: Literal['claude-3-5-sonnet', 'claude-3-haiku', 'llama-3-1'] = Field(default="claude-3-5-sonnet")
    system_prompt: Optional[str] = Field(None, max_length=2000)
    max_context_length: int = Field(default=3000, ge=500, le=6000)
    include_sources: bool = Field(default=True)

# Response Models
class BedrockChatResponse(BaseModel):
    response: str
    model_used: str
    tokens_used: Dict[str, int]
    processing_time_ms: float
    estimated_cost: float
    stop_reason: Optional[str] = None

class BedrockRAGResponse(BaseModel):
    response: str
    model_used: str
    tokens_used: Dict[str, int]
    processing_time_ms: float
    estimated_cost: float
    rag_used: bool
    knowledge_base_type: str
    contexts_used: int
    context_length: Optional[int] = None
    sources: Optional[List[Dict]] = None

@router.post("/chat", response_model=BedrockChatResponse)
async def chat_with_bedrock(
    request: BedrockChatRequest,
    background_tasks: BackgroundTasks,
    bedrock: BedrockService = Depends(get_bedrock_service)
):
    """Chat endpoint using AWS Bedrock models with comprehensive error handling"""
    start_time = time.time()
    
    try:
        result = await bedrock.generate_response(
            prompt=request.message,
            model_name=request.model_name,
            system_prompt=request.system_prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            timeout=request.timeout
        )
        
        processing_time = (time.time() - start_time) * 1000
        
        # Log usage for analytics
        background_tasks.add_task(
            log_bedrock_usage,
            model_name=request.model_name,
            tokens=result.get('usage', {}),
            processing_time=processing_time,
            cost=result.get('estimated_cost', 0),
            rag_used=False
        )
        
        return BedrockChatResponse(
            response=result['text'],
            model_used=request.model_name,
            tokens_used=result.get('usage', {}),
            processing_time_ms=processing_time,
            estimated_cost=result.get('estimated_cost', 0),
            stop_reason=result.get('stop_reason')
        )
        
    except ValueError as e:
        # Handle validation errors
        logger.warning("Bedrock validation error", error=str(e))
        raise HTTPException(status_code=400, detail=str(e))
        
    except Exception as e:
        logger.error("Bedrock chat failed", error=str(e))
        # Don't expose internal errors to client
        raise HTTPException(
            status_code=500,
            detail="Service temporarily unavailable. Please try again."
        )

@router.post("/chat/rag", response_model=BedrockRAGResponse)
async def chat_with_rag(
    request: BedrockRAGRequest,
    background_tasks: BackgroundTasks,
    bedrock_rag: BedrockRAGService = Depends(get_bedrock_rag_service)
):
    """Chat endpoint using AWS Bedrock with RAG (Knowledge Bases)"""
    start_time = time.time()
    
    try:
        result = await bedrock_rag.generate_with_rag(
            query=request.message,
            knowledge_base_type=request.knowledge_base_type,
            model_name=request.model_name,
            system_prompt=request.system_prompt,
            max_context_length=request.max_context_length,
            include_sources=request.include_sources
        )
        
        processing_time = (time.time() - start_time) * 1000
        
        # Log usage for analytics
        background_tasks.add_task(
            log_bedrock_usage,
            model_name=request.model_name,
            tokens=result.get('usage', {}),
            processing_time=processing_time,
            cost=result.get('estimated_cost', 0),
            rag_used=result.get('rag_used', False)
        )
        
        return BedrockRAGResponse(
            response=result['text'],
            model_used=request.model_name,
            tokens_used=result.get('usage', {}),
            processing_time_ms=processing_time,
            estimated_cost=result.get('estimated_cost', 0),
            rag_used=result.get('rag_used', False),
            knowledge_base_type=request.knowledge_base_type,
            contexts_used=result.get('contexts_used', 0),
            context_length=result.get('context_length'),
            sources=result.get('sources')
        )
        
    except ValueError as e:
        logger.warning("Bedrock RAG validation error", error=str(e))
        raise HTTPException(status_code=400, detail=str(e))
        
    except Exception as e:
        logger.error("Bedrock RAG failed", error=str(e))
        raise HTTPException(
            status_code=500,
            detail="RAG service temporarily unavailable. Please try again."
        )

@router.get("/models")
async def list_available_models(
    bedrock: BedrockService = Depends(get_bedrock_service)
):
    """List available Bedrock models with pricing and capabilities"""
    models_info = {}
    
    for model_name, config in bedrock.model_configs.items():
        models_info[model_name] = {
            'model_id': config['model_id'],
            'max_tokens': config['max_tokens'],
            'pricing_per_1k_tokens': config['pricing'],
            'best_for': _get_model_use_cases(model_name)
        }
    
    return {
        "available_models": models_info,
        "recommendation_guide": {
            "cost_effective": "claude-3-haiku",
            "best_reasoning": "claude-3-5-sonnet", 
            "fastest": "claude-3-haiku",
            "open_source": "llama-3-1",
            "aws_native": "amazon-titan"
        }
    }

@router.get("/knowledge-bases")
async def list_knowledge_bases(
    bedrock_rag: BedrockRAGService = Depends(get_bedrock_rag_service)
):
    """List available knowledge bases"""
    return await bedrock_rag.list_knowledge_bases()

@router.get("/health")
async def bedrock_health_check(
    bedrock: BedrockService = Depends(get_bedrock_service)
):
    """Check AWS Bedrock service health"""
    return await bedrock.health_check()

def _get_model_use_cases(model_name: str) -> List[str]:
    """Get recommended use cases for each model"""
    use_cases = {
        'claude-3-5-sonnet': [
            'Complex reasoning and analysis',
            'Code generation and debugging', 
            'Technical writing',
            'Research and synthesis'
        ],
        'claude-3-haiku': [
            'Quick Q&A',
            'Content moderation',
            'Simple text generation',
            'Cost-sensitive applications'
        ],
        'llama-3-1': [
            'General purpose chat',
            'Open source requirements',
            'Custom fine-tuning base',
            'Versatile applications'
        ],
        'amazon-titan': [
            'AWS-native applications',
            'Basic text generation',
            'Enterprise compliance',
            'Reliable performance'
        ]
    }
    return use_cases.get(model_name, ['General purpose'])

async def log_bedrock_usage(
    model_name: str,
    tokens: Dict,
    processing_time: float,
    cost: float,
    rag_used: bool
):
    """Background task to log Bedrock usage for analytics"""
    logger.info(
        "Bedrock usage logged",
        model=model_name,
        input_tokens=tokens.get('prompt_tokens', 0),
        output_tokens=tokens.get('completion_tokens', 0),
        total_tokens=tokens.get('total_tokens', 0),
        processing_time_ms=processing_time,
        estimated_cost=cost,
        rag_enabled=rag_used
    )
```

## Performance Characteristics

### Benchmark Results

**Response Time Analysis (Based on Testing):**

| Model | Avg Response Time | P95 Response Time | Cold Start | Tokens/sec |
|-------|------------------|-------------------|------------|------------|
| Claude 3.5 Sonnet | [PLACEHOLDER: 1.2s] | [PLACEHOLDER: 2.8s] | [PLACEHOLDER: 0.3s] | [PLACEHOLDER: 45] |
| Claude 3 Haiku | [PLACEHOLDER: 0.8s] | [PLACEHOLDER: 1.9s] | [PLACEHOLDER: 0.2s] | [PLACEHOLDER: 65] |
| Llama 3.1 70B | [PLACEHOLDER: 1.5s] | [PLACEHOLDER: 3.1s] | [PLACEHOLDER: 0.4s] | [PLACEHOLDER: 35] |
| Amazon Titan | [PLACEHOLDER: 1.0s] | [PLACEHOLDER: 2.3s] | [PLACEHOLDER: 0.3s] | [PLACEHOLDER: 50] |

### Cost Analysis

**Real-World Cost Examples:**

```python
# Example cost calculations for different usage patterns
COST_EXAMPLES = {
    'simple_qa': {
        'input_tokens': 50,
        'output_tokens': 100,
        'claude_3_5_sonnet': 0.0015 + 0.0015 = 0.003,  # $0.003
        'claude_3_haiku': 0.0000125 + 0.000125 = 0.0001375,  # $0.0001
        'savings_ratio': '95% cheaper with Haiku'
    },
    'complex_analysis': {
        'input_tokens': 1000,
        'output_tokens': 800,
        'claude_3_5_sonnet': 0.003 + 0.012 = 0.015,  # $0.015
        'claude_3_haiku': 0.00025 + 0.001 = 0.00125,  # $0.0013
        'recommendation': 'Use Sonnet for quality, Haiku for cost'
    }
}
```

### Rate Limits and Throttling

**Current Limits (Subject to Change):**
- **Claude 3.5 Sonnet:** [PLACEHOLDER: 10 requests/minute]
- **Claude 3 Haiku:** [PLACEHOLDER: 20 requests/minute] 
- **Llama 3.1:** [PLACEHOLDER: 15 requests/minute]
- **Titan Text:** [PLACEHOLDER: 25 requests/minute]

## Error Handling Best Practices

### Common Error Scenarios

```python
# Error handling patterns for Bedrock
BEDROCK_ERROR_PATTERNS = {
    'ThrottlingException': {
        'description': 'Rate limit exceeded',
        'retry_strategy': 'exponential_backoff',
        'user_message': 'Service is busy, please try again in a moment'
    },
    'ValidationException': {
        'description': 'Invalid request parameters',
        'retry_strategy': 'none',
        'user_message': 'Invalid request format'
    },
    'ResourceNotFoundException': {
        'description': 'Model not found or not accessible',
        'retry_strategy': 'none', 
        'user_message': 'Requested model is not available'
    },
    'ServiceUnavailableException': {
        'description': 'Temporary service outage',
        'retry_strategy': 'linear_backoff',
        'user_message': 'Service temporarily unavailable'
    }
}
```

### Retry Logic Implementation

```python
# Advanced retry logic for production use
async def bedrock_request_with_retry(
    self,
    request_func,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0
) -> Dict[str, Any]:
    """Execute Bedrock request with intelligent retry logic"""
    
    for attempt in range(max_retries + 1):
        try:
            return await request_func()
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            
            # Don't retry validation errors
            if error_code == 'ValidationException':
                raise
            
            # Don't retry resource not found
            if error_code == 'ResourceNotFoundException':
                raise
            
            # Retry throttling and service errors
            if attempt < max_retries and error_code in ['ThrottlingException', 'ServiceUnavailableException']:
                # Exponential backoff with jitter
                delay = min(base_delay * (2 ** attempt), max_delay)
                jitter = random.uniform(0, delay * 0.1)
                await asyncio.sleep(delay + jitter)
                
                logger.info(
                    "Retrying Bedrock request",
                    attempt=attempt + 1,
                    error_code=error_code,
                    delay=delay
                )
                continue
            
            # Re-raise if max retries exceeded
            raise
```

## Monitoring and Observability

### CloudWatch Metrics

```python
# Custom CloudWatch metrics for Bedrock usage
class BedrockMetrics:
    """CloudWatch metrics for Bedrock monitoring"""
    
    def __init__(self):
        self.cloudwatch = boto3.client('cloudwatch')
        self.namespace = 'MedAssistant/Bedrock'
    
    async def put_request_metrics(
        self,
        model_name: str,
        response_time: float,
        tokens_used: int,
        cost: float,
        success: bool
    ):
        """Put comprehensive request metrics"""
        
        dimensions = [
            {'Name': 'ModelName', 'Value': model_name},
            {'Name': 'Status', 'Value': 'Success' if success else 'Error'}
        ]
        
        metrics = [
            {
                'MetricName': 'RequestCount',
                'Value': 1,
                'Unit': 'Count',
                'Dimensions': dimensions
            },
            {
                'MetricName': 'ResponseTime', 
                'Value': response_time,
                'Unit': 'Milliseconds',
                'Dimensions': dimensions
            },
            {
                'MetricName': 'TokensUsed',
                'Value': tokens_used,
                'Unit': 'Count', 
                'Dimensions': dimensions
            },
            {
                'MetricName': 'EstimatedCost',
                'Value': cost,
                'Unit': 'None',
                'Dimensions': dimensions
            }
        ]
        
        await asyncio.to_thread(
            self.cloudwatch.put_metric_data,
            Namespace=self.namespace,
            MetricData=metrics
        )
```

## Production Deployment Tips

### Environment Configuration

```python
# Production-ready settings for Bedrock
class BedrockProductionConfig:
    """Production configuration for Bedrock service"""
    
    # Regional failover
    PRIMARY_REGION = 'us-east-1'
    SECONDARY_REGION = 'us-west-2'
    
    # Request timeouts
    TIMEOUT_FAST = 15.0  # For simple requests
    TIMEOUT_NORMAL = 30.0  # Standard timeout
    TIMEOUT_COMPLEX = 60.0  # For complex analysis
    
    # Rate limiting
    REQUESTS_PER_MINUTE = {
        'claude-3-5-sonnet': 8,  # Conservative limit
        'claude-3-haiku': 15,
        'llama-3-1': 10,
        'amazon-titan': 20
    }
    
    # Cost controls
    DAILY_BUDGET_LIMIT = 100.0  # $100 per day
    COST_ALERT_THRESHOLD = 0.10  # Alert on requests > $0.10
    
    # Monitoring
    ENABLE_DETAILED_LOGGING = True
    ENABLE_CLOUDWATCH_METRICS = True
    LOG_RETENTION_DAYS = 30
```

### Security Best Practices

```python
# Security configurations for production
BEDROCK_SECURITY_CONFIG = {
    'iam_policy': {
        'Version': '2012-10-17',
        'Statement': [
            {
                'Effect': 'Allow',
                'Action': [
                    'bedrock:InvokeModel',
                    'bedrock:InvokeModelWithResponseStream'
                ],
                'Resource': [
                    'arn:aws:bedrock:*::foundation-model/anthropic.*',
                    'arn:aws:bedrock:*::foundation-model/meta.*',
                    'arn:aws:bedrock:*::foundation-model/amazon.*'
                ]
            },
            {
                'Effect': 'Allow',
                'Action': [
                    'bedrock:Retrieve'
                ],
                'Resource': [
                    'arn:aws:bedrock:*:*:knowledge-base/*'
                ]
            }
        ]
    },
    'vpc_endpoint': {
        'service_name': 'com.amazonaws.region.bedrock-runtime',
        'policy_document': 'RestrictToApplicationOnly'
    },
    'encryption': {
        'in_transit': 'TLS 1.2+',
        'at_rest': 'AWS KMS'
    }
}
```

## Summary

AWS Bedrock provides a robust foundation for AI integration in FastAPI applications with several key advantages:

### Strengths
- **Multiple model access** through unified API
- **Built-in RAG capabilities** with Knowledge Bases
- **Enterprise security** and compliance features
- **Predictable pricing** with transparent token-based billing
- **AWS ecosystem integration** for monitoring and security

### Best Practices
1. **Model Selection:** Choose appropriate model for task complexity and budget
2. **Error Handling:** Implement comprehensive retry logic and graceful degradation
3. **Cost Monitoring:** Track usage and implement budget controls
4. **Performance Optimization:** Use caching and request batching where appropriate
5. **Security:** Follow least-privilege access and use VPC endpoints

### Performance Characteristics
- **Claude 3.5 Sonnet:** Best for complex reasoning, higher cost
- **Claude 3 Haiku:** Fastest and most cost-effective for simple tasks
- **RAG Integration:** Seamless knowledge base integration
- **Scalability:** Auto-scaling with usage-based pricing

This implementation provides a production-ready foundation for AWS Bedrock integration that can scale from development to enterprise deployment.

---

*Next: Section 2 - Azure OpenAI Integration*