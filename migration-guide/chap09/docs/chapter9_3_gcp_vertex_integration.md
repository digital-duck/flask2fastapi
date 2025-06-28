        # Enhanced system instruction for RAG
        rag_system_instruction = system_instruction or ""
        if rag_system_instruction:
            rag_system_instruction += "\n\n"
        
        rag_system_instruction += """You are an AI assistant that answers questions based on search results and summaries.
        Guidelines:
        1. Use the provided search summary and results to inform your answer
        2. Cite specific sources when making claims
        3. If information is conflicting, acknowledge this and explain the differences
        4. If the search results don't contain sufficient information, clearly state this
        5. Prioritize more recent and higher-scored sources
        6. Provide comprehensive answers with relevant context"""
        
        enhanced_prompt = f"""Based on the following search information, please answer the question:

{context_text}

Question: {query}

Please provide a detailed answer based on the search information provided."""
        
        # Generate response with context
        result = await self.vertex_service.generate_response(
            prompt=enhanced_prompt,
            model_name=model_name,
            system_instruction=rag_system_instruction,
            temperature=temperature
        )
        
        # Add RAG metadata
        result['rag_used'] = True
        result['rag_context'] = search_results['contexts']
        result['rag_query'] = query
        result['search_type'] = search_type
        result['search_summary'] = search_results.get('summary')
        result['context_length'] = len(context_text)
        result['contexts_used'] = len(search_results['contexts'])
        
        # Include source information if requested
        if include_sources:
            result['sources'] = [
                {
                    'title': ctx['title'],
                    'uri': ctx['uri'],
                    'score': ctx['score'],
                    'document_id': ctx['document_id'],
                    'metadata': ctx['metadata']
                }
                for ctx in search_results['contexts']
            ]
        
        logger.info(
            "Vertex RAG generation completed",
            model=model_name,
            search_type=search_type,
            contexts_used=len(search_results['contexts']),
            has_summary=bool(search_results.get('summary')),
            context_length=len(context_text)
        )
        
        return result
    
    async def get_search_analytics(self, days: int = 7) -> Dict[str, Any]:
        """Get search analytics for the past N days"""
        try:
            # This would use the Analytics API when available
            # For now, return placeholder data
            return {
                'time_period': f'last_{days}_days',
                'total_queries': '[PLACEHOLDER]',
                'avg_results_per_query': '[PLACEHOLDER]',
                'top_queries': '[PLACEHOLDER]',
                'search_quality_metrics': {
                    'click_through_rate': '[PLACEHOLDER]',
                    'avg_result_relevance': '[PLACEHOLDER]'
                }
            }
        except Exception as e:
            logger.error("Search analytics failed", error=str(e))
            return {'error': str(e)}

# Global instance for dependency injection
vertex_rag_service = VertexRAGService()

async def get_vertex_rag_service() -> VertexRAGService:
    """FastAPI dependency for Vertex RAG service"""
    return vertex_rag_service
```

## FastAPI Routes

### Vertex AI Chat Routes

```python
# routers/vertex_chat.py
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, File, UploadFile
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Literal, Union
import structlog
import time
import asyncio

from services.gcp_vertex_service import (
    VertexAIService, 
    get_vertex_ai_service
)
from services.vertex_rag_service import (
    VertexRAGService,
    get_vertex_rag_service
)

logger = structlog.get_logger()
router = APIRouter(prefix="/vertex", tags=["Google Cloud Vertex AI"])

# Request Models
class VertexChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=8000, description="User message")
    model_name: Literal[
        'gemini-1.0-pro', 
        'gemini-1.0-pro-vision', 
        'text-bison@001', 
        'chat-bison@001', 
        'code-bison@001'
    ] = Field(default="gemini-1.0-pro", description="Vertex AI model to use")
    system_instruction: Optional[str] = Field(None, max_length=2000, description="System instruction")
    max_tokens: Optional[int] = Field(default=None, ge=1, le=8192, description="Max output tokens")
    temperature: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Temperature")
    top_p: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Top-p sampling")
    top_k: Optional[int] = Field(default=None, ge=1, le=100, description="Top-k sampling")
    safety_settings: Optional[Dict[str, str]] = Field(None, description="Safety filter settings")

class VertexRAGRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=8000)
    model_name: Literal['gemini-1.0-pro', 'text-bison@001', 'chat-bison@001'] = Field(default="gemini-1.0-pro")
    system_instruction: Optional[str] = Field(None, max_length=2000)
    search_type: Literal['unstructured', 'structured', 'website'] = Field(default='unstructured')
    max_results: int = Field(default=5, ge=1, le=20, description="Number of search results")
    max_context_length: int = Field(default=4000, ge=500, le=8000)
    include_sources: bool = Field(default=True)
    use_search_summary: bool = Field(default=True)
    temperature: float = Field(default=0.7, ge=0.0, le=1.0)

class VertexVisionRequest(BaseModel):
    message: str = Field(..., min_length=1)
    image_url: Optional[str] = Field(None, description="URL of the image to analyze")
    model_name: Literal['gemini-1.0-pro-vision'] = Field(default="gemini-1.0-pro-vision")
    max_tokens: Optional[int] = Field(default=2048, ge=1, le=4096)
    temperature: Optional[float] = Field(default=0.4, ge=0.0, le=1.0)

class EmbeddingRequest(BaseModel):
    texts: List[str] = Field(..., min_items=1, max_items=100, description="Texts to embed")
    model_name: str = Field(default='textembedding-gecko@001')
    task_type: Literal[
        'RETRIEVAL_DOCUMENT',
        'RETRIEVAL_QUERY', 
        'SEMANTIC_SIMILARITY',
        'CLASSIFICATION',
        'CLUSTERING'
    ] = Field(default='RETRIEVAL_DOCUMENT')

# Response Models
class VertexChatResponse(BaseModel):
    response: str
    model_used: str
    model_type: str
    tokens_used: Dict[str, int]
    processing_time_ms: float
    estimated_cost: float
    finish_reason: str
    safety_ratings: List[Dict] = []

class VertexRAGResponse(BaseModel):
    response: str
    model_used: str
    tokens_used: Dict[str, int]
    processing_time_ms: float
    estimated_cost: float
    rag_used: bool
    search_type: str
    contexts_used: int
    context_length: Optional[int] = None
    search_summary: Optional[Dict] = None
    sources: Optional[List[Dict]] = None

class VertexVisionResponse(BaseModel):
    analysis: str
    model_used: str
    tokens_used: Dict[str, int]
    processing_time_ms: float
    estimated_cost: float
    multimodal_input: Dict
    safety_ratings: List[Dict] = []

class EmbeddingResponse(BaseModel):
    embeddings: List[List[float]]
    model: str
    task_type: str
    usage: Dict[str, int]
    estimated_cost: float

@router.post("/chat", response_model=VertexChatResponse)
async def chat_with_vertex_ai(
    request: VertexChatRequest,
    background_tasks: BackgroundTasks,
    vertex_ai: VertexAIService = Depends(get_vertex_ai_service)
):
    """Chat endpoint using Google Cloud Vertex AI models"""
    start_time = time.time()
    
    try:
        result = await vertex_ai.generate_response(
            prompt=request.message,
            model_name=request.model_name,
            system_instruction=request.system_instruction,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            safety_settings=request.safety_settings
        )
        
        processing_time = (time.time() - start_time) * 1000
        
        # Log usage for analytics
        background_tasks.add_task(
            log_vertex_usage,
            model_name=request.model_name,
            model_type=result.get('model_type', 'unknown'),
            tokens=result.get('usage', {}),
            processing_time=processing_time,
            cost=result.get('estimated_cost', 0),
            rag_used=False,
            safety_ratings=result.get('safety_ratings', [])
        )
        
        return VertexChatResponse(
            response=result['text'],
            model_used=request.model_name,
            model_type=result.get('model_type', 'unknown'),
            tokens_used=result.get('usage', {}),
            processing_time_ms=processing_time,
            estimated_cost=result.get('estimated_cost', 0),
            finish_reason=result.get('finish_reason', 'COMPLETED'),
            safety_ratings=result.get('safety_ratings', [])
        )
        
    except ValueError as e:
        logger.warning("Vertex AI validation error", error=str(e))
        raise HTTPException(status_code=400, detail=str(e))
        
    except Exception as e:
        logger.error("Vertex AI chat failed", error=str(e))
        raise HTTPException(
            status_code=500,
            detail="Vertex AI service temporarily unavailable"
        )

@router.post("/chat/rag", response_model=VertexRAGResponse)
async def chat_with_vertex_rag(
    request: VertexRAGRequest,
    background_tasks: BackgroundTasks,
    vertex_rag: VertexRAGService = Depends(get_vertex_rag_service)
):
    """Chat endpoint using Vertex AI with RAG (Vertex AI Search)"""
    start_time = time.time()
    
    try:
        result = await vertex_rag.generate_with_rag(
            query=request.message,
            model_name=request.model_name,
            system_instruction=request.system_instruction,
            search_type=request.search_type,
            max_results=request.max_results,
            max_context_length=request.max_context_length,
            include_sources=request.include_sources,
            use_search_summary=request.use_search_summary,
            temperature=request.temperature
        )
        
        processing_time = (time.time() - start_time) * 1000
        
        # Log usage for analytics
        background_tasks.add_task(
            log_vertex_usage,
            model_name=request.model_name,
            model_type='rag',
            tokens=result.get('usage', {}),
            processing_time=processing_time,
            cost=result.get('estimated_cost', 0),
            rag_used=result.get('rag_used', False),
            search_type=request.search_type
        )
        
        return VertexRAGResponse(
            response=result['text'],
            model_used=request.model_name,
            tokens_used=result.get('usage', {}),
            processing_time_ms=processing_time,
            estimated_cost=result.get('estimated_cost', 0),
            rag_used=result.get('rag_used', False),
            search_type=request.search_type,
            contexts_used=result.get('contexts_used', 0),
            context_length=result.get('context_length'),
            search_summary=result.get('search_summary'),
            sources=result.get('sources')
        )
        
    except ValueError as e:
        logger.warning("Vertex RAG validation error", error=str(e))
        raise HTTPException(status_code=400, detail=str(e))
        
    except Exception as e:
        logger.error("Vertex RAG failed", error=str(e))
        raise HTTPException(
            status_code=500,
            detail="Vertex RAG service temporarily unavailable"
        )

@router.post("/vision", response_model=VertexVisionResponse)
async def analyze_with_vertex_vision(
    request: VertexVisionRequest,
    background_tasks: BackgroundTasks,
    vertex_ai: VertexAIService = Depends(get_vertex_ai_service)
):
    """Vision analysis endpoint using Vertex AI Gemini Vision"""
    start_time = time.time()
    
    try:
        result = await vertex_ai.generate_with_vision(
            prompt=request.message,
            image_url=request.image_url,
            model_name=request.model_name,
            max_tokens=request.max_tokens,
            temperature=request.temperature
        )
        
        processing_time = (time.time() - start_time) * 1000
        
        # Log vision usage
        background_tasks.add_task(
            log_vertex_vision_usage,
            model_name=request.model_name,
            tokens=result.get('usage', {}),
            processing_time=processing_time,
            cost=result.get('estimated_cost', 0),
            multimodal_input=result.get('multimodal_input', {})
        )
        
        return VertexVisionResponse(
            analysis=result['text'],
            model_used=request.model_name,
            tokens_used=result.get('usage', {}),
            processing_time_ms=processing_time,
            estimated_cost=result.get('estimated_cost', 0),
            multimodal_input=result.get('multimodal_input', {}),
            safety_ratings=result.get('safety_ratings', [])
        )
        
    except Exception as e:
        logger.error("Vertex AI vision failed", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Vision analysis failed: {str(e)}"
        )

@router.post("/vision/upload", response_model=VertexVisionResponse)
async def analyze_uploaded_media(
    message: str,
    file: UploadFile = File(...),
    model_name: Literal['gemini-1.0-pro-vision'] = 'gemini-1.0-pro-vision',
    max_tokens: int = 2048,
    temperature: float = 0.4,
    background_tasks: BackgroundTasks = BackgroundTasks(),
    vertex_ai: VertexAIService = Depends(get_vertex_ai_service)
):
    """Upload and analyze image/video using Vertex AI Gemini Vision"""
    start_time = time.time()
    
    try:
        # Validate file type
        if not (file.content_type.startswith('image/') or file.content_type.startswith('video/')):
            raise HTTPException(status_code=400, detail="File must be an image or video")
        
        # Read file data
        file_data = await file.read()
        
        # Limit file size (e.g., 20MB for video, 10MB for images)
        max_size = 20 * 1024 * 1024 if file.content_type.startswith('video/') else 10 * 1024 * 1024
        if len(file_data) > max_size:
            raise HTTPException(
                status_code=400, 
                detail=f"File too large (max {max_size // (1024*1024)}MB)"
            )
        
        # Analyze based on file type
        if file.content_type.startswith('image/'):
            result = await vertex_ai.generate_with_vision(
                prompt=message,
                image_data=file_data,
                model_name=model_name,
                max_tokens=max_tokens,
                temperature=temperature
            )
        else:  # video
            result = await vertex_ai.generate_with_vision(
                prompt=message,
                video_data=file_data,
                model_name=model_name,
                max_tokens=max_tokens,
                temperature=temperature
            )
        
        processing_time = (time.time() - start_time) * 1000
        
        # Log vision usage
        background_tasks.add_task(
            log_vertex_vision_usage,
            model_name=model_name,
            tokens=result.get('usage', {}),
            processing_time=processing_time,
            cost=result.get('estimated_cost', 0),
            multimodal_input=result.get('multimodal_input', {})
        )
        
        return VertexVisionResponse(
            analysis=result['text'],
            model_used=model_name,
            tokens_used=result.get('usage', {}),
            processing_time_ms=processing_time,
            estimated_cost=result.get('estimated_cost', 0),
            multimodal_input=result.get('multimodal_input', {}),
            safety_ratings=result.get('safety_ratings', [])
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Media upload analysis failed", error=str(e))
        raise HTTPException(
            status_code=500,
            detail="Media analysis failed"
        )

@router.post("/embeddings", response_model=EmbeddingResponse)
async def create_embeddings(
    request: EmbeddingRequest,
    vertex_ai: VertexAIService = Depends(get_vertex_ai_service)
):
    """Create embeddings using Vertex AI Text Embedding models"""
    try:
        # Validate input
        if any(len(text) > 5000 for text in request.texts):
            raise HTTPException(status_code=400, detail="Text too long (max 5000 characters)")
        
        result = await vertex_ai.create_embeddings(
            texts=request.texts,
            model_name=request.model_name,
            task_type=request.task_type
        )
        
        return EmbeddingResponse(
            embeddings=result['embeddings'],
            model=result['model'],
            task_type=result['task_type'],
            usage=result['usage'],
            estimated_cost=result['estimated_cost']
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Embeddings creation failed", error=str(e))
        raise HTTPException(
            status_code=500,
            detail="Embeddings service unavailable"
        )

@router.get("/models")
async def list_available_models(
    vertex_ai: VertexAIService = Depends(get_vertex_ai_service)
):
    """List available Vertex AI models with capabilities and pricing"""
    models_info = {}
    
    for model_name, config in vertex_ai.model_configs.items():
        models_info[model_name] = {
            'model_name': config['model_name'],
            'type': config['type'],
            'max_output_tokens': config['max_output_tokens'],
            'multimodal': config['multimodal'],
            'pricing_per_1k_tokens': config['pricing'],
            'capabilities': _get_model_capabilities(model_name),
            'best_for': _get_model_use_cases(model_name)
        }
    
    return {
        "available_models": models_info,
        "recommendation_guide": {
            "latest_capabilities": "gemini-1.0-pro",
            "multimodal": "gemini-1.0-pro-vision",
            "cost_effective": "text-bison@001",
            "code_generation": "code-bison@001",
            "chat_applications": "chat-bison@001",
            "embeddings": "textembedding-gecko@001"
        }
    }

@router.get("/search/analytics")
async def get_search_analytics(
    days: int = Field(default=7, ge=1, le=30),
    vertex_rag: VertexRAGService = Depends(get_vertex_rag_service)
):
    """Get Vertex AI Search analytics"""
    try:
        analytics = await vertex_rag.get_search_analytics(days)
        return analytics
    except Exception as e:
        logger.error("Search analytics failed", error=str(e))
        return {"error": "Analytics temporarily unavailable"}

@router.get("/health")
async def vertex_ai_health_check(
    vertex_ai: VertexAIService = Depends(get_vertex_ai_service)
):
    """Check Vertex AI service health"""
    return await vertex_ai.health_check()

def _get_model_capabilities(model_name: str) -> List[str]:
    """Get capabilities for each model"""
    capabilities = {
        'gemini-1.0-pro': [
            'Advanced text generation',
            'Complex reasoning',
            'Code understanding',
            'Multilingual support',
            'Safety filtering'
        ],
        'gemini-1.0-pro-vision': [
            'Multimodal understanding',
            'Image analysis',
            'Video analysis',
            'Text generation',
            'Visual Q&A'
        ],
        'text-bison@001': [
            'Text generation',
            'Content creation',
            'Summarization',
            'Translation'
        ],
        'chat-bison@001': [
            'Conversational AI',
            'Multi-turn dialogue',
            'Context awareness',
            'Chat applications'
        ],
        'code-bison@001': [
            'Code generation',
            'Code completion',
            'Code explanation',
            'Debugging assistance'
        ]
    }
    return capabilities.get(model_name, ['Text generation'])

def _get_model_use_cases(model_name: str) -> List[str]:
    """Get recommended use cases for each model"""
    use_cases = {
        'gemini-1.0-pro': [
            'Complex analysis and reasoning',
            'Content creation and editing',
            'Code generation and review',
            'Research and synthesis'
        ],
        'gemini-1.0-pro-vision': [
            'Image and video analysis',
            'Visual content understanding',
            'Multimodal applications',
            'Document processing with images'
        ],
        'text-bison@001': [
            'Article writing and editing',
            'Email and document generation',
            'Content summarization',
            'Language translation'
        ],
        'chat-bison@001': [
            'Customer service chatbots',
            'Virtual assistants',
            'Interactive Q&A systems',
            'Conversational interfaces'
        ],
        'code-bison@001': [
            'Code completion tools',
            'Programming tutors',
            'Code review assistance',
            'Documentation generation'
        ]
    }
    return use_cases.get(model_name, ['General text generation'])

# Background task functions
async def log_vertex_usage(
    model_name: str,
    model_type: str,
    tokens: Dict,
    processing_time: float,
    cost: float,
    rag_used: bool,
    safety_ratings: Optional[List] = None,
    search_type: Optional[str] = None
):
    """Background task to log Vertex AI usage"""
    logger.info(
        "Vertex AI usage logged",
        model=model_name,
        model_type=model_type,
        prompt_tokens=tokens.get('prompt_tokens', 0),
        completion_tokens=tokens.get('completion_tokens', 0),
        total_tokens=tokens.get('total_tokens', 0),
        processing_time_ms=processing_time,
        estimated_cost=cost,
        rag_enabled=rag_used,
        safety_issues=len([r for r in (safety_ratings or []) if r.get('blocked', False)]),
        search_type=search_type
    )

async def log_vertex_vision_usage(
    model_name: str,
    tokens: Dict,
    processing_time: float,
    cost: float,
    multimodal_input: Dict
):
    """Background task to log Vertex AI vision usage"""
    logger.info(
        "Vertex AI Vision usage logged",
        model=model_name,
        tokens_used=tokens.get('total_tokens', 0),
        processing_time_ms=processing_time,
        estimated_cost=cost,
        input_types=multimodal_input.get('input_types', []),
        has_image=multimodal_input.get('has_image', False),
        has_video=multimodal_input.get('has_video', False)
    )
```

## Performance Characteristics

### Benchmark Results

**Response Time Analysis (Based on Testing):**

| Model | Avg Response Time | P95 Response Time | Cold Start | Multimodal Capability |
|-------|------------------|-------------------|------------|----------------------|
| Gemini 1.0 Pro | [PLACEHOLDER: 1.4s] | [PLACEHOLDER: 2.9s] | [PLACEHOLDER: 0.4s] | Text only |
| Gemini Pro Vision | [PLACEHOLDER: 2.8s] | [PLACEHOLDER: 5.2s] | [PLACEHOLDER: 0.6s] | Text + Image + Video |
| Text Bison | [PLACEHOLDER: 1.1s] | [PLACEHOLDER: 2.2s] | [PLACEHOLDER: 0.3s] | Text only |
| Chat Bison | [PLACEHOLDER: 1.0s] | [PLACEHOLDER: 2.1s] | [PLACEHOLDER: 0.3s] | Text only |
| Code Bison | [PLACEHOLDER: 1.3s] | [PLACEHOLDER: 2.7s] | [PLACEHOLDER: 0.4s] | Code + Text |

### Cost Analysis

**Real-World Cost Examples:**

```python
# Vertex AI cost calculations for different scenarios
VERTEX_COST_EXAMPLES = {
    'simple_chat': {
        'input_tokens': 100,
        'output_tokens': 150,
        'gemini_pro': (100/1000 * 0.00025) + (150/1000 * 0.0005) = 0.000175,  # $0.000175
        'text_bison': (100/1000 * 0.0005) + (150/1000 * 0.0005) = 0.000125,  # $0.000125
        'note': 'Very cost-effective compared to other providers'
    },
    'code_generation': {
        'input_tokens': 500,
        'output_tokens': 800,
        'gemini_pro': (500/1000 * 0.00025) + (800/1000 * 0.0005) = 0.000525,  # $0.000525
        'code_bison': (500/1000 * 0.0005) + (800/1000 * 0.0005) = 0.00065,  # $0.00065
        'recommendation': 'Gemini Pro offers better capabilities at lower cost'
    },
    'multimodal_analysis': {
        'image_tokens': 800,  # Estimated for image processing
        'text_tokens': 200,
        'output_tokens': 300,
        'gemini_pro_vision': ((800 + 200)/1000 * 0.00025) + (300/1000 * 0.0005) = 0.0004,  # $0.0004
        'note': 'Excellent value for multimodal capabilities'
    }
}
```

### Rate Limits and Quotas

**Current Limits (Varies by Project and Region):**

```python
VERTEX_RATE_LIMITS = {
    'default_quotas': {
        'gemini_pro': {
            'requests_per_minute': 60,
            'tokens_per_minute': 240000
        },
        'gemini_pro_vision': {
            'requests_per_minute': 60,
            'tokens_per_minute': 240000
        },
        'text_bison': {
            'requests_per_minute': 60,
            'tokens_per_minute': 240000
        }
    },
    'enterprise_quotas': {
        'note': 'Higher limits available through quota increase requests',
        'process': 'Submit quota increase request in Google Cloud Console'
    }
}
```

## Advanced Features

### Safety and Content Filtering

```python
# Advanced safety configuration for Vertex AI
VERTEX_SAFETY_CONFIG = {
    'harm_categories': {
        'HARM_CATEGORY_HATE_SPEECH': {
            'threshold': 'BLOCK_MEDIUM_AND_ABOVE',
            'description': 'Hate speech and discrimination'
        },
        'HARM_CATEGORY_DANGEROUS_CONTENT': {
            'threshold': 'BLOCK_MEDIUM_AND_ABOVE',
            'description': 'Dangerous or harmful content'
        },
        'HARM_CATEGORY_SEXUAL': {
            'threshold': 'BLOCK_MEDIUM_AND_ABOVE',
            'description': 'Sexual content'
        },
        'HARM_CATEGORY_HARASSMENT': {
            'threshold': 'BLOCK_MEDIUM_AND_ABOVE',
            'description': 'Harassment and bullying'
        }
    },
    'custom_safety_settings': {
        'enabled': True,
        'description': 'Custom safety filters for specific use cases'
    }
}
```

### MLOps Integration

```python
# MLOps features for model management
class VertexMLOpsService:
    """Integration with Vertex AI MLOps features"""
    
    async def create_model_endpoint(
        self,
        model_name: str,
        endpoint_name: str,
        machine_type: str = 'n1-standard-4',
        min_replica_count: int = 1,
        max_replica_count: int = 10
    ) -> Dict[str, Any]:
        """Create a model endpoint for custom models"""
        
        # This would use the Vertex AI Model Registry and Endpoint APIs
        endpoint_config = {
            'display_name': endpoint_name,
            'model_name': model_name,
            'machine_type': machine_type,
            'min_replica_count': min_replica_count,
            'max_replica_count': max_replica_count,
            'auto_scaling': True
        }
        
        logger.info("Model endpoint configuration", config=endpoint_config)
        
        return {
            'endpoint_id': f'endpoint-{endpoint_name}-123',
            'status': 'creating',
            'config': endpoint_config
        }
    
    async def monitor_model_performance(
        self,
        endpoint_id: str,
        metrics: List[str] = None
    ) -># Chapter 9, Section 3: Google Cloud Vertex AI Integration

## Overview

Google Cloud Vertex AI provides access to Google's latest foundation models including Gemini Pro, PaLM 2, Codey, and specialized models for various tasks. It offers unique advantages including strong multimodal capabilities, competitive pricing, advanced MLOps features, and tight integration with Google's ecosystem including Search, Knowledge Graph, and Cloud services.

## Key Advantages

- **Google's Latest Models:** Access to Gemini Pro, PaLM 2, Codey with cutting-edge capabilities
- **Strong Multimodal Support:** Native text, image, video, and audio processing
- **Competitive Pricing:** Often more cost-effective than competitors
- **MLOps Integration:** Comprehensive model management, training, and deployment
- **Google Ecosystem:** Integration with Google Search, Knowledge Graph, and Workspace
- **Advanced Safety:** Built-in safety filters and responsible AI features

## Model Landscape

### Available Models

| Model Family | Model Name | Strengths | Use Cases |
|--------------|------------|-----------|-----------|
| **Gemini Pro** | `gemini-1.0-pro` | Latest capabilities, multimodal | Complex reasoning, code, analysis |
| **Gemini Pro Vision** | `gemini-1.0-pro-vision` | Advanced vision understanding | Image analysis, visual Q&A |
| **PaLM 2** | `text-bison@001` | Reliable text generation | General purpose, chat |
| **Codey** | `code-bison@001` | Code generation and analysis | Programming, debugging |
| **Text Embedding** | `textembedding-gecko@001` | High-quality embeddings | RAG, similarity search |
| **Chat Bison** | `chat-bison@001` | Conversational AI | Chatbots, dialogue systems |

### Model Selection Guide

```python
VERTEX_MODEL_SELECTION_GUIDE = {
    'latest_capabilities': 'gemini-1.0-pro',
    'multimodal_analysis': 'gemini-1.0-pro-vision',
    'code_generation': 'code-bison@001',
    'cost_optimization': 'text-bison@001',
    'embeddings': 'textembedding-gecko@001',
    'chat_applications': 'chat-bison@001',
    'general_purpose': 'gemini-1.0-pro',
    'image_understanding': 'gemini-1.0-pro-vision'
}
```

## FastAPI Service Implementation

### Core Vertex AI Service

```python
# services/gcp_vertex_service.py
import asyncio
from typing import Dict, Any, Optional, List, Union
import structlog
from google.cloud import aiplatform
from google.auth import default
import json
import base64
from vertexai.generative_models import GenerativeModel, GenerationConfig, Part, Image
from vertexai.language_models import TextGenerationModel, ChatModel, TextEmbeddingModel
import vertexai

from config.settings import settings

logger = structlog.get_logger()

class VertexAIService:
    """Google Cloud Vertex AI service integration with comprehensive model support"""
    
    def __init__(self):
        # Initialize Vertex AI
        vertexai.init(
            project=settings.gcp_project_id,
            location=settings.gcp_region
        )
        
        self.model_configs = {
            'gemini-1.0-pro': {
                'model_name': 'gemini-1.0-pro',
                'type': 'generative',
                'max_output_tokens': 8192,
                'temperature': 0.7,
                'top_p': 0.8,
                'top_k': 40,
                'multimodal': False,
                'pricing': {'input': 0.00025, 'output': 0.0005}  # per 1K tokens
            },
            'gemini-1.0-pro-vision': {
                'model_name': 'gemini-1.0-pro-vision',
                'type': 'generative',
                'max_output_tokens': 4096,
                'temperature': 0.4,
                'top_p': 0.8,
                'top_k': 32,
                'multimodal': True,
                'pricing': {'input': 0.00025, 'output': 0.0005}
            },
            'text-bison@001': {
                'model_name': 'text-bison@001',
                'type': 'language',
                'max_output_tokens': 1024,
                'temperature': 0.7,
                'top_p': 0.8,
                'top_k': 40,
                'multimodal': False,
                'pricing': {'input': 0.0005, 'output': 0.0005}
            },
            'chat-bison@001': {
                'model_name': 'chat-bison@001',
                'type': 'chat',
                'max_output_tokens': 1024,
                'temperature': 0.7,
                'top_p': 0.8,
                'top_k': 40,
                'multimodal': False,
                'pricing': {'input': 0.0005, 'output': 0.0005}
            },
            'code-bison@001': {
                'model_name': 'code-bison@001',
                'type': 'language',
                'max_output_tokens': 1024,
                'temperature': 0.2,  # Lower temperature for code
                'top_p': 0.8,
                'top_k': 40,
                'multimodal': False,
                'pricing': {'input': 0.0005, 'output': 0.0005}
            }
        }
        
        # Safety settings
        self.safety_settings = {
            'HARM_CATEGORY_HATE_SPEECH': 'BLOCK_MEDIUM_AND_ABOVE',
            'HARM_CATEGORY_DANGEROUS_CONTENT': 'BLOCK_MEDIUM_AND_ABOVE',
            'HARM_CATEGORY_SEXUAL': 'BLOCK_MEDIUM_AND_ABOVE',
            'HARM_CATEGORY_HARASSMENT': 'BLOCK_MEDIUM_AND_ABOVE'
        }
    
    async def generate_response(
        self,
        prompt: str,
        model_name: str = 'gemini-1.0-pro',
        system_instruction: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        safety_settings: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Generate response using Vertex AI models with comprehensive configuration"""
        
        if model_name not in self.model_configs:
            available_models = list(self.model_configs.keys())
            raise ValueError(f"Unsupported model: {model_name}. Available: {available_models}")
        
        config = self.model_configs[model_name]
        
        try:
            if config['type'] == 'generative':
                return await self._generate_with_gemini(
                    prompt, config, system_instruction, max_tokens, 
                    temperature, top_p, top_k, safety_settings
                )
            elif config['type'] == 'language':
                return await self._generate_with_language_model(
                    prompt, config, max_tokens, temperature, top_p, top_k
                )
            elif config['type'] == 'chat':
                return await self._generate_with_chat_model(
                    prompt, config, max_tokens, temperature, top_p, top_k
                )
            else:
                raise ValueError(f"Unsupported model type: {config['type']}")
                
        except Exception as e:
            logger.error(
                "Vertex AI generation failed", 
                error=str(e), 
                model=model_name,
                model_type=config['type']
            )
            
            # Handle specific Vertex AI errors
            if "quota" in str(e).lower():
                raise Exception("Quota exceeded for Vertex AI. Please check your project limits.")
            elif "permission" in str(e).lower():
                raise Exception("Permission denied. Please check your service account permissions.")
            elif "safety" in str(e).lower():
                raise Exception("Content blocked by safety filters.")
            else:
                raise Exception(f"Vertex AI error: {str(e)}")
    
    async def _generate_with_gemini(
        self,
        prompt: str,
        config: Dict,
        system_instruction: Optional[str],
        max_tokens: Optional[int],
        temperature: Optional[float],
        top_p: Optional[float],
        top_k: Optional[int],
        safety_settings: Optional[Dict]
    ) -> Dict[str, Any]:
        """Generate response using Gemini models"""
        
        # Configure generation parameters
        generation_config = GenerationConfig(
            max_output_tokens=max_tokens or config['max_output_tokens'],
            temperature=temperature or config['temperature'],
            top_p=top_p or config['top_p'],
            top_k=top_k or config['top_k']
        )
        
        # Configure safety settings
        safety_config = safety_settings or self.safety_settings
        
        # Initialize model
        model = GenerativeModel(
            config['model_name'],
            system_instruction=system_instruction,
            generation_config=generation_config,
            safety_settings=safety_config
        )
        
        # Generate response (run in thread to avoid blocking)
        response = await asyncio.to_thread(
            model.generate_content,
            prompt,
            generation_config=generation_config
        )
        
        # Parse response
        generated_text = response.text if response.text else ""
        
        # Extract usage and safety information
        usage_metadata = response.usage_metadata if hasattr(response, 'usage_metadata') else None
        
        result = {
            'text': generated_text,
            'model': config['model_name'],
            'model_type': 'gemini',
            'usage': {
                'prompt_tokens': usage_metadata.prompt_token_count if usage_metadata else 0,
                'completion_tokens': usage_metadata.candidates_token_count if usage_metadata else 0,
                'total_tokens': usage_metadata.total_token_count if usage_metadata else 0
            },
            'finish_reason': response.candidates[0].finish_reason.name if response.candidates else 'COMPLETED',
            'safety_ratings': [
                {
                    'category': rating.category.name,
                    'probability': rating.probability.name,
                    'blocked': rating.blocked if hasattr(rating, 'blocked') else False
                }
                for rating in (response.candidates[0].safety_ratings if response.candidates else [])
            ],
            'raw_response': response
        }
        
        # Add cost estimation
        result['estimated_cost'] = self._calculate_cost(config['model_name'], result['usage'])
        
        return result
    
    async def _generate_with_language_model(
        self,
        prompt: str,
        config: Dict,
        max_tokens: Optional[int],
        temperature: Optional[float],
        top_p: Optional[float],
        top_k: Optional[int]
    ) -> Dict[str, Any]:
        """Generate response using PaLM language models"""
        
        model = TextGenerationModel.from_pretrained(config['model_name'])
        
        # Generate response
        response = await asyncio.to_thread(
            model.predict,
            prompt,
            max_output_tokens=max_tokens or config['max_output_tokens'],
            temperature=temperature or config['temperature'],
            top_p=top_p or config['top_p'],
            top_k=top_k or config['top_k']
        )
        
        result = {
            'text': response.text,
            'model': config['model_name'],
            'model_type': 'language',
            'usage': {
                'prompt_tokens': 0,  # PaLM doesn't provide detailed token counts
                'completion_tokens': 0,
                'total_tokens': 0
            },
            'safety_attributes': response.safety_attributes.__dict__ if response.safety_attributes else {},
            'raw_response': response
        }
        
        # Estimate tokens for cost calculation
        estimated_input_tokens = len(prompt.split()) * 1.3
        estimated_output_tokens = len(result['text'].split()) * 1.3
        result['usage']['prompt_tokens'] = int(estimated_input_tokens)
        result['usage']['completion_tokens'] = int(estimated_output_tokens)
        result['usage']['total_tokens'] = int(estimated_input_tokens + estimated_output_tokens)
        
        result['estimated_cost'] = self._calculate_cost(config['model_name'], result['usage'])
        
        return result
    
    async def _generate_with_chat_model(
        self,
        prompt: str,
        config: Dict,
        max_tokens: Optional[int],
        temperature: Optional[float],
        top_p: Optional[float],
        top_k: Optional[int]
    ) -> Dict[str, Any]:
        """Generate response using PaLM chat models"""
        
        chat_model = ChatModel.from_pretrained(config['model_name'])
        
        # Start chat session
        chat = chat_model.start_chat()
        
        # Send message
        response = await asyncio.to_thread(
            chat.send_message,
            prompt,
            max_output_tokens=max_tokens or config['max_output_tokens'],
            temperature=temperature or config['temperature'],
            top_p=top_p or config['top_p'],
            top_k=top_k or config['top_k']
        )
        
        result = {
            'text': response.text,
            'model': config['model_name'],
            'model_type': 'chat',
            'usage': {
                'prompt_tokens': 0,
                'completion_tokens': 0,
                'total_tokens': 0
            },
            'chat_history': [
                {'role': 'user', 'content': prompt},
                {'role': 'assistant', 'content': response.text}
            ],
            'raw_response': response
        }
        
        # Estimate tokens
        estimated_input_tokens = len(prompt.split()) * 1.3
        estimated_output_tokens = len(result['text'].split()) * 1.3
        result['usage']['prompt_tokens'] = int(estimated_input_tokens)
        result['usage']['completion_tokens'] = int(estimated_output_tokens)
        result['usage']['total_tokens'] = int(estimated_input_tokens + estimated_output_tokens)
        
        result['estimated_cost'] = self._calculate_cost(config['model_name'], result['usage'])
        
        return result
    
    async def generate_with_vision(
        self,
        prompt: str,
        image_data: Optional[bytes] = None,
        image_url: Optional[str] = None,
        video_data: Optional[bytes] = None,
        model_name: str = 'gemini-1.0-pro-vision',
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None
    ) -> Dict[str, Any]:
        """Generate response with multimodal input using Gemini Vision"""
        
        if model_name not in self.model_configs or not self.model_configs[model_name]['multimodal']:
            raise ValueError("Vision capabilities require a multimodal model like gemini-1.0-pro-vision")
        
        config = self.model_configs[model_name]
        
        # Configure generation
        generation_config = GenerationConfig(
            max_output_tokens=max_tokens or config['max_output_tokens'],
            temperature=temperature or config['temperature']
        )
        
        model = GenerativeModel(config['model_name'])
        
        # Prepare content parts
        content_parts = [prompt]
        
        try:
            # Add image if provided
            if image_data:
                image = Image.from_bytes(image_data)
                content_parts.append(image)
            elif image_url:
                # For URL, we'd need to fetch and convert to bytes
                import httpx
                async with httpx.AsyncClient() as client:
                    img_response = await client.get(image_url)
                    if img_response.status_code == 200:
                        image = Image.from_bytes(img_response.content)
                        content_parts.append(image)
                    else:
                        raise Exception(f"Failed to fetch image from URL: {image_url}")
            
            # Add video if provided (Gemini supports video analysis)
            if video_data:
                video_part = Part.from_data(
                    mime_type="video/mp4",  # Adjust based on actual format
                    data=video_data
                )
                content_parts.append(video_part)
            
            # Generate response
            response = await asyncio.to_thread(
                model.generate_content,
                content_parts,
                generation_config=generation_config
            )
            
            result = await self._parse_gemini_response(response, config['model_name'])
            result['multimodal_input'] = {
                'has_image': bool(image_data or image_url),
                'has_video': bool(video_data),
                'input_types': []
            }
            
            if image_data or image_url:
                result['multimodal_input']['input_types'].append('image')
            if video_data:
                result['multimodal_input']['input_types'].append('video')
            
            return result
            
        except Exception as e:
            logger.error("Vertex AI vision error", error=str(e))
            raise Exception(f"Vision analysis failed: {str(e)}")
    
    async def create_embeddings(
        self,
        texts: List[str],
        model_name: str = 'textembedding-gecko@001',
        task_type: str = 'RETRIEVAL_DOCUMENT'
    ) -> Dict[str, Any]:
        """Create embeddings using Vertex AI Text Embedding models"""
        
        try:
            model = TextEmbeddingModel.from_pretrained(model_name)
            
            # Generate embeddings
            embeddings = await asyncio.to_thread(
                model.get_embeddings,
                texts,
                task_type=task_type
            )
            
            result = {
                'embeddings': [emb.values for emb in embeddings],
                'model': model_name,
                'task_type': task_type,
                'usage': {
                    'prompt_tokens': sum(len(text.split()) for text in texts),
                    'total_tokens': sum(len(text.split()) for text in texts)
                }
            }
            
            # Estimate cost for embeddings (typically much cheaper)
            embedding_cost = (result['usage']['total_tokens'] / 1000) * 0.0001  # Estimated cost
            result['estimated_cost'] = round(embedding_cost, 6)
            
            return result
            
        except Exception as e:
            logger.error("Vertex AI embeddings error", error=str(e))
            raise Exception(f"Embeddings generation failed: {str(e)}")
    
    async def _parse_gemini_response(self, response, model_name: str) -> Dict[str, Any]:
        """Parse Gemini model response with comprehensive metadata"""
        
        generated_text = response.text if response.text else ""
        usage_metadata = response.usage_metadata if hasattr(response, 'usage_metadata') else None
        
        result = {
            'text': generated_text,
            'model': model_name,
            'model_type': 'gemini',
            'usage': {
                'prompt_tokens': usage_metadata.prompt_token_count if usage_metadata else 0,
                'completion_tokens': usage_metadata.candidates_token_count if usage_metadata else 0,
                'total_tokens': usage_metadata.total_token_count if usage_metadata else 0
            },
            'finish_reason': response.candidates[0].finish_reason.name if response.candidates else 'COMPLETED',
            'safety_ratings': [
                {
                    'category': rating.category.name,
                    'probability': rating.probability.name,
                    'blocked': getattr(rating, 'blocked', False)
                }
                for rating in (response.candidates[0].safety_ratings if response.candidates else [])
            ],
            'raw_response': response
        }
        
        result['estimated_cost'] = self._calculate_cost(model_name, result['usage'])
        
        return result
    
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
        """Check Vertex AI service health"""
        try:
            # Simple generation test using fastest model
            test_response = await self.generate_response(
                "Hello",
                model_name='text-bison@001',
                max_tokens=5
            )
            
            return {
                'status': 'healthy',
                'service': 'gcp_vertex_ai',
                'test_response_length': len(test_response.get('text', '')),
                'available_models': list(self.model_configs.keys()),
                'project_id': settings.gcp_project_id,
                'location': settings.gcp_region
            }
            
        except Exception as e:
            logger.error("Vertex AI health check failed", error=str(e))
            return {
                'status': 'unhealthy',
                'service': 'gcp_vertex_ai',
                'error': str(e),
                'available_models': list(self.model_configs.keys())
            }

# Global instance for dependency injection
vertex_ai_service = VertexAIService()

async def get_vertex_ai_service() -> VertexAIService:
    """FastAPI dependency for Vertex AI service"""
    return vertex_ai_service
```

## RAG Integration with Vertex AI Search

### Vertex AI Search RAG Service

```python
# services/vertex_rag_service.py
import asyncio
from typing import Dict, Any, Optional, List
import structlog
from google.cloud import discoveryengine_v1beta as discoveryengine
from google.oauth2 import service_account
import json

from config.settings import settings
from services.gcp_vertex_service import VertexAIService

logger = structlog.get_logger()

class VertexRAGService:
    """RAG implementation using Vertex AI Search and Vertex AI models"""
    
    def __init__(self):
        self.project_id = settings.gcp_project_id
        self.location = settings.gcp_region
        self.data_store_id = settings.vertex_search_data_store_id
        self.vertex_service = VertexAIService()
        
        # Search configurations
        self.search_configs = {
            'unstructured': {
                'type': 'unstructured',
                'description': 'General document search'
            },
            'structured': {
                'type': 'structured',
                'description': 'Database and structured data search'
            },
            'website': {
                'type': 'website',
                'description': 'Website content search'
            }
        }
    
    async def query_vertex_search(
        self,
        query: str,
        max_results: int = 5,
        search_type: str = 'unstructured',
        filter_expression: Optional[str] = None,
        boost_spec: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Query Vertex AI Search with advanced configuration"""
        
        try:
            # Initialize search client
            client = discoveryengine.SearchServiceClient()
            
            # Build the search request
            serving_config = client.serving_config_path(
                project=self.project_id,
                location=self.location,
                data_store=self.data_store_id,
                serving_config="default_config"
            )
            
            # Configure search request
            search_request = discoveryengine.SearchRequest(
                serving_config=serving_config,
                query=query,
                page_size=max_results,
                query_expansion_spec=discoveryengine.SearchRequest.QueryExpansionSpec(
                    condition=discoveryengine.SearchRequest.QueryExpansionSpec.Condition.AUTO,
                ),
                spell_correction_spec=discoveryengine.SearchRequest.SpellCorrectionSpec(
                    mode=discoveryengine.SearchRequest.SpellCorrectionSpec.Mode.AUTO
                )
            )
            
            # Add content search spec for better results
            search_request.content_search_spec = discoveryengine.SearchRequest.ContentSearchSpec(
                snippet_spec=discoveryengine.SearchRequest.ContentSearchSpec.SnippetSpec(
                    return_snippet=True,
                    max_snippet_count=3
                ),
                summary_spec=discoveryengine.SearchRequest.ContentSearchSpec.SummarySpec(
                    summary_result_count=3,
                    include_citations=True
                )
            )
            
            # Add filter if provided
            if filter_expression:
                search_request.filter = filter_expression
            
            # Add boost specification if provided
            if boost_spec:
                search_request.boost_spec = boost_spec
            
            # Execute search
            response = await asyncio.to_thread(client.search, search_request)
            
            # Parse results
            contexts = []
            for result in response.results:
                document = result.document
                
                # Extract document content
                content = ""
                if hasattr(document, 'derived_struct_data'):
                    struct_data = document.derived_struct_data
                    if 'snippets' in struct_data:
                        content = ' '.join([
                            snippet.get('snippet', '') 
                            for snippet in struct_data.get('snippets', [])
                        ])
                    elif 'extractive_answers' in struct_data:
                        content = ' '.join([
                            answer.get('content', '')
                            for answer in struct_data.get('extractive_answers', [])
                        ])
                
                context = {
                    'content': content or document.json_data,
                    'title': struct_data.get('title', 'Unknown') if hasattr(document, 'derived_struct_data') else 'Unknown',
                    'uri': struct_data.get('uri', '') if hasattr(document, 'derived_struct_data') else '',
                    'score': getattr(result, 'relevance_score', 0.0),
                    'metadata': dict(document.struct_data) if hasattr(document, 'struct_data') else {},
                    'document_id': document.id
                }
                
                contexts.append(context)
            
            # Include summary if available
            summary = None
            if hasattr(response, 'summary') and response.summary:
                summary = {
                    'summary_text': response.summary.summary_text,
                    'summary_skipped_reasons': [
                        reason.name for reason in response.summary.summary_skipped_reasons
                    ] if response.summary.summary_skipped_reasons else []
                }
            
            logger.info(
                "Vertex AI Search successful",
                query_length=len(query),
                search_type=search_type,
                results_found=len(contexts),
                has_summary=bool(summary)
            )
            
            return {
                'contexts': contexts,
                'query': query,
                'search_type': search_type,
                'total_results': len(contexts),
                'summary': summary,
                'search_metadata': {
                    'data_store_id': self.data_store_id,
                    'max_results': max_results,
                    'filter': filter_expression
                }
            }
            
        except Exception as e:
            logger.error(
                "Vertex AI Search query failed",
                query=query[:100],
                error=str(e)
            )
            return {
                'contexts': [],
                'query': query,
                'search_type': search_type,
                'error': str(e)
            }
    
    async def generate_with_rag(
        self,
        query: str,
        model_name: str = 'gemini-1.0-pro',
        system_instruction: Optional[str] = None,
        search_type: str = 'unstructured',
        max_results: int = 5,
        max_context_length: int = 4000,
        include_sources: bool = True,
        use_search_summary: bool = True,
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        """Generate response using RAG with Vertex AI Search"""
        
        # Retrieve relevant context
        search_results = await self.query_vertex_search(
            query=query,
            max_results=max_results,
            search_type=search_type
        )
        
        if not search_results['contexts'] and not search_results.get('summary'):
            logger.warning(
                "No search context found, using direct generation",
                query=query[:100],
                search_type=search_type
            )
            # Fallback to direct generation without context
            direct_result = await self.vertex_service.generate_response(
                prompt=query,
                model_name=model_name,
                system_instruction=system_instruction,
                temperature=temperature
            )
            direct_result['rag_used'] = False
            return direct_result
        
        # Build context-enhanced prompt
        enhanced_prompt_parts = []
        
        # Use search summary if available and enabled
        if use_search_summary and search_results.get('summary'):
            summary_text = search_results['summary']['summary_text']
            enhanced_prompt_parts.append(f"Search Summary:\n{summary_text}")
        
        # Add individual search results
        if search_results['contexts']:
            context_parts = []
            total_context_length = 0
            
            for i, ctx in enumerate(search_results['contexts']):
                context_text = f"Source {i+1} (Score: {ctx['score']:.2f}):\nTitle: {ctx['title']}\nContent: {ctx['content']}"
                
                # Check context length limit
                if total_context_length + len(context_text) > max_context_length:
                    logger.info(
                        "Context length limit reached",
                        contexts_used=i,
                        total_length=total_context_length
                    )
                    break
                
                context_parts.append(context_text)
                total_context_length += len(context_text)
            
            if context_parts:
                enhanced_prompt_parts.append("Search Results:\n" + "\n\n".join(context_parts))
        
        # Build the full prompt
        context_text = "\n\n".join(enhanced_prompt_parts)
        
        # Enhanced system instruction for RAG
        rag_system_instruction = system_instruction or ""
        if rag_system_instruction