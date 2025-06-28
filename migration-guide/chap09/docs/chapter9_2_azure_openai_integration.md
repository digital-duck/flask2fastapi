# Chapter 9, Section 2: Azure OpenAI Integration

## Overview

Azure OpenAI Service provides access to OpenAI's latest models including GPT-4, GPT-3.5, DALL-E, and Whisper through Microsoft's enterprise-grade cloud infrastructure. It offers unique advantages including guaranteed SLAs, content filtering, custom fine-tuning, and seamless integration with Azure's ecosystem.

## Key Advantages

- **Latest OpenAI Models:** Access to GPT-4 Turbo, GPT-4, GPT-3.5 Turbo with enterprise SLAs
- **Enterprise Security:** Advanced security features, compliance certifications, and data residency
- **Content Filtering:** Built-in content moderation and safety filters
- **Custom Fine-tuning:** Ability to fine-tune models on custom datasets
- **Azure Integration:** Native integration with Azure AI Search, Cognitive Services, and other Azure services
- **Guaranteed SLAs:** Enterprise-grade availability and performance guarantees

## Model Landscape

### Available Models

| Model Family | Deployment Name | Strengths | Use Cases |
|--------------|-----------------|-----------|-----------|
| **GPT-4 Turbo** | `gpt-4-turbo` | Latest capabilities, 128K context | Complex reasoning, analysis, coding |
| **GPT-4** | `gpt-4` | High quality, reliable | Professional writing, complex tasks |
| **GPT-3.5 Turbo** | `gpt-35-turbo` | Fast, cost-effective | Chat, simple tasks, high volume |
| **GPT-4 Vision** | `gpt-4-vision-preview` | Multimodal capabilities | Image analysis, visual Q&A |
| **DALL-E 3** | `dall-e-3` | Image generation | Creative visual content |
| **Whisper** | `whisper` | Speech-to-text | Audio transcription, voice interfaces |

### Model Selection Guide

```python
AZURE_MODEL_SELECTION_GUIDE = {
    'complex_reasoning': 'gpt-4-turbo',
    'fast_responses': 'gpt-35-turbo',
    'code_generation': 'gpt-4-turbo',
    'cost_optimization': 'gpt-35-turbo',
    'image_analysis': 'gpt-4-vision-preview',
    'creative_writing': 'gpt-4',
    'high_volume': 'gpt-35-turbo',
    'enterprise_grade': 'gpt-4'
}
```

## FastAPI Service Implementation

### Core Azure OpenAI Service

```python
# services/azure_openai_service.py
import asyncio
import json
from typing import Dict, Any, Optional, List, AsyncGenerator
import structlog
import httpx
from openai import AsyncAzureOpenAI
from openai.types.chat import ChatCompletion
import base64
import io

from config.settings import settings

logger = structlog.get_logger()

class AzureOpenAIService:
    """Azure OpenAI service integration with comprehensive features"""
    
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
                'temperature': 0.7,
                'context_window': 128000,
                'pricing': {'input': 0.01, 'output': 0.03}  # per 1K tokens
            },
            'gpt-4': {
                'deployment_name': 'gpt-4',
                'max_tokens': 8192,
                'temperature': 0.7,
                'context_window': 8192,
                'pricing': {'input': 0.03, 'output': 0.06}
            },
            'gpt-35-turbo': {
                'deployment_name': 'gpt-35-turbo',
                'max_tokens': 4096,
                'temperature': 0.7,
                'context_window': 16385,
                'pricing': {'input': 0.0015, 'output': 0.002}
            },
            'gpt-4-vision': {
                'deployment_name': 'gpt-4-vision-preview',
                'max_tokens': 4096,
                'temperature': 0.7,
                'context_window': 128000,
                'pricing': {'input': 0.01, 'output': 0.03}
            }
        }
        
        # Content filtering levels
        self.content_filter_levels = {
            'strict': 0,
            'medium': 1, 
            'low': 2,
            'off': 3
        }
    
    async def generate_response(
        self,
        prompt: str,
        model_name: str = 'gpt-4-turbo',
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stream: bool = False,
        tools: Optional[List[Dict]] = None,
        content_filter_level: str = 'medium'
    ) -> Dict[str, Any]:
        """Generate response using Azure OpenAI models with advanced features"""
        
        if model_name not in self.model_configs:
            available_models = list(self.model_configs.keys())
            raise ValueError(f"Unsupported model: {model_name}. Available: {available_models}")
        
        config = self.model_configs[model_name]
        deployment_name = config['deployment_name']
        
        # Build messages array
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        # Prepare request parameters
        request_params = {
            "model": deployment_name,
            "messages": messages,
            "max_tokens": max_tokens or config['max_tokens'],
            "temperature": temperature or config['temperature'],
            "stream": stream
        }
        
        # Add tools if provided (function calling)
        if tools:
            request_params["tools"] = tools
            request_params["tool_choice"] = "auto"
        
        try:
            if stream:
                return await self._handle_streaming_response(request_params, model_name)
            else:
                response = await self.client.chat.completions.create(**request_params)
                return await self._handle_standard_response(response, model_name)
                
        except Exception as e:
            logger.error(
                "Azure OpenAI API error",
                error=str(e),
                model=model_name,
                deployment=deployment_name
            )
            
            # Handle specific Azure OpenAI errors
            if "content_filter" in str(e).lower():
                raise Exception("Content filtered due to policy violation")
            elif "rate_limit" in str(e).lower():
                raise Exception("Rate limit exceeded. Please try again later.")
            elif "quota" in str(e).lower():
                raise Exception("Quota exceeded for this deployment")
            else:
                raise Exception(f"Azure OpenAI error: {str(e)}")
    
    async def _handle_standard_response(
        self, 
        response: ChatCompletion, 
        model_name: str
    ) -> Dict[str, Any]:
        """Handle standard (non-streaming) response"""
        choice = response.choices[0]
        
        # Handle function calls if present
        function_call = None
        if choice.message.tool_calls:
            function_call = {
                'name': choice.message.tool_calls[0].function.name,
                'arguments': choice.message.tool_calls[0].function.arguments
            }
        
        result = {
            'text': choice.message.content or '',
            'model': model_name,
            'usage': {
                'prompt_tokens': response.usage.prompt_tokens,
                'completion_tokens': response.usage.completion_tokens,
                'total_tokens': response.usage.total_tokens
            },
            'finish_reason': choice.finish_reason,
            'function_call': function_call,
            'raw_response': response.model_dump()
        }
        
        # Add cost estimation
        result['estimated_cost'] = self._calculate_cost(model_name, result['usage'])
        
        return result
    
    async def _handle_streaming_response(
        self, 
        request_params: Dict,
        model_name: str
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Handle streaming response for real-time applications"""
        full_content = ""
        usage_info = {}
        
        try:
            stream = await self.client.chat.completions.create(**request_params)
            
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    content_delta = chunk.choices[0].delta.content
                    full_content += content_delta
                    
                    yield {
                        'delta': content_delta,
                        'accumulated_text': full_content,
                        'model': model_name,
                        'streaming': True
                    }
                
                # Capture usage info from final chunk
                if hasattr(chunk, 'usage') and chunk.usage:
                    usage_info = {
                        'prompt_tokens': chunk.usage.prompt_tokens,
                        'completion_tokens': chunk.usage.completion_tokens,
                        'total_tokens': chunk.usage.total_tokens
                    }
            
            # Final summary chunk
            yield {
                'text': full_content,
                'model': model_name,
                'usage': usage_info,
                'estimated_cost': self._calculate_cost(model_name, usage_info),
                'streaming': False,
                'final': True
            }
            
        except Exception as e:
            logger.error("Streaming response failed", error=str(e))
            yield {
                'error': str(e),
                'model': model_name,
                'streaming': False,
                'final': True
            }
    
    async def generate_with_vision(
        self,
        prompt: str,
        image_url: Optional[str] = None,
        image_data: Optional[bytes] = None,
        model_name: str = 'gpt-4-vision',
        max_tokens: Optional[int] = None,
        detail_level: str = 'auto'
    ) -> Dict[str, Any]:
        """Generate response with image analysis using GPT-4 Vision"""
        
        if 'vision' not in model_name:
            raise ValueError("Vision capabilities require a vision-enabled model")
        
        config = self.model_configs[model_name]
        
        # Prepare image content
        image_content = None
        if image_url:
            image_content = {"type": "image_url", "image_url": {"url": image_url, "detail": detail_level}}
        elif image_data:
            # Convert bytes to base64 data URL
            base64_image = base64.b64encode(image_data).decode('utf-8')
            data_url = f"data:image/jpeg;base64,{base64_image}"
            image_content = {"type": "image_url", "image_url": {"url": data_url, "detail": detail_level}}
        else:
            raise ValueError("Either image_url or image_data must be provided")
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    image_content
                ]
            }
        ]
        
        try:
            response = await self.client.chat.completions.create(
                model=config['deployment_name'],
                messages=messages,
                max_tokens=max_tokens or config['max_tokens']
            )
            
            result = await self._handle_standard_response(response, model_name)
            result['vision_analysis'] = True
            result['detail_level'] = detail_level
            
            return result
            
        except Exception as e:
            logger.error("Azure OpenAI vision error", error=str(e))
            raise Exception(f"Vision analysis failed: {str(e)}")
    
    async def create_embeddings(
        self,
        texts: List[str],
        model_name: str = 'text-embedding-ada-002'
    ) -> Dict[str, Any]:
        """Create embeddings using Azure OpenAI"""
        try:
            response = await self.client.embeddings.create(
                model=model_name,
                input=texts
            )
            
            embeddings = [data.embedding for data in response.data]
            
            result = {
                'embeddings': embeddings,
                'model': model_name,
                'usage': {
                    'prompt_tokens': response.usage.prompt_tokens,
                    'total_tokens': response.usage.total_tokens
                }
            }
            
            # Add cost estimation for embeddings
            embedding_cost = (response.usage.prompt_tokens / 1000) * 0.0001  # $0.0001 per 1K tokens
            result['estimated_cost'] = round(embedding_cost, 6)
            
            return result
            
        except Exception as e:
            logger.error("Azure OpenAI embeddings error", error=str(e))
            raise Exception(f"Embeddings generation failed: {str(e)}")
    
    async def moderate_content(
        self, 
        text: str,
        model: str = 'text-moderation-latest'
    ) -> Dict[str, Any]:
        """Check content using Azure OpenAI moderation"""
        try:
            response = await self.client.moderations.create(
                input=text,
                model=model
            )
            
            result = response.results[0]
            
            moderation_result = {
                'flagged': result.flagged,
                'categories': result.categories.model_dump(),
                'category_scores': result.category_scores.model_dump(),
                'model': model
            }
            
            # Add detailed analysis
            if result.flagged:
                flagged_categories = [
                    category for category, flagged in result.categories.model_dump().items()
                    if flagged
                ]
                moderation_result['flagged_categories'] = flagged_categories
                moderation_result['highest_score_category'] = max(
                    result.category_scores.model_dump().items(),
                    key=lambda x: x[1]
                )[0]
            
            return moderation_result
            
        except Exception as e:
            logger.error("Azure OpenAI moderation error", error=str(e))
            raise Exception(f"Content moderation failed: {str(e)}")
    
    async def generate_with_functions(
        self,
        prompt: str,
        functions: List[Dict],
        model_name: str = 'gpt-4-turbo',
        system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate response with function calling capabilities"""
        
        # Convert functions to tools format
        tools = [
            {
                "type": "function",
                "function": func
            }
            for func in functions
        ]
        
        result = await self.generate_response(
            prompt=prompt,
            model_name=model_name,
            system_prompt=system_prompt,
            tools=tools
        )
        
        result['functions_available'] = len(functions)
        result['function_calling_enabled'] = True
        
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
        """Check Azure OpenAI service health"""
        try:
            # Simple completion test using fastest model
            test_response = await self.generate_response(
                "Hello",
                model_name='gpt-35-turbo',
                max_tokens=5
            )
            
            return {
                'status': 'healthy',
                'service': 'azure_openai',
                'test_response_tokens': test_response.get('usage', {}).get('total_tokens', 0),
                'available_models': list(self.model_configs.keys()),
                'endpoint': settings.azure_openai_endpoint,
                'api_version': settings.azure_openai_api_version
            }
            
        except Exception as e:
            logger.error("Azure OpenAI health check failed", error=str(e))
            return {
                'status': 'unhealthy',
                'service': 'azure_openai',
                'error': str(e),
                'available_models': list(self.model_configs.keys())
            }

# Global instance for dependency injection
azure_openai_service = AzureOpenAIService()

async def get_azure_openai_service() -> AzureOpenAIService:
    """FastAPI dependency for Azure OpenAI service"""
    return azure_openai_service
```

## RAG Integration with Azure AI Search

### Azure AI Search RAG Service

```python
# services/azure_rag_service.py
import asyncio
from typing import Dict, Any, Optional, List
import structlog
import httpx
import json

from config.settings import settings
from services.azure_openai_service import AzureOpenAIService

logger = structlog.get_logger()

class AzureRAGService:
    """RAG implementation using Azure AI Search and Azure OpenAI"""
    
    def __init__(self):
        self.search_endpoint = settings.azure_search_endpoint
        self.search_key = settings.azure_search_key
        self.search_index = settings.azure_search_index
        self.openai_service = AzureOpenAIService()
        
        # Search configurations
        self.search_configs = {
            'hybrid': {
                'search_type': 'hybrid',
                'semantic_configuration': 'default',
                'query_type': 'semantic'
            },
            'vector': {
                'search_type': 'vector',
                'vector_fields': ['content_vector']
            },
            'text': {
                'search_type': 'text',
                'search_mode': 'all'
            }
        }
    
    async def query_search_index(
        self,
        query: str,
        top_k: int = 5,
        search_type: str = 'hybrid',
        filter_expression: Optional[str] = None
    ) -> Dict[str, Any]:
        """Query Azure AI Search index with multiple search types"""
        
        search_url = f"{self.search_endpoint}/indexes/{self.search_index}/docs/search"
        
        headers = {
            'Content-Type': 'application/json',
            'api-key': self.search_key
        }
        
        # Build search payload based on search type
        if search_type == 'hybrid':
            # Create query embedding for hybrid search
            embedding_result = await self.openai_service.create_embeddings([query])
            query_vector = embedding_result['embeddings'][0]
            
            search_payload = {
                "search": query,
                "vectors": [{
                    "value": query_vector,
                    "fields": "content_vector",
                    "k": top_k
                }],
                "queryType": "semantic",
                "semanticConfiguration": "default",
                "top": top_k,
                "select": "content, title, metadata, url",
                "highlightFields": "content",
                "highlightPreTag": "<mark>",
                "highlightPostTag": "</mark>"
            }
        
        elif search_type == 'vector':
            # Vector-only search
            embedding_result = await self.openai_service.create_embeddings([query])
            query_vector = embedding_result['embeddings'][0]
            
            search_payload = {
                "vectors": [{
                    "value": query_vector,
                    "fields": "content_vector",
                    "k": top_k
                }],
                "top": top_k,
                "select": "content, title, metadata, url"
            }
        
        else:  # text search
            search_payload = {
                "search": query,
                "searchMode": "all",
                "top": top_k,
                "select": "content, title, metadata, url",
                "highlightFields": "content"
            }
        
        # Add filter if provided
        if filter_expression:
            search_payload["filter"] = filter_expression
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    search_url,
                    headers=headers,
                    json=search_payload
                )
                
                if response.status_code == 200:
                    search_results = response.json()
                    
                    contexts = []
                    for result in search_results.get('value', []):
                        context = {
                            'content': result.get('content', ''),
                            'title': result.get('title', ''),
                            'url': result.get('url', ''),
                            'metadata': result.get('metadata', {}),
                            'highlights': result.get('@search.highlights', {}),
                            'score': result.get('@search.score', 0.0),
                            'reranker_score': result.get('@search.rerankerScore', 0.0)
                        }
                        contexts.append(context)
                    
                    logger.info(
                        "Azure AI Search successful",
                        query_length=len(query),
                        search_type=search_type,
                        results_found=len(contexts),
                        avg_score=sum(c['score'] for c in contexts) / len(contexts) if contexts else 0
                    )
                    
                    return {
                        'contexts': contexts,
                        'query': query,
                        'search_type': search_type,
                        'total_results': len(contexts),
                        'search_metadata': {
                            'index': self.search_index,
                            'top_k': top_k,
                            'filter': filter_expression
                        }
                    }
                else:
                    logger.error(
                        "Azure AI Search error",
                        status_code=response.status_code,
                        response=response.text[:500]
                    )
                    return {
                        'contexts': [], 
                        'query': query, 
                        'error': f'Search failed with status {response.status_code}'
                    }
                    
        except Exception as e:
            logger.error("Azure AI Search query failed", error=str(e))
            return {
                'contexts': [], 
                'query': query, 
                'error': str(e)
            }
    
    async def generate_with_rag(
        self,
        query: str,
        model_name: str = 'gpt-4-turbo',
        system_prompt: Optional[str] = None,
        search_type: str = 'hybrid',
        top_k: int = 5,
        max_context_length: int = 4000,
        include_sources: bool = True,
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        """Generate response using RAG with Azure AI Search"""
        
        # Retrieve relevant context
        search_results = await self.query_search_index(
            query=query,
            top_k=top_k,
            search_type=search_type
        )
        
        if not search_results['contexts']:
            logger.warning(
                "No search context found, using direct generation",
                query=query[:100],
                search_type=search_type
            )
            # Fallback to direct generation without context
            direct_result = await self.openai_service.generate_response(
                prompt=query,
                model_name=model_name,
                system_prompt=system_prompt,
                temperature=temperature
            )
            direct_result['rag_used'] = False
            return direct_result
        
        # Build context-enhanced prompt
        context_parts = []
        total_context_length = 0
        
        for i, ctx in enumerate(search_results['contexts']):
            # Use highlights if available, otherwise use full content
            content = ctx.get('highlights', {}).get('content', [])
            if content:
                context_text = ' '.join(content)
            else:
                context_text = ctx['content']
            
            source_info = f"Title: {ctx['title']}\nURL: {ctx['url']}\n" if ctx.get('title') else ""
            formatted_context = f"Source {i+1} (Relevance: {ctx['score']:.2f}):\n{source_info}Content: {context_text}"
            
            # Check context length limit
            if total_context_length + len(formatted_context) > max_context_length:
                logger.info(
                    "Context length limit reached",
                    contexts_used=i,
                    total_length=total_context_length
                )
                break
            
            context_parts.append(formatted_context)
            total_context_length += len(formatted_context)
        
        context_text = "\n\n".join(context_parts)
        
        # Enhanced system prompt for RAG
        rag_system_prompt = system_prompt or ""
        if rag_system_prompt:
            rag_system_prompt += "\n\n"
        
        rag_system_prompt += """You are an AI assistant that answers questions based on provided search results. 
        Guidelines:
        1. Base your answer on the provided sources and their relevance scores
        2. Cite specific sources when making claims
        3. If information is insufficient, clearly state this
        4. Prioritize information from higher-scoring sources
        5. Provide comprehensive answers with relevant details"""
        
        enhanced_prompt = f"""Based on the following search results, please answer the question:

Search Results:
{context_text}

Question: {query}

Please provide a detailed answer based on the search results, citing relevant sources."""
        
        # Generate response with context
        result = await self.openai_service.generate_response(
            prompt=enhanced_prompt,
            model_name=model_name,
            system_prompt=rag_system_prompt,
            temperature=temperature
        )
        
        # Add RAG metadata
        result['rag_used'] = True
        result['rag_context'] = search_results['contexts']
        result['rag_query'] = query
        result['search_type'] = search_type
        result['context_length'] = total_context_length
        result['contexts_used'] = len(context_parts)
        
        # Include source information if requested
        if include_sources:
            result['sources'] = [
                {
                    'title': ctx['title'],
                    'url': ctx['url'],
                    'score': ctx['score'],
                    'reranker_score': ctx.get('reranker_score', 0.0),
                    'metadata': ctx['metadata']
                }
                for ctx in search_results['contexts'][:len(context_parts)]
            ]
        
        logger.info(
            "Azure RAG generation completed",
            model=model_name,
            search_type=search_type,
            contexts_used=len(context_parts),
            context_length=total_context_length
        )
        
        return result
    
    async def get_search_suggestions(
        self,
        partial_query: str,
        max_suggestions: int = 5
    ) -> List[str]:
        """Get search suggestions for autocomplete"""
        
        suggest_url = f"{self.search_endpoint}/indexes/{self.search_index}/docs/suggest"
        
        headers = {
            'Content-Type': 'application/json',
            'api-key': self.search_key
        }
        
        suggest_payload = {
            "search": partial_query,
            "suggesterName": "sg-default",  # Configure suggester in your index
            "top": max_suggestions,
            "select": "title, content"
        }
        
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    suggest_url,
                    headers=headers,
                    json=suggest_payload
                )
                
                if response.status_code == 200:
                    suggestions = response.json()
                    return [
                        result['@search.text']
                        for result in suggestions.get('value', [])
                    ]
                else:
                    logger.error("Search suggestions failed", status_code=response.status_code)
                    return []
                    
        except Exception as e:
            logger.error("Search suggestions error", error=str(e))
            return []

# Global instance for dependency injection
azure_rag_service = AzureRAGService()

async def get_azure_rag_service() -> AzureRAGService:
    """FastAPI dependency for Azure RAG service"""
    return azure_rag_service
```

## FastAPI Routes

### Azure OpenAI Chat Routes

```python
# routers/azure_chat.py
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, File, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Literal, Union
import structlog
import time
import json
import asyncio

from services.azure_openai_service import (
    AzureOpenAIService, 
    get_azure_openai_service
)
from services.azure_rag_service import (
    AzureRAGService,
    get_azure_rag_service
)

logger = structlog.get_logger()
router = APIRouter(prefix="/azure", tags=["Azure OpenAI"])

# Request Models
class AzureChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=8000, description="User message")
    model_name: Literal['gpt-4-turbo', 'gpt-4', 'gpt-35-turbo'] = Field(
        default="gpt-4-turbo",
        description="Azure OpenAI model to use"
    )
    system_prompt: Optional[str] = Field(None, max_length=2000, description="System prompt")
    max_tokens: Optional[int] = Field(default=None, ge=1, le=8000, description="Max tokens")
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0, description="Temperature")
    stream: bool = Field(default=False, description="Enable streaming response")
    functions: Optional[List[Dict]] = Field(None, description="Available functions for calling")

class AzureRAGRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=8000)
    model_name: Literal['gpt-4-turbo', 'gpt-4', 'gpt-35-turbo'] = Field(default="gpt-4-turbo")
    system_prompt: Optional[str] = Field(None, max_length=2000)
    search_type: Literal['hybrid', 'vector', 'text'] = Field(default='hybrid')
    top_k: int = Field(default=5, ge=1, le=20, description="Number of search results")
    max_context_length: int = Field(default=4000, ge=500, le=8000)
    include_sources: bool = Field(default=True)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)

class AzureVisionRequest(BaseModel):
    message: str = Field(..., min_length=1)
    image_url: Optional[str] = Field(None, description="URL of the image to analyze")
    detail_level: Literal['low', 'high', 'auto'] = Field(default='auto')
    model_name: Literal['gpt-4-vision'] = Field(default="gpt-4-vision")
    max_tokens: Optional[int] = Field(default=300, ge=1, le=4096)

class ContentModerationRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000)
    model: str = Field(default='text-moderation-latest')

# Response Models
class AzureChatResponse(BaseModel):
    response: str
    model_used: str
    tokens_used: Dict[str, int]
    processing_time_ms: float
    estimated_cost: float
    finish_reason: Optional[str] = None
    function_call: Optional[Dict] = None

class AzureRAGResponse(BaseModel):
    response: str
    model_used: str
    tokens_used: Dict[str, int]
    processing_time_ms: float
    estimated_cost: float
    rag_used: bool
    search_type: str
    contexts_used: int
    context_length: Optional[int] = None
    sources: Optional[List[Dict]] = None

class AzureVisionResponse(BaseModel):
    analysis: str
    model_used: str
    tokens_used: Dict[str, int]
    processing_time_ms: float
    estimated_cost: float
    detail_level: str
    vision_analysis: bool = True

class ContentModerationResponse(BaseModel):
    flagged: bool
    categories: Dict[str, bool]
    category_scores: Dict[str, float]
    flagged_categories: Optional[List[str]] = None
    highest_score_category: Optional[str] = None
    model: str

@router.post("/chat", response_model=AzureChatResponse)
async def chat_with_azure_openai(
    request: AzureChatRequest,
    background_tasks: BackgroundTasks,
    azure_openai: AzureOpenAIService = Depends(get_azure_openai_service)
):
    """Chat endpoint using Azure OpenAI models with function calling support"""
    start_time = time.time()
    
    try:
        # Handle function calling if functions provided
        if request.functions:
            result = await azure_openai.generate_with_functions(
                prompt=request.message,
                functions=request.functions,
                model_name=request.model_name,
                system_prompt=request.system_prompt
            )
        else:
            result = await azure_openai.generate_response(
                prompt=request.message,
                model_name=request.model_name,
                system_prompt=request.system_prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                stream=request.stream
            )
        
        processing_time = (time.time() - start_time) * 1000
        
        # Log usage for analytics
        background_tasks.add_task(
            log_azure_usage,
            model_name=request.model_name,
            tokens=result.get('usage', {}),
            processing_time=processing_time,
            cost=result.get('estimated_cost', 0),
            rag_used=False,
            function_calling=bool(request.functions)
        )
        
        return AzureChatResponse(
            response=result['text'],
            model_used=request.model_name,
            tokens_used=result.get('usage', {}),
            processing_time_ms=processing_time,
            estimated_cost=result.get('estimated_cost', 0),
            finish_reason=result.get('finish_reason'),
            function_call=result.get('function_call')
        )
        
    except ValueError as e:
        logger.warning("Azure OpenAI validation error", error=str(e))
        raise HTTPException(status_code=400, detail=str(e))
        
    except Exception as e:
        logger.error("Azure OpenAI chat failed", error=str(e))
        raise HTTPException(
            status_code=500,
            detail="Azure OpenAI service temporarily unavailable"
        )

@router.post("/chat/stream")
async def stream_chat_with_azure_openai(
    request: AzureChatRequest,
    azure_openai: AzureOpenAIService = Depends(get_azure_openai_service)
):
    """Streaming chat endpoint using Azure OpenAI"""
    
    async def generate_stream():
        try:
            # Force streaming mode
            request.stream = True
            
            async for chunk in azure_openai.generate_response(
                prompt=request.message,
                model_name=request.model_name,
                system_prompt=request.system_prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                stream=True
            ):
                # Format as Server-Sent Events
                if 'error' in chunk:
                    yield f"data: {json.dumps({'error': chunk['error']})}\n\n"
                    break
                elif chunk.get('final'):
                    yield f"data: {json.dumps({'content': chunk.get('text', ''), 'done': True, 'usage': chunk.get('usage', {})})}\n\n"
                    break
                else:
                    yield f"data: {json.dumps({'content': chunk.get('delta', ''), 'done': False})}\n\n"
            
        except Exception as e:
            logger.error("Azure OpenAI streaming failed", error=str(e))
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"  # Disable nginx buffering
        }
    )

@router.post("/chat/rag", response_model=AzureRAGResponse)
async def chat_with_azure_rag(
    request: AzureRAGRequest,
    background_tasks: BackgroundTasks,
    azure_rag: AzureRAGService = Depends(get_azure_rag_service)
):
    """Chat endpoint using Azure OpenAI with RAG (Azure AI Search)"""
    start_time = time.time()
    
    try:
        result = await azure_rag.generate_with_rag(
            query=request.message,
            model_name=request.model_name,
            system_prompt=request.system_prompt,
            search_type=request.search_type,
            top_k=request.top_k,
            max_context_length=request.max_context_length,
            include_sources=request.include_sources,
            temperature=request.temperature
        )
        
        processing_time = (time.time() - start_time) * 1000
        
        # Log usage for analytics
        background_tasks.add_task(
            log_azure_usage,
            model_name=request.model_name,
            tokens=result.get('usage', {}),
            processing_time=processing_time,
            cost=result.get('estimated_cost', 0),
            rag_used=result.get('rag_used', False),
            search_type=request.search_type
        )
        
        return AzureRAGResponse(
            response=result['text'],
            model_used=request.model_name,
            tokens_used=result.get('usage', {}),
            processing_time_ms=processing_time,
            estimated_cost=result.get('estimated_cost', 0),
            rag_used=result.get('rag_used', False),
            search_type=request.search_type,
            contexts_used=result.get('contexts_used', 0),
            context_length=result.get('context_length'),
            sources=result.get('sources')
        )
        
    except ValueError as e:
        logger.warning("Azure RAG validation error", error=str(e))
        raise HTTPException(status_code=400, detail=str(e))
        
    except Exception as e:
        logger.error("Azure RAG failed", error=str(e))
        raise HTTPException(
            status_code=500,
            detail="Azure RAG service temporarily unavailable"
        )

@router.post("/vision", response_model=AzureVisionResponse)
async def analyze_image_with_azure_openai(
    request: AzureVisionRequest,
    background_tasks: BackgroundTasks,
    azure_openai: AzureOpenAIService = Depends(get_azure_openai_service)
):
    """Image analysis endpoint using Azure OpenAI GPT-4 Vision"""
    start_time = time.time()
    
    try:
        result = await azure_openai.generate_with_vision(
            prompt=request.message,
            image_url=request.image_url,
            model_name=request.model_name,
            max_tokens=request.max_tokens,
            detail_level=request.detail_level
        )
        
        processing_time = (time.time() - start_time) * 1000
        
        # Log vision usage
        background_tasks.add_task(
            log_azure_vision_usage,
            model_name=request.model_name,
            tokens=result.get('usage', {}),
            processing_time=processing_time,
            cost=result.get('estimated_cost', 0),
            detail_level=request.detail_level
        )
        
        return AzureVisionResponse(
            analysis=result['text'],
            model_used=request.model_name,
            tokens_used=result.get('usage', {}),
            processing_time_ms=processing_time,
            estimated_cost=result.get('estimated_cost', 0),
            detail_level=request.detail_level
        )
        
    except Exception as e:
        logger.error("Azure OpenAI vision failed", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Vision analysis failed: {str(e)}"
        )

@router.post("/vision/upload", response_model=AzureVisionResponse)
async def analyze_uploaded_image(
    message: str,
    image: UploadFile = File(...),
    model_name: Literal['gpt-4-vision'] = 'gpt-4-vision',
    detail_level: Literal['low', 'high', 'auto'] = 'auto',
    max_tokens: int = 300,
    background_tasks: BackgroundTasks = BackgroundTasks(),
    azure_openai: AzureOpenAIService = Depends(get_azure_openai_service)
):
    """Upload and analyze image using Azure OpenAI GPT-4 Vision"""
    start_time = time.time()
    
    try:
        # Validate file type
        if not image.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read image data
        image_data = await image.read()
        
        # Limit file size (e.g., 10MB)
        if len(image_data) > 10 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="Image file too large (max 10MB)")
        
        result = await azure_openai.generate_with_vision(
            prompt=message,
            image_data=image_data,
            model_name=model_name,
            max_tokens=max_tokens,
            detail_level=detail_level
        )
        
        processing_time = (time.time() - start_time) * 1000
        
        # Log vision usage
        background_tasks.add_task(
            log_azure_vision_usage,
            model_name=model_name,
            tokens=result.get('usage', {}),
            processing_time=processing_time,
            cost=result.get('estimated_cost', 0),
            detail_level=detail_level
        )
        
        return AzureVisionResponse(
            analysis=result['text'],
            model_used=model_name,
            tokens_used=result.get('usage', {}),
            processing_time_ms=processing_time,
            estimated_cost=result.get('estimated_cost', 0),
            detail_level=detail_level
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Image upload analysis failed", error=str(e))
        raise HTTPException(
            status_code=500,
            detail="Image analysis failed"
        )

@router.post("/moderate", response_model=ContentModerationResponse)
async def moderate_content(
    request: ContentModerationRequest,
    azure_openai: AzureOpenAIService = Depends(get_azure_openai_service)
):
    """Content moderation using Azure OpenAI"""
    try:
        result = await azure_openai.moderate_content(
            text=request.text,
            model=request.model
        )
        return ContentModerationResponse(**result)
    except Exception as e:
        logger.error("Azure OpenAI moderation failed", error=str(e))
        raise HTTPException(
            status_code=500,
            detail="Content moderation service unavailable"
        )

@router.post("/embeddings")
async def create_embeddings(
    texts: List[str] = Field(..., min_items=1, max_items=100),
    model: str = Field(default='text-embedding-ada-002'),
    azure_openai: AzureOpenAIService = Depends(get_azure_openai_service)
):
    """Create embeddings using Azure OpenAI"""
    try:
        # Validate input
        if any(len(text) > 8000 for text in texts):
            raise HTTPException(status_code=400, detail="Text too long (max 8000 characters)")
        
        result = await azure_openai.create_embeddings(texts, model)
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Embeddings creation failed", error=str(e))
        raise HTTPException(
            status_code=500,
            detail="Embeddings service unavailable"
        )

@router.get("/search/suggestions")
async def get_search_suggestions(
    query: str = Field(..., min_length=1, max_length=100),
    max_suggestions: int = Field(default=5, ge=1, le=10),
    azure_rag: AzureRAGService = Depends(get_azure_rag_service)
):
    """Get search suggestions for autocomplete"""
    try:
        suggestions = await azure_rag.get_search_suggestions(query, max_suggestions)
        return {"suggestions": suggestions, "query": query}
    except Exception as e:
        logger.error("Search suggestions failed", error=str(e))
        return {"suggestions": [], "query": query}

@router.get("/models")
async def list_available_models(
    azure_openai: AzureOpenAIService = Depends(get_azure_openai_service)
):
    """List available Azure OpenAI models with capabilities and pricing"""
    models_info = {}
    
    for model_name, config in azure_openai.model_configs.items():
        models_info[model_name] = {
            'deployment_name': config['deployment_name'],
            'max_tokens': config['max_tokens'],
            'context_window': config['context_window'],
            'pricing_per_1k_tokens': config['pricing'],
            'capabilities': _get_model_capabilities(model_name),
            'best_for': _get_model_use_cases(model_name)
        }
    
    return {
        "available_models": models_info,
        "recommendation_guide": {
            "latest_capabilities": "gpt-4-turbo",
            "cost_effective": "gpt-35-turbo",
            "highest_quality": "gpt-4",
            "image_analysis": "gpt-4-vision",
            "function_calling": "gpt-4-turbo",
            "high_volume": "gpt-35-turbo"
        }
    }

@router.get("/health")
async def azure_openai_health_check(
    azure_openai: AzureOpenAIService = Depends(get_azure_openai_service)
):
    """Check Azure OpenAI service health"""
    return await azure_openai.health_check()

def _get_model_capabilities(model_name: str) -> List[str]:
    """Get capabilities for each model"""
    capabilities = {
        'gpt-4-turbo': [
            'Text generation',
            'Function calling',
            'JSON mode',
            'Seed parameter',
            '128K context window'
        ],
        'gpt-4': [
            'Text generation', 
            'Function calling',
            'High quality reasoning',
            '8K context window'
        ],
        'gpt-35-turbo': [
            'Text generation',
            'Function calling',
            'Fast responses',
            '16K context window'
        ],
        'gpt-4-vision': [
            'Text generation',
            'Image analysis',
            'Multimodal understanding',
            '128K context window'
        ]
    }
    return capabilities.get(model_name, ['Text generation'])

def _get_model_use_cases(model_name: str) -> List[str]:
    """Get recommended use cases for each model"""
    use_cases = {
        'gpt-4-turbo': [
            'Complex reasoning and analysis',
            'Code generation with latest knowledge',
            'Long document processing',
            'Function calling applications'
        ],
        'gpt-4': [
            'Professional writing and editing',
            'Complex problem solving',
            'Creative content generation',
            'Technical analysis'
        ],
        'gpt-35-turbo': [
            'Chatbots and conversational AI',
            'Content summarization',
            'Simple Q&A systems',
            'High-volume applications'
        ],
        'gpt-4-vision': [
            'Image description and analysis',
            'Visual Q&A systems',
            'Document processing with images',
            'Multimodal applications'
        ]
    }
    return use_cases.get(model_name, ['General purpose text generation'])

# Background task functions
async def log_azure_usage(
    model_name: str,
    tokens: Dict,
    processing_time: float,
    cost: float,
    rag_used: bool,
    function_calling: bool = False,
    search_type: Optional[str] = None
):
    """Background task to log Azure OpenAI usage"""
    logger.info(
        "Azure OpenAI usage logged",
        model=model_name,
        prompt_tokens=tokens.get('prompt_tokens', 0),
        completion_tokens=tokens.get('completion_tokens', 0),
        total_tokens=tokens.get('total_tokens', 0),
        processing_time_ms=processing_time,
        estimated_cost=cost,
        rag_enabled=rag_used,
        function_calling=function_calling,
        search_type=search_type
    )

async def log_azure_vision_usage(
    model_name: str,
    tokens: Dict,
    processing_time: float,
    cost: float,
    detail_level: str
):
    """Background task to log Azure OpenAI vision usage"""
    logger.info(
        "Azure OpenAI Vision usage logged",
        model=model_name,
        tokens_used=tokens.get('total_tokens', 0),
        processing_time_ms=processing_time,
        estimated_cost=cost,
        detail_level=detail_level
    )
```

## Performance Characteristics

### Benchmark Results

**Response Time Analysis (Based on Testing):**

| Model | Avg Response Time | P95 Response Time | Cold Start | Streaming Latency |
|-------|------------------|-------------------|------------|------------------|
| GPT-4 Turbo | [PLACEHOLDER: 1.8s] | [PLACEHOLDER: 3.5s] | [PLACEHOLDER: 0.5s] | [PLACEHOLDER: 150ms TTFT] |
| GPT-4 | [PLACEHOLDER: 2.2s] | [PLACEHOLDER: 4.1s] | [PLACEHOLDER: 0.6s] | [PLACEHOLDER: 200ms TTFT] |
| GPT-3.5 Turbo | [PLACEHOLDER: 1.1s] | [PLACEHOLDER: 2.3s] | [PLACEHOLDER: 0.3s] | [PLACEHOLDER: 100ms TTFT] |
| GPT-4 Vision | [PLACEHOLDER: 3.2s] | [PLACEHOLDER: 6.1s] | [PLACEHOLDER: 0.8s] | [PLACEHOLDER: N/A] |

*TTFT = Time to First Token*

### Cost Analysis

**Real-World Cost Examples:**

```python
# Azure OpenAI cost calculations for different scenarios
AZURE_COST_EXAMPLES = {
    'simple_chat': {
        'input_tokens': 100,
        'output_tokens': 150,
        'gpt_4_turbo': (100/1000 * 0.01) + (150/1000 * 0.03) = 0.0055,  # $0.0055
        'gpt_35_turbo': (100/1000 * 0.0015) + (150/1000 * 0.002) = 0.00045,  # $0.00045
        'savings_ratio': '92% cheaper with GPT-3.5 Turbo'
    },
    'document_analysis': {
        'input_tokens': 3000,  # Long document
        'output_tokens': 800,
        'gpt_4_turbo': (3000/1000 * 0.01) + (800/1000 * 0.03) = 0.054,  # $0.054
        'gpt_4': (3000/1000 * 0.03) + (800/1000 * 0.06) = 0.138,  # $0.138
        'recommendation': 'Use GPT-4 Turbo for cost savings with latest capabilities'
    },
    'vision_analysis': {
        'image_tokens': 1000,  # Estimated for high-detail image
        'text_tokens': 500,
        'gpt_4_vision': ((1000 + 500)/1000 * 0.01) + (200/1000 * 0.03) = 0.021,  # $0.021
        'note': 'Vision pricing includes image processing costs'
    }
}
```

### Rate Limits and Quotas

**Current Limits (Varies by Subscription Tier):**

```python
AZURE_RATE_LIMITS = {
    'standard_tier': {
        'gpt_4_turbo': {
            'requests_per_minute': 30,
            'tokens_per_minute': 150000
        },
        'gpt_4': {
            'requests_per_minute': 20,
            'tokens_per_minute': 80000
        },
        'gpt_35_turbo': {
            'requests_per_minute': 60,
            'tokens_per_minute': 240000
        }
    },
    'enterprise_tier': {
        # Higher limits available with enterprise agreements
        'note': 'Contact Microsoft for enterprise-specific limits'
    }
}
```

## Enterprise Features

### Content Filtering Configuration

```python
# Content filtering setup for enterprise compliance
AZURE_CONTENT_FILTERS = {
    'hate': {
        'threshold': 'medium',
        'enabled': True,
        'description': 'Hate and fairness-related harms'
    },
    'sexual': {
        'threshold': 'medium', 
        'enabled': True,
        'description': 'Sexual content'
    },
    'violence': {
        'threshold': 'medium',
        'enabled': True,
        'description': 'Violence-related content'
    },
    'self_harm': {
        'threshold': 'medium',
        'enabled': True,
        'description': 'Self-harm related content'
    },
    'custom_filters': {
        'enabled': True,
        'description': 'Custom filters for organization-specific content'
    }
}
```

### Fine-tuning Capabilities

```python
# Example fine-tuning configuration
class AzureFineTuningService:
    """Service for managing Azure OpenAI fine-tuning"""
    
    async def create_fine_tuning_job(
        self,
        training_file_id: str,
        model: str = 'gpt-35-turbo',
        validation_file_id: Optional[str] = None,
        hyperparameters: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Create a fine-tuning job"""
        
        # Implementation would use Azure OpenAI fine-tuning APIs
        # Currently in preview - check Azure documentation for latest API
        
        job_config = {
            'model': model,
            'training_file': training_file_id,
            'validation_file': validation_file_id,
            'hyperparameters': hyperparameters or {
                'n_epochs': 3,
                'batch_size': 1,
                'learning_rate_multiplier': 0.1
            }
        }
        
        # Placeholder for actual fine-tuning API call
        logger.info("Fine-tuning job created", config=job_config)
        
        return {
            'job_id': 'ft-job-example-123',
            'status': 'pending',
            'config': job_config
        }
```

## Security and Compliance

### Production Security Configuration

```python
# Security best practices for Azure OpenAI
AZURE_SECURITY_CONFIG = {
    'authentication': {
        'method': 'azure_ad',  # Preferred over API keys
        'managed_identity': True,
        'rbac_roles': [
            'Cognitive Services OpenAI User',
            'Cognitive Services OpenAI Contributor'
        ]
    },
    'networking': {
        'private_endpoints': True,
        'vnet_integration': True,
        'firewall_rules': 'restrictive'
    },
    'data_residency': {
        'region': 'specific_region_required',
        'data_processing': 'in_region_only'
    },
    'compliance': {
        'certifications': [
            'SOC 2 Type 2',
            'ISO 27001',
            'HIPAA',
            'EU GDPR'
        ],
        'audit_logging': True
    }
}
```

### Monitoring and Observability

```python
# Azure-specific monitoring setup
class AzureOpenAIMonitoring:
    """Monitoring for Azure OpenAI usage"""
    
    def __init__(self):
        # Azure Monitor integration
        self.log_analytics_workspace = settings.azure_log_analytics_workspace
        self.application_insights = settings.azure_application_insights
    
    async def log_custom_metrics(
        self,
        model_name: str,
        response_time: float,
        tokens_used: int,
        cost: float,
        success: bool,
        content_filtered: bool = False
    ):
        """Log custom metrics to Azure Monitor"""
        
        metrics = {
            'ModelName': model_name,
            'ResponseTime': response_time,
            'TokensUsed': tokens_used,
            'Cost': cost,
            'Success': success,
            'ContentFiltered': content_filtered,
            'Timestamp': time.time()
        }
        
        # Send to Azure Monitor
        # Implementation would use Azure Monitor SDK
        logger.info("Azure OpenAI metrics logged", **metrics)
    
    async def create_alert_rules(self):
        """Create monitoring alert rules"""
        
        alert_rules = [
            {
                'name': 'High Error Rate',
                'condition': 'error_rate > 5%',
                'action': 'send_email_notification'
            },
            {
                'name': 'High Costs',
                'condition': 'daily_cost > $100',
                'action': 'send_teams_notification'
            },
            {
                'name': 'Rate Limit Exceeded',
                'condition': 'rate_limit_errors > 10',
                'action': 'scale_up_deployment'
            }
        ]
        
        for rule in alert_rules:
            logger.info("Alert rule configured", rule=rule)
```

## Summary

Azure OpenAI Service provides enterprise-grade AI capabilities with several key advantages:

### Strengths
- **Latest OpenAI Models** with enterprise SLAs and guaranteed availability
- **Enterprise Security** with advanced compliance and data residency options
- **Content Filtering** with customizable safety policies
- **Function Calling** for building AI agents and complex applications
- **Multimodal Capabilities** with GPT-4 Vision for image analysis
- **Fine-tuning Support** for custom model training (preview)

### Best Practices
1. **Model Selection:** Use GPT-4 Turbo for latest capabilities, GPT-3.5 Turbo for cost optimization
2. **Streaming:** Implement streaming for better user experience in chat applications
3. **Content Filtering:** Configure appropriate safety filters for your use case
4. **Cost Management:** Monitor usage and implement budgets and alerts
5. **Security:** Use Azure AD authentication and private endpoints for production

### Performance Characteristics
- **GPT-4 Turbo:** Best balance of capability and cost with 128K context
- **GPT-3.5 Turbo:** Fastest and most cost-effective for high-volume applications
- **GPT-4 Vision:** Powerful multimodal capabilities for image analysis
- **Streaming Support:** Low latency real-time responses

### Integration Benefits
- **Azure Ecosystem:** Native integration with Azure AI Search, Cognitive Services
- **Enterprise Features:** Advanced security, compliance, and monitoring
- **Global Scale:** Available in multiple regions with data residency options

This implementation provides a comprehensive foundation for Azure OpenAI integration that scales from development to enterprise production deployment.

---

*Next: Section 3 - Google Cloud Vertex AI Integration*