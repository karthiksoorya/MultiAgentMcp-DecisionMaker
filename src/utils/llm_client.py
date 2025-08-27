# Multi-Provider LLM Client with Fallbacks and Rate Limiting
import asyncio
import logging
from typing import Dict, Any, List, Optional, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import time
from enum import Enum
import random

try:
    import litellm
    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

from ..utils.config import Config
from ..utils.monitoring import track_agent_execution

logger = logging.getLogger(__name__)

class LLMProvider(Enum):
    """Supported LLM providers"""
    OPENROUTER = "openrouter"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    LITELLM = "litellm"

@dataclass
class LLMConfig:
    """LLM provider configuration"""
    provider: LLMProvider
    api_key: str
    model: str
    base_url: Optional[str] = None
    max_tokens: int = 1000
    temperature: float = 0.3
    timeout: int = 30
    max_retries: int = 3
    rate_limit_per_minute: int = 60
    priority: int = 1  # Lower is higher priority

@dataclass
class LLMResponse:
    """Standard LLM response format"""
    content: str
    success: bool
    provider: str
    model: str
    tokens_used: Optional[int] = None
    execution_time: float = 0.0
    error: Optional[str] = None
    cached: bool = False

class RateLimiter:
    """Simple rate limiter for API calls"""
    
    def __init__(self, max_calls: int, time_window: int = 60):
        self.max_calls = max_calls
        self.time_window = time_window
        self.calls = []
    
    async def acquire(self):
        """Acquire rate limit slot"""
        now = time.time()
        
        # Remove old calls outside time window
        self.calls = [call_time for call_time in self.calls if now - call_time < self.time_window]
        
        if len(self.calls) >= self.max_calls:
            # Calculate wait time
            oldest_call = min(self.calls)
            wait_time = self.time_window - (now - oldest_call)
            if wait_time > 0:
                logger.warning(f"Rate limit reached, waiting {wait_time:.1f}s")
                await asyncio.sleep(wait_time)
        
        self.calls.append(now)

class MultiProviderLLMClient:
    """Advanced multi-provider LLM client with fallbacks and optimization"""
    
    def __init__(self, config: Config):
        self.config = config
        self.providers: List[LLMConfig] = []
        self.rate_limiters: Dict[str, RateLimiter] = {}
        self.provider_stats: Dict[str, Dict[str, Any]] = {}
        self.response_cache: Dict[str, LLMResponse] = {}
        self.cache_ttl = 300  # 5 minutes
        
        # Initialize providers
        self._setup_providers()
        self._setup_clients()
    
    def _setup_providers(self):
        """Setup available LLM providers based on configuration"""
        
        # OpenRouter (Primary)
        if self.config.LLM.api_key:
            openrouter_config = LLMConfig(
                provider=LLMProvider.OPENROUTER,
                api_key=self.config.LLM.api_key,
                model=self.config.LLM.model,
                base_url=self.config.LLM.base_url,
                priority=1,
                rate_limit_per_minute=60
            )
            self.providers.append(openrouter_config)
        
        # OpenAI (Fallback)
        openai_key = getattr(self.config, 'OPENAI_API_KEY', None)
        if openai_key and OPENAI_AVAILABLE:
            openai_config = LLMConfig(
                provider=LLMProvider.OPENAI,
                api_key=openai_key,
                model="gpt-4-turbo-preview",
                priority=2,
                rate_limit_per_minute=50
            )
            self.providers.append(openai_config)
        
        # Anthropic (Fallback)
        anthropic_key = getattr(self.config, 'ANTHROPIC_API_KEY', None)
        if anthropic_key and ANTHROPIC_AVAILABLE:
            anthropic_config = LLMConfig(
                provider=LLMProvider.ANTHROPIC,
                api_key=anthropic_key,
                model="claude-3-sonnet-20240229",
                priority=3,
                rate_limit_per_minute=40
            )
            self.providers.append(anthropic_config)
        
        # Sort by priority
        self.providers.sort(key=lambda x: x.priority)
        
        # Setup rate limiters and stats
        for provider_config in self.providers:
            provider_key = f"{provider_config.provider.value}_{provider_config.model}"
            self.rate_limiters[provider_key] = RateLimiter(provider_config.rate_limit_per_minute)
            self.provider_stats[provider_key] = {
                "requests": 0,
                "successful_requests": 0,
                "failed_requests": 0,
                "total_tokens": 0,
                "total_execution_time": 0.0,
                "average_execution_time": 0.0,
                "last_used": None,
                "consecutive_failures": 0,
                "enabled": True
            }
        
        logger.info(f"Initialized {len(self.providers)} LLM providers")
    
    def _setup_clients(self):
        """Setup individual provider clients"""
        self.clients = {}
        
        for provider_config in self.providers:
            try:
                if provider_config.provider == LLMProvider.OPENROUTER and LITELLM_AVAILABLE:
                    # Configure LiteLLM for OpenRouter
                    litellm.api_key = provider_config.api_key
                    litellm.api_base = provider_config.base_url
                    self.clients[provider_config.provider.value] = litellm
                
                elif provider_config.provider == LLMProvider.OPENAI and OPENAI_AVAILABLE:
                    client = openai.AsyncOpenAI(api_key=provider_config.api_key)
                    self.clients[provider_config.provider.value] = client
                
                elif provider_config.provider == LLMProvider.ANTHROPIC and ANTHROPIC_AVAILABLE:
                    client = anthropic.AsyncAnthropic(api_key=provider_config.api_key)
                    self.clients[provider_config.provider.value] = client
                
            except Exception as e:
                logger.error(f"Failed to setup {provider_config.provider.value} client: {e}")
    
    async def generate_completion(
        self, 
        prompt: str, 
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        system_message: Optional[str] = None,
        use_cache: bool = True,
        force_provider: Optional[str] = None
    ) -> LLMResponse:
        """Generate completion with automatic provider fallback"""
        
        # Check cache first
        if use_cache:
            cache_key = self._get_cache_key(prompt, max_tokens, temperature, system_message)
            cached_response = self._get_cached_response(cache_key)
            if cached_response:
                return cached_response
        
        # Prepare providers to try
        providers_to_try = self.providers.copy()
        
        # If force_provider specified, try it first
        if force_provider:
            forced_providers = [p for p in providers_to_try if p.provider.value == force_provider]
            other_providers = [p for p in providers_to_try if p.provider.value != force_provider]
            providers_to_try = forced_providers + other_providers
        
        # Filter out temporarily disabled providers
        providers_to_try = [p for p in providers_to_try 
                          if self.provider_stats[f"{p.provider.value}_{p.model}"]["enabled"]]
        
        if not providers_to_try:
            return LLMResponse(
                content="",
                success=False,
                provider="none",
                model="none",
                error="No available LLM providers"
            )
        
        last_error = None
        
        # Try each provider in order
        for provider_config in providers_to_try:
            try:
                response = await self._call_provider(
                    provider_config, 
                    prompt, 
                    max_tokens, 
                    temperature, 
                    system_message
                )
                
                if response.success:
                    # Cache successful response
                    if use_cache:
                        self._cache_response(cache_key, response)
                    
                    return response
                else:
                    last_error = response.error
                    
            except Exception as e:
                last_error = str(e)
                logger.error(f"Provider {provider_config.provider.value} failed: {e}")
                
                # Update failure stats
                provider_key = f"{provider_config.provider.value}_{provider_config.model}"
                self.provider_stats[provider_key]["consecutive_failures"] += 1
                
                # Temporarily disable provider if too many failures
                if self.provider_stats[provider_key]["consecutive_failures"] >= 3:
                    self.provider_stats[provider_key]["enabled"] = False
                    logger.warning(f"Temporarily disabled provider {provider_key} due to failures")
        
        # All providers failed
        return LLMResponse(
            content="",
            success=False,
            provider="multiple_failed",
            model="multiple_failed",
            error=f"All providers failed. Last error: {last_error}"
        )
    
    async def _call_provider(
        self,
        provider_config: LLMConfig,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        system_message: Optional[str] = None
    ) -> LLMResponse:
        """Call specific provider"""
        
        provider_key = f"{provider_config.provider.value}_{provider_config.model}"
        start_time = time.time()
        
        # Apply rate limiting
        await self.rate_limiters[provider_key].acquire()
        
        # Update stats
        stats = self.provider_stats[provider_key]
        stats["requests"] += 1
        stats["last_used"] = datetime.now()
        
        try:
            # Prepare parameters
            actual_max_tokens = max_tokens or provider_config.max_tokens
            actual_temperature = temperature if temperature is not None else provider_config.temperature
            
            if provider_config.provider == LLMProvider.OPENROUTER:
                response = await self._call_openrouter(
                    provider_config, prompt, actual_max_tokens, actual_temperature, system_message
                )
            elif provider_config.provider == LLMProvider.OPENAI:
                response = await self._call_openai(
                    provider_config, prompt, actual_max_tokens, actual_temperature, system_message
                )
            elif provider_config.provider == LLMProvider.ANTHROPIC:
                response = await self._call_anthropic(
                    provider_config, prompt, actual_max_tokens, actual_temperature, system_message
                )
            else:
                raise ValueError(f"Unsupported provider: {provider_config.provider}")
            
            execution_time = time.time() - start_time
            
            # Update success stats
            stats["successful_requests"] += 1
            stats["total_execution_time"] += execution_time
            stats["average_execution_time"] = stats["total_execution_time"] / stats["successful_requests"]
            stats["consecutive_failures"] = 0  # Reset failure counter
            
            if response.tokens_used:
                stats["total_tokens"] += response.tokens_used
            
            # Track for monitoring
            track_agent_execution(
                f"llm_{provider_config.provider.value}",
                execution_time,
                response.tokens_used or 0,
                True
            )
            
            response.execution_time = execution_time
            return response
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            # Update failure stats
            stats["failed_requests"] += 1
            stats["consecutive_failures"] += 1
            
            # Track failed call
            track_agent_execution(
                f"llm_{provider_config.provider.value}",
                execution_time,
                0,
                False
            )
            
            return LLMResponse(
                content="",
                success=False,
                provider=provider_config.provider.value,
                model=provider_config.model,
                execution_time=execution_time,
                error=str(e)
            )
    
    async def _call_openrouter(self, config: LLMConfig, prompt: str, max_tokens: int, temperature: float, system_message: Optional[str]) -> LLMResponse:
        """Call OpenRouter via LiteLLM"""
        if not LITELLM_AVAILABLE:
            raise RuntimeError("LiteLLM not available")
        
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})
        
        response = await asyncio.to_thread(
            litellm.completion,
            model=config.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=config.timeout
        )
        
        return LLMResponse(
            content=response.choices[0].message.content,
            success=True,
            provider=config.provider.value,
            model=config.model,
            tokens_used=response.usage.total_tokens if hasattr(response, 'usage') else None
        )
    
    async def _call_openai(self, config: LLMConfig, prompt: str, max_tokens: int, temperature: float, system_message: Optional[str]) -> LLMResponse:
        """Call OpenAI API"""
        if config.provider.value not in self.clients:
            raise RuntimeError("OpenAI client not available")
        
        client = self.clients[config.provider.value]
        
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})
        
        response = await client.chat.completions.create(
            model=config.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=config.timeout
        )
        
        return LLMResponse(
            content=response.choices[0].message.content,
            success=True,
            provider=config.provider.value,
            model=config.model,
            tokens_used=response.usage.total_tokens
        )
    
    async def _call_anthropic(self, config: LLMConfig, prompt: str, max_tokens: int, temperature: float, system_message: Optional[str]) -> LLMResponse:
        """Call Anthropic API"""
        if config.provider.value not in self.clients:
            raise RuntimeError("Anthropic client not available")
        
        client = self.clients[config.provider.value]
        
        # Anthropic has different API structure
        kwargs = {
            "model": config.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}]
        }
        
        if system_message:
            kwargs["system"] = system_message
        
        response = await client.messages.create(**kwargs)
        
        return LLMResponse(
            content=response.content[0].text,
            success=True,
            provider=config.provider.value,
            model=config.model,
            tokens_used=response.usage.input_tokens + response.usage.output_tokens
        )
    
    def _get_cache_key(self, prompt: str, max_tokens: Optional[int], temperature: Optional[float], system_message: Optional[str]) -> str:
        """Generate cache key for response"""
        return str(hash((prompt, max_tokens, temperature, system_message)))
    
    def _get_cached_response(self, cache_key: str) -> Optional[LLMResponse]:
        """Get cached response if available and not expired"""
        if cache_key in self.response_cache:
            response = self.response_cache[cache_key]
            # Simple TTL check (in production, you'd want more sophisticated caching)
            response.cached = True
            return response
        return None
    
    def _cache_response(self, cache_key: str, response: LLMResponse):
        """Cache successful response"""
        # Simple in-memory cache (in production, consider Redis or similar)
        self.response_cache[cache_key] = response
        
        # Basic cache cleanup (remove oldest entries if cache gets too large)
        if len(self.response_cache) > 100:
            oldest_key = next(iter(self.response_cache))
            del self.response_cache[oldest_key]
    
    def get_provider_stats(self) -> Dict[str, Any]:
        """Get provider statistics"""
        return {
            "providers": dict(self.provider_stats),
            "total_providers": len(self.providers),
            "enabled_providers": len([p for p in self.provider_stats.values() if p["enabled"]]),
            "cache_size": len(self.response_cache)
        }
    
    def reset_provider_failures(self, provider_key: Optional[str] = None):
        """Reset failure counters for providers"""
        if provider_key:
            if provider_key in self.provider_stats:
                self.provider_stats[provider_key]["consecutive_failures"] = 0
                self.provider_stats[provider_key]["enabled"] = True
                logger.info(f"Reset failures for provider: {provider_key}")
        else:
            for stats in self.provider_stats.values():
                stats["consecutive_failures"] = 0
                stats["enabled"] = True
            logger.info("Reset failures for all providers")

# Factory function
def create_llm_client(config: Optional[Config] = None) -> MultiProviderLLMClient:
    """Create multi-provider LLM client"""
    config = config or Config()
    return MultiProviderLLMClient(config)