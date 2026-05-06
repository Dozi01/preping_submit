"""
Embedding Model Module

Provides embedding clients for semantic similarity search.
Supports OpenAI and LiteLLM embedding models.
"""

from typing import List, Optional

from openai import OpenAI


def get_api_key(provider: str) -> Optional[str]:
    """Get API key from environment variable."""
    import os
    key_map = {
        'openai': 'OPENAI_API_KEY',
        'vllm': 'VLLM_API_KEY',
    }
    env_var = key_map.get(provider.lower(), f"{provider.upper()}_API_KEY")
    return os.environ.get(env_var)


class EmbeddingClient:
    """
    Simple embedding client for semantic similarity search.
    
    Supports OpenAI, vLLM, and LiteLLM embedding models.
    
    Usage:
        # OpenAI
        client = EmbeddingClient(model="text-embedding-3-small")
        
        # vLLM server (OpenAI-compatible)
        client = EmbeddingClient(
            model="Qwen/Qwen3-Embedding-0.6B",
            base_url="http://localhost:8201/v1"
        )
        
        # Other providers via LiteLLM  
        client = EmbeddingClient(model="voyage/voyage-3", use_litellm=True)
        
        # Get embedding
        embedding = client.embed("Hello world")
    """
    
    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        use_litellm: bool = False,
    ):
        """
        Initialize the embedding client.
        
        Args:
            model: Embedding model name.
                - OpenAI: "text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002"
                - vLLM: Any model served via vLLM (e.g., "Qwen/Qwen3-Embedding-0.6B")
                - LiteLLM: "voyage/voyage-3", "cohere/embed-english-v3.0", etc.
            api_key: Optional API key. If None, uses environment variable.
                For vLLM, can be set to "EMPTY" or any string if server doesn't require auth.
            base_url: Optional base URL for OpenAI-compatible servers (e.g., vLLM).
                Example: "http://localhost:8201/v1"
            use_litellm: If True, use LiteLLM for embedding (supports multiple providers).
        """
        self.model = model
        self.use_litellm = use_litellm
        self.base_url = base_url
        
        if use_litellm:
            import litellm
            self._litellm = litellm
            self._api_key = api_key
        else:
            # For vLLM or other OpenAI-compatible servers, use provided base_url
            # api_key can be "EMPTY" for local servers without auth
            if base_url is None:
                api_key = get_api_key('openai')
            else:
                api_key = api_key or get_api_key('vllm')
            self._client = OpenAI(
                api_key=api_key,
                base_url=base_url,
            )
    
    def embed(self, text: str) -> List[float]:
        """
        Get embedding vector for a text string.
        
        Args:
            text: Text to embed.
        
        Returns:
            List of floats representing the embedding vector.
        """
        return self.embed_batch([text])[0]
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Get embeddings for multiple texts in a single API call.
        
        Args:
            texts: List of texts to embed.
        
        Returns:
            List of embedding vectors.
        """
        if self.use_litellm:
            response = self._litellm.embedding(
                model=self.model,
                input=texts,
                api_key=self._api_key,
            )
            return [item['embedding'] for item in response.data]
        else:
            response = self._client.embeddings.create(
                model=self.model,
                input=texts,
            )
            return [item.embedding for item in response.data]


__all__ = [
    'EmbeddingClient',
    'get_api_key',
]
