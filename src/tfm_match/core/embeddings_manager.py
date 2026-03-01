"""
Embeddings Manager - Genera embeddings usando OpenAI.
Código extraído de api/main.py (líneas 120-124).
"""

from typing import List
from openai import OpenAI

from tfm_match.gold.text_sanitizer import sanitize_text


class EmbeddingsManager:
    """Maneja la generación de embeddings usando OpenAI API."""
    
    def __init__(self, openai_client: OpenAI, model: str):
        """
        Args:
            openai_client: Cliente OpenAI configurado
            model: Modelo de embeddings (ej: 'text-embedding-3-small')
        """
        self.client = openai_client
        self.model = model
    
    def embed_text(self, q: str) -> List[float]:
        """
        Genera embedding para un texto.
        LÓGICA EXACTA de la función embed_text original.
        
        Args:
            q: Texto a embedizar
            
        Returns:
            Vector de embeddings
        """
        q = sanitize_text(q)
        return self.client.embeddings.create(
            model=self.model,
            input=[q]
        ).data[0].embedding
