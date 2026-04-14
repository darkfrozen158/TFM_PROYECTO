"""
rag_service.py
==============
Integración RAG (Retrieval-Augmented Generation) con Azure AI Search.
Busca contexto relevante en la base de conocimientos del banco:
  - Preguntas frecuentes de productos
  - Términos y condiciones
  - Tasas y tarifas vigentes
  - Procedimientos y políticas
"""

import logging
import os
from dataclasses import dataclass

from azure.core.credentials import AzureKeyCredential
from azure.search.documents.aio import SearchClient
from azure.search.documents.models import VectorizedQuery
from openai import AsyncAzureOpenAI

logger = logging.getLogger("rag_service")


@dataclass
class RAGResult:
    """Resultado de una búsqueda RAG."""
    content: str
    score: float
    source: str
    chunk_id: str


class BankingRAGService:
    """
    Servicio RAG bancario usando Azure AI Search con búsqueda híbrida
    (semántica + vectorial + keyword fusion).
    """

    def __init__(self):
        self._search_client: SearchClient | None = None
        self._openai_client: AsyncAzureOpenAI | None = None

        self.search_endpoint = os.environ["AZURE_SEARCH_ENDPOINT"]
        self.search_key = os.environ["AZURE_SEARCH_API_KEY"]
        self.index_name = os.environ.get("AZURE_SEARCH_INDEX", "banking-knowledge-base")
        self.embedding_deployment = os.environ["AZURE_EMBEDDING_DEPLOYMENT"]

    def _get_search_client(self) -> SearchClient:
        """Lazy init del cliente de búsqueda."""
        if self._search_client is None:
            self._search_client = SearchClient(
                endpoint=self.search_endpoint,
                index_name=self.index_name,
                credential=AzureKeyCredential(self.search_key),
            )
        return self._search_client

    def _get_openai_client(self) -> AsyncAzureOpenAI:
        """Lazy init del cliente OpenAI para embeddings."""
        if self._openai_client is None:
            self._openai_client = AsyncAzureOpenAI(
                azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
                api_key=os.environ["AZURE_OPENAI_API_KEY"],
                api_version="2024-02-01",
            )
        return self._openai_client

    async def _generate_embedding(self, text: str) -> list[float]:
        """Genera embedding vectorial del texto de consulta."""
        client = self._get_openai_client()
        response = await client.embeddings.create(
            model=self.embedding_deployment,
            input=text,
        )
        return response.data[0].embedding

    async def search(
        self,
        query: str,
        customer_id: str,
        top_k: int = 3,
        min_score: float = 0.6,
    ) -> str:
        """
        Realiza búsqueda híbrida (vectorial + semántica) en Azure AI Search.

        Args:
            query:       Consulta del cliente en lenguaje natural.
            customer_id: ID del cliente (para filtros de segmento si aplica).
            top_k:       Número de resultados a recuperar.
            min_score:   Score mínimo de relevancia para incluir resultado.

        Returns:
            Contexto formateado para incluir en el prompt del LLM.
        """
        try:
            # Generar embedding de la consulta
            query_vector = await self._generate_embedding(query)

            # Configurar búsqueda vectorial
            vector_query = VectorizedQuery(
                vector=query_vector,
                k_nearest_neighbors=top_k,
                fields="content_vector",
            )

            client = self._get_search_client()

            # Búsqueda híbrida: vectorial + keyword con re-ranking semántico
            results = await client.search(
                search_text=query,
                vector_queries=[vector_query],
                select=["id", "title", "content", "source", "product_type", "chunk_id"],
                query_type="semantic",
                semantic_configuration_name="banking-semantic-config",
                top=top_k,
                query_caption="extractive",
                query_answer="extractive",
            )

            rag_results: list[RAGResult] = []

            async for result in results:
                score = result.get("@search.reranker_score", 0.0) or result.get(
                    "@search.score", 0.0
                )
                if score < min_score:
                    continue

                rag_results.append(
                    RAGResult(
                        content=result.get("content", ""),
                        score=score,
                        source=result.get("source", "knowledge_base"),
                        chunk_id=result.get("chunk_id", result.get("id", "")),
                    )
                )

            if not rag_results:
                logger.debug("RAG sin resultados relevantes para query='%s'", query[:50])
                return ""

            # Formatear contexto para el LLM
            context_parts = ["--- Información de la base de conocimiento del banco ---"]
            for i, res in enumerate(rag_results, 1):
                context_parts.append(
                    f"\n[Fuente {i} | Relevancia: {res.score:.2f}]\n{res.content.strip()}"
                )
            context_parts.append("--- Fin del contexto ---")

            logger.info(
                "RAG: %d fragmentos recuperados para customer_id=%s",
                len(rag_results),
                customer_id,
            )
            return "\n".join(context_parts)

        except Exception as exc:
            logger.exception(
                "Error en RAG para customer_id=%s, query='%s'", customer_id, query[:50]
            )
            return ""

    async def index_document(
        self,
        doc_id: str,
        title: str,
        content: str,
        source: str,
        product_type: str,
    ) -> bool:
        """
        Indexa un documento en Azure AI Search.
        Úsalo para actualizar la base de conocimiento del banco.

        Args:
            doc_id:       ID único del documento.
            title:        Título del documento.
            content:      Contenido del chunk.
            source:       Fuente (ej: 'reglamento_tarjeta.pdf').
            product_type: Tipo de producto bancario asociado.

        Returns:
            True si se indexó correctamente.
        """
        try:
            content_vector = await self._generate_embedding(content)
            client = self._get_search_client()

            document = {
                "id": doc_id,
                "title": title,
                "content": content,
                "source": source,
                "product_type": product_type,
                "content_vector": content_vector,
                "chunk_id": doc_id,
            }

            await client.upload_documents(documents=[document])
            logger.info("Documento indexado: id=%s, source=%s", doc_id, source)
            return True

        except Exception as exc:
            logger.exception("Error indexando documento id=%s", doc_id)
            return False

    async def close(self):
        """Cierra conexiones de forma limpia."""
        if self._search_client:
            await self._search_client.close()
        if self._openai_client:
            await self._openai_client.close()
