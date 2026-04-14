"""
AI Banking Agent - Semantic Kernel + Azure
==========================================
Agente bancario inteligente con:
  - Semantic Kernel para orquestación
  - Azure OpenAI (GPT-4o) como LLM
  - Azure AI Search para RAG
  - Function calling automático
  - Azure Content Safety para moderación
  - Redis Cache for Azure para memoria eficiente
"""

import asyncio
import json
import logging
import os
from typing import Annotated, Any
from uuid import uuid4

from dotenv import load_dotenv
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion, AzureTextEmbedding
from semantic_kernel.contents import ChatHistory
from semantic_kernel.core_plugins import TimePlugin
from semantic_kernel.functions import kernel_function

from banking_plugins import AccountPlugin, CardPlugin, LoanPlugin, TransferPlugin
from cache_manager import RedisCacheManager
from content_safety import ContentSafetyFilter
from rag_service import BankingRAGService
from security import SecurityContext, validate_jwt_token

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("banking_agent")


# ─────────────────────────────────────────────
# KERNEL FACTORY
# ─────────────────────────────────────────────

def build_kernel() -> Kernel:
    """Construye el Kernel de Semantic Kernel con todos los servicios Azure."""
    kernel = Kernel()

    # Azure OpenAI Chat
    kernel.add_service(
        AzureChatCompletion(
            deployment_name=os.environ["AZURE_OPENAI_DEPLOYMENT"],
            endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            service_id="chat",
        )
    )

    # Azure OpenAI Embeddings (para RAG)
    kernel.add_service(
        AzureTextEmbedding(
            deployment_name=os.environ["AZURE_EMBEDDING_DEPLOYMENT"],
            endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            service_id="embeddings",
        )
    )

    # Plugins bancarios
    kernel.add_plugin(AccountPlugin(), plugin_name="Accounts")
    kernel.add_plugin(CardPlugin(), plugin_name="Cards")
    kernel.add_plugin(LoanPlugin(), plugin_name="Loans")
    kernel.add_plugin(TransferPlugin(), plugin_name="Transfers")
    kernel.add_plugin(TimePlugin(), plugin_name="Time")

    return kernel


# ─────────────────────────────────────────────
# BANKING AGENT
# ─────────────────────────────────────────────

class BankingAgent:
    """
    Agente bancario principal.
    Gestiona el ciclo completo: seguridad → caché → RAG → LLM → function calling.
    """

    SYSTEM_PROMPT = """Eres un asistente bancario virtual inteligente del banco.
Ayudas a los clientes con consultas sobre sus productos digitales:
  - Tarjetas de crédito digitales
  - Cuentas en soles y dólares
  - Préstamos digitales
  - Transferencias y operaciones

Reglas estrictas:
1. NUNCA inventes saldos, tasas ni datos financieros. Usa SIEMPRE las funciones disponibles.
2. Si el cliente pide operaciones (transferencias, pagos), confirma antes de ejecutar.
3. Responde SIEMPRE en el mismo idioma que usa el cliente.
4. Para datos sensibles, muestra solo los últimos 4 dígitos de cuentas/tarjetas.
5. Si detectas actividad sospechosa, escala al equipo humano.
6. Sé conciso, preciso y empático.

Contexto adicional de la base de conocimiento del banco:
{rag_context}
"""

    def __init__(self):
        self.kernel = build_kernel()
        self.cache = RedisCacheManager()
        self.rag = BankingRAGService()
        self.safety = ContentSafetyFilter()
        logger.info("BankingAgent inicializado correctamente.")

    async def chat(
        self,
        user_message: str,
        session_id: str,
        jwt_token: str,
        customer_id: str,
    ) -> dict[str, Any]:
        """
        Punto de entrada principal para una interacción del cliente.

        Args:
            user_message: Mensaje en lenguaje natural del cliente.
            session_id:   ID de sesión del cliente en la app móvil.
            jwt_token:    Token JWT del cliente autenticado.
            customer_id:  ID del cliente en el core bancario.

        Returns:
            dict con 'response', 'session_id', 'functions_called' y 'sources'.
        """

        # ── 1. VALIDAR SEGURIDAD ────────────────────────────────────────────
        security_ctx = validate_jwt_token(jwt_token, customer_id)
        if not security_ctx.is_valid:
            logger.warning("Token JWT inválido para customer_id=%s", customer_id)
            return {
                "response": "Tu sesión ha expirado. Por favor, vuelve a iniciar sesión.",
                "session_id": session_id,
                "error": "UNAUTHORIZED",
            }

        # ── 2. FILTRO DE CONTENIDO (INPUT) ──────────────────────────────────
        safety_result = await self.safety.analyze_input(user_message, customer_id)
        if safety_result.is_blocked:
            logger.warning(
                "Contenido bloqueado para customer_id=%s, razón=%s",
                customer_id,
                safety_result.reason,
            )
            return {
                "response": "Lo siento, no puedo procesar esa solicitud. Si necesitas ayuda, contacta a nuestro soporte.",
                "session_id": session_id,
                "error": "CONTENT_BLOCKED",
            }

        # ── 3. RECUPERAR HISTORIAL DESDE REDIS ──────────────────────────────
        history = await self.cache.get_chat_history(session_id)
        if history is None:
            history = ChatHistory()
            logger.info("Nueva sesión creada: %s", session_id)

        # ── 4. BÚSQUEDA RAG (Azure AI Search) ────────────────────────────────
        rag_context = await self.rag.search(
            query=user_message,
            customer_id=customer_id,
            top_k=3,
        )

        # ── 5. CONSTRUIR PROMPT CON CONTEXTO RAG ─────────────────────────────
        system_with_context = self.SYSTEM_PROMPT.format(
            rag_context=rag_context or "Sin contexto adicional disponible."
        )

        if not history.messages:
            history.add_system_message(system_with_context)

        history.add_user_message(user_message)

        # ── 6. INVOCAR LLM CON FUNCTION CALLING AUTOMÁTICO ──────────────────
        chat_service = self.kernel.get_service("chat")
        settings = chat_service.get_prompt_execution_settings_class()(
            service_id="chat",
            max_tokens=1500,
            temperature=0.2,
            function_choice_behavior=FunctionChoiceBehavior.Auto(
                filters={"excluded_plugins": ["Time"]}
            ),
        )

        functions_called: list[str] = []
        response_text = ""

        try:
            result = await self.kernel.invoke_prompt(
                prompt="{{$history}}",
                arguments={"history": history},
                service_id="chat",
                prompt_execution_settings=settings,
            )

            response_text = str(result)

            # Extraer funciones ejecutadas del metadata
            if hasattr(result, "metadata") and result.metadata:
                for item in result.metadata.get("function_calls", []):
                    functions_called.append(item.get("name", ""))

        except Exception as exc:
            logger.exception("Error invocando el LLM para session_id=%s", session_id)
            response_text = (
                "Lo siento, ocurrió un error procesando tu consulta. "
                "Por favor intenta nuevamente."
            )

        # ── 7. FILTRO DE CONTENIDO (OUTPUT) ─────────────────────────────────
        output_safety = await self.safety.analyze_output(response_text, customer_id)
        if output_safety.is_blocked:
            logger.error(
                "Respuesta bloqueada en output safety para customer_id=%s", customer_id
            )
            response_text = (
                "No puedo proporcionar esa información. "
                "Contáctanos en nuestra línea de atención al cliente."
            )

        # ── 8. ACTUALIZAR HISTORIAL EN REDIS ────────────────────────────────
        history.add_assistant_message(response_text)
        await self.cache.save_chat_history(session_id, history, ttl_seconds=1800)

        logger.info(
            "Respuesta generada | session=%s | customer=%s | functions=%s",
            session_id,
            customer_id,
            functions_called,
        )

        return {
            "response": response_text,
            "session_id": session_id,
            "functions_called": functions_called,
            "sources": rag_context[:200] if rag_context else None,
        }

    async def close(self):
        """Cierra conexiones de forma limpia."""
        await self.cache.close()
        logger.info("BankingAgent cerrado correctamente.")


# ─────────────────────────────────────────────
# DEMO LOCAL
# ─────────────────────────────────────────────

async def _demo():
    agent = BankingAgent()
    session = str(uuid4())

    queries = [
        "¿Cuál es el saldo de mi cuenta en soles?",
        "¿Qué tarjetas de crédito digitales tengo disponibles?",
        "¿Cuánto debo en mi préstamo personal?",
    ]

    for query in queries:
        print(f"\n>>> Cliente: {query}")
        result = await agent.chat(
            user_message=query,
            session_id=session,
            jwt_token="DEMO_TOKEN",
            customer_id="DEMO_CUSTOMER_001",
        )
        print(f"<<< Agente: {result['response']}")
        if result.get("functions_called"):
            print(f"    [Funciones ejecutadas: {result['functions_called']}]")

    await agent.close()


if __name__ == "__main__":
    asyncio.run(_demo())
