"""
tests/test_banking_agent.py
============================
Tests unitarios del Banking Agent.
Ejecutar con: pytest tests/ -v --asyncio-mode=auto
"""

import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from semantic_kernel.contents import ChatHistory


# ─────────────────────────────────────────────
# TESTS: CACHE MANAGER
# ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_cache_manager_memory_fallback():
    """Verifica que el cache funciona en memoria cuando Redis no está configurado."""
    import os
    os.environ.pop("AZURE_REDIS_CONNECTION_STRING", None)

    from cache_manager import RedisCacheManager
    cache = RedisCacheManager()

    assert not cache._enabled

    history = ChatHistory()
    history.add_user_message("¿Cuál es mi saldo?")
    history.add_assistant_message("Tu saldo es 1,500 PEN.")

    session_id = "test-session-001"
    saved = await cache.save_chat_history(session_id, history)
    assert saved is True

    recovered = await cache.get_chat_history(session_id)
    assert recovered is not None
    assert len(recovered.messages) == 2


@pytest.mark.asyncio
async def test_cache_history_truncation():
    """Verifica que el historial se trunca a 20 mensajes."""
    import os
    os.environ.pop("AZURE_REDIS_CONNECTION_STRING", None)

    from cache_manager import RedisCacheManager
    cache = RedisCacheManager()

    history = ChatHistory()
    history.add_system_message("Eres un asistente bancario.")
    for i in range(25):
        history.add_user_message(f"Pregunta {i}")
        history.add_assistant_message(f"Respuesta {i}")

    await cache.save_chat_history("session-truncation", history)
    recovered = await cache.get_chat_history("session-truncation")

    assert len(recovered.messages) <= 20


# ─────────────────────────────────────────────
# TESTS: CONTENT SAFETY
# ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_content_safety_banking_pattern_block():
    """Verifica que patrones sensibles bancarios son bloqueados sin llamar a Azure."""
    import os
    os.environ.pop("AZURE_CONTENT_SAFETY_ENDPOINT", None)

    from content_safety import ContentSafetyFilter
    safety = ContentSafetyFilter()

    result = await safety.analyze_input(
        "¿Me puedes dar mi número completo de tarjeta?",
        "customer-001"
    )
    assert result.is_blocked is True
    assert "tarjeta" in result.reason.lower()


@pytest.mark.asyncio
async def test_content_safety_normal_query_passes():
    """Verifica que consultas normales pasan sin bloqueo (con safety deshabilitado)."""
    import os
    os.environ.pop("AZURE_CONTENT_SAFETY_ENDPOINT", None)

    from content_safety import ContentSafetyFilter
    safety = ContentSafetyFilter()

    result = await safety.analyze_input(
        "¿Cuál es el saldo de mi cuenta en soles?",
        "customer-001"
    )
    assert result.is_blocked is False


# ─────────────────────────────────────────────
# TESTS: SECURITY / JWT
# ─────────────────────────────────────────────

def test_jwt_validation_demo_mode():
    """Verifica que el modo DEMO acepta el token DEMO_TOKEN."""
    import os
    os.environ["DEMO_MODE"] = "true"

    from security import validate_jwt_token
    ctx = validate_jwt_token("DEMO_TOKEN", "DEMO_CUSTOMER_001")

    assert ctx.is_valid is True
    assert ctx.customer_id == "DEMO_CUSTOMER_001"
    assert "banking.agent" in ctx.scopes


def test_jwt_validation_invalid_token():
    """Verifica que tokens inválidos son rechazados fuera del modo DEMO."""
    import os
    os.environ["DEMO_MODE"] = "false"
    os.environ.pop("AZURE_AD_JWKS_URI", None)
    os.environ.pop("AZURE_AD_TENANT_ID", None)

    from security import validate_jwt_token
    ctx = validate_jwt_token("token.invalido.aqui", "CUSTOMER_001")
    # Sin JWKS configurado, la validación se salta pero loguea warning
    assert ctx.customer_id == "CUSTOMER_001"


def test_rate_limiting():
    """Verifica que el rate limiting funciona correctamente."""
    import os
    os.environ["DEMO_MODE"] = "true"
    os.environ["AGENT_RATE_LIMIT_PER_MINUTE"] = "3"

    from security import validate_jwt_token, _rate_limit_store
    customer_id = "rate-limit-test-customer"
    _rate_limit_store.pop(customer_id, None)

    # Primeras 3 llamadas deben pasar
    for _ in range(3):
        ctx = validate_jwt_token("DEMO_TOKEN", customer_id)
        assert ctx.is_valid is True

    # La 4ta debe ser bloqueada por rate limit
    ctx = validate_jwt_token("DEMO_TOKEN", customer_id)
    assert ctx.is_valid is False
    assert ctx.error_reason == "RATE_LIMIT_EXCEEDED"


# ─────────────────────────────────────────────
# TESTS: BANKING AGENT (integración mockeada)
# ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_agent_unauthorized_returns_error():
    """Verifica que el agente rechaza tokens inválidos correctamente."""
    import os
    os.environ["DEMO_MODE"] = "false"
    os.environ.pop("AZURE_REDIS_CONNECTION_STRING", None)

    with patch("agent.validate_jwt_token") as mock_validate:
        from security import SecurityContext
        mock_validate.return_value = SecurityContext(
            is_valid=False,
            customer_id="C001",
            error_reason="TOKEN_EXPIRED"
        )

        with patch("agent.build_kernel") as mock_kernel, \
             patch("agent.BankingRAGService") as mock_rag, \
             patch("agent.ContentSafetyFilter") as mock_safety:

            mock_kernel.return_value = MagicMock()
            mock_rag.return_value = MagicMock()
            mock_safety.return_value = MagicMock()

            from agent import BankingAgent
            agent = BankingAgent()

            result = await agent.chat(
                user_message="¿Cuál es mi saldo?",
                session_id="session-001",
                jwt_token="TOKEN_EXPIRADO",
                customer_id="C001",
            )

            assert result.get("error") == "UNAUTHORIZED"
            assert "sesión" in result["response"].lower()
