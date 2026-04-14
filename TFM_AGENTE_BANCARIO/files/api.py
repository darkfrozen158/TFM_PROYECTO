"""
api.py
======
FastAPI endpoint que expone el Banking Agent como REST API.
Se integra con la app de Banca Móvil existente.

Endpoints:
  POST /api/v1/agent/chat     → Enviar mensaje al agente
  DELETE /api/v1/agent/session/{session_id} → Cerrar sesión
  GET  /api/v1/agent/health   → Health check
"""

import logging
import os
from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from agent import BankingAgent

logger = logging.getLogger("banking_api")

# Instancia del agente (singleton)
_agent: BankingAgent | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Ciclo de vida de la aplicación: inicializa y cierra el agente."""
    global _agent
    logger.info("Iniciando Banking Agent...")
    _agent = BankingAgent()
    logger.info("Banking Agent listo.")
    yield
    if _agent:
        await _agent.close()
    logger.info("Banking Agent cerrado.")


app = FastAPI(
    title="Banking AI Agent API",
    description="Agente bancario inteligente con Semantic Kernel + Azure",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs" if os.getenv("ENVIRONMENT") != "production" else None,
    redoc_url=None,
)

# CORS para la app móvil
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "https://bancamovil.banco.com").split(","),
    allow_credentials=True,
    allow_methods=["POST", "DELETE", "GET"],
    allow_headers=["Authorization", "Content-Type", "X-Session-ID", "X-Customer-ID"],
)


# ─────────────────────────────────────────────
# MODELOS DE DATOS
# ─────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000, description="Mensaje del cliente")
    session_id: str | None = Field(None, description="ID de sesión existente (opcional)")

    @field_validator("message")
    @classmethod
    def clean_message(cls, v: str) -> str:
        return v.strip()


class ChatResponse(BaseModel):
    response: str
    session_id: str
    functions_called: list[str] = []
    sources: str | None = None


class HealthResponse(BaseModel):
    status: str
    version: str = "1.0.0"
    redis: dict | None = None


# ─────────────────────────────────────────────
# DEPENDENCIAS DE SEGURIDAD
# ─────────────────────────────────────────────

async def get_auth_headers(
    authorization: str = Header(..., description="Bearer {jwt_token}"),
    x_customer_id: str = Header(..., alias="X-Customer-ID", description="ID del cliente"),
) -> tuple[str, str]:
    """Extrae y valida los headers de autenticación."""
    if not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Formato de Authorization inválido. Use: Bearer {token}",
        )
    jwt_token = authorization.removeprefix("Bearer ").strip()

    if not x_customer_id or len(x_customer_id) < 3:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Header X-Customer-ID es requerido.",
        )

    return jwt_token, x_customer_id


def get_agent() -> BankingAgent:
    if _agent is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="El agente bancario no está disponible en este momento.",
        )
    return _agent


# ─────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────

@app.post(
    "/api/v1/agent/chat",
    response_model=ChatResponse,
    summary="Enviar mensaje al agente bancario",
    responses={
        200: {"description": "Respuesta del agente"},
        401: {"description": "No autorizado"},
        429: {"description": "Demasiadas solicitudes"},
        503: {"description": "Servicio no disponible"},
    },
)
async def chat(
    request: ChatRequest,
    auth: tuple[str, str] = Depends(get_auth_headers),
    agent: BankingAgent = Depends(get_agent),
) -> ChatResponse:
    """
    Envía un mensaje al agente bancario y recibe respuesta en lenguaje natural.
    
    El agente consultará automáticamente las funciones bancarias necesarias
    (saldos, tarjetas, préstamos, etc.) según la consulta del cliente.
    """
    jwt_token, customer_id = auth
    session_id = request.session_id or str(uuid4())

    result = await agent.chat(
        user_message=request.message,
        session_id=session_id,
        jwt_token=jwt_token,
        customer_id=customer_id,
    )

    if result.get("error") == "UNAUTHORIZED":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Sesión inválida o expirada. Por favor, vuelve a iniciar sesión.",
        )

    if result.get("error") == "CONTENT_BLOCKED":
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="La solicitud contiene contenido no permitido.",
        )

    return ChatResponse(
        response=result["response"],
        session_id=result["session_id"],
        functions_called=result.get("functions_called", []),
        sources=result.get("sources"),
    )


@app.delete(
    "/api/v1/agent/session/{session_id}",
    summary="Cerrar sesión del agente",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def close_session(
    session_id: str,
    auth: tuple[str, str] = Depends(get_auth_headers),
    agent: BankingAgent = Depends(get_agent),
):
    """Elimina el historial de conversación y libera recursos de la sesión."""
    await agent.cache.delete_chat_history(session_id)
    _, customer_id = auth
    await agent.cache.invalidate_customer_cache(customer_id)
    logger.info("Sesión cerrada: session_id=%s, customer_id=%s", session_id, customer_id)


@app.get(
    "/api/v1/agent/health",
    response_model=HealthResponse,
    summary="Health check del agente",
)
async def health_check(agent: BankingAgent = Depends(get_agent)) -> HealthResponse:
    """Verifica el estado del agente y sus dependencias."""
    redis_health = await agent.cache.health_check()
    overall = "healthy" if redis_health.get("status") == "healthy" else "degraded"
    return HealthResponse(status=overall, redis=redis_health)


# ─────────────────────────────────────────────
# MANEJO GLOBAL DE ERRORES
# ─────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception("Error no manejado en %s: %s", request.url.path, exc)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Error interno del servidor. Por favor intenta más tarde."},
    )
