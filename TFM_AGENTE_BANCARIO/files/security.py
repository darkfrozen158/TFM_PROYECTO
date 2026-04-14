"""
security.py
===========
Validación de seguridad del agente bancario:
  - Validación de tokens JWT (Azure AD B2C / MSAL)
  - Contexto de seguridad por sesión
  - Rate limiting por cliente
  - Auditoría de accesos
"""

import logging
import os
import time
from dataclasses import dataclass, field
from functools import lru_cache

import jwt
from jwt import PyJWKClient

logger = logging.getLogger("security")

# ─────────────────────────────────────────────
# SECURITY CONTEXT
# ─────────────────────────────────────────────

@dataclass
class SecurityContext:
    """Contexto de seguridad validado para una petición."""
    is_valid: bool
    customer_id: str
    scopes: list[str] = field(default_factory=list)
    token_exp: int = 0
    error_reason: str = ""

    @property
    def has_scope(self) -> bool:
        """Verifica si el token tiene el scope de agente bancario."""
        return "banking.agent" in self.scopes or "banking.read" in self.scopes

    @property
    def can_query_loans(self) -> bool:
        return "banking.loans.read" in self.scopes or "banking.read" in self.scopes

    @property
    def can_query_cards(self) -> bool:
        return "banking.cards.read" in self.scopes or "banking.read" in self.scopes


# ─────────────────────────────────────────────
# JWT VALIDATOR
# ─────────────────────────────────────────────

AZURE_AD_TENANT_ID = os.environ.get("AZURE_AD_TENANT_ID", "")
AZURE_AD_CLIENT_ID = os.environ.get("AZURE_AD_CLIENT_ID", "")
JWKS_URI = os.environ.get(
    "AZURE_AD_JWKS_URI",
    f"https://login.microsoftonline.com/{AZURE_AD_TENANT_ID}/discovery/v2.0/keys"
    if AZURE_AD_TENANT_ID else "",
)

# Rate limiting en memoria (en producción usar Redis)
_rate_limit_store: dict[str, list[float]] = {}
RATE_LIMIT_MAX = int(os.environ.get("AGENT_RATE_LIMIT_PER_MINUTE", "30"))
DEMO_MODE = os.environ.get("DEMO_MODE", "false").lower() == "true"


@lru_cache(maxsize=1)
def _get_jwks_client() -> PyJWKClient | None:
    """Crea cliente JWKS (cacheado para reutilizar claves públicas)."""
    if not JWKS_URI:
        return None
    return PyJWKClient(JWKS_URI, cache_keys=True)


def _check_rate_limit(customer_id: str) -> bool:
    """
    Rate limiting simple por customer_id.
    Ventana deslizante de 60 segundos.
    En producción, reemplazar con Redis INCR + EXPIRE.
    
    Returns:
        True si está dentro del límite, False si lo excede.
    """
    now = time.time()
    window_start = now - 60.0

    if customer_id not in _rate_limit_store:
        _rate_limit_store[customer_id] = []

    # Limpiar timestamps fuera de la ventana
    _rate_limit_store[customer_id] = [
        ts for ts in _rate_limit_store[customer_id] if ts > window_start
    ]

    if len(_rate_limit_store[customer_id]) >= RATE_LIMIT_MAX:
        logger.warning(
            "Rate limit excedido para customer_id=%s (%d req/min)",
            customer_id,
            RATE_LIMIT_MAX,
        )
        return False

    _rate_limit_store[customer_id].append(now)
    return True


def validate_jwt_token(jwt_token: str, customer_id: str) -> SecurityContext:
    """
    Valida el token JWT del cliente autenticado.
    
    En modo DEMO (DEMO_MODE=true), acepta el token 'DEMO_TOKEN' sin validar.
    En producción, valida contra Azure AD B2C con JWKS.

    Args:
        jwt_token:   Token JWT de la app móvil.
        customer_id: ID del cliente a validar contra el token.

    Returns:
        SecurityContext con el resultado de la validación.
    """

    # ── MODO DEMO ────────────────────────────────────────────────────────────
    if DEMO_MODE and jwt_token == "DEMO_TOKEN":
        logger.debug("Modo DEMO activo para customer_id=%s", customer_id)
        return SecurityContext(
            is_valid=True,
            customer_id=customer_id,
            scopes=["banking.read", "banking.agent"],
            token_exp=int(time.time()) + 3600,
        )

    # ── RATE LIMITING ────────────────────────────────────────────────────────
    if not _check_rate_limit(customer_id):
        return SecurityContext(
            is_valid=False,
            customer_id=customer_id,
            error_reason="RATE_LIMIT_EXCEEDED",
        )

    # ── VALIDACIÓN JWT ───────────────────────────────────────────────────────
    jwks_client = _get_jwks_client()
    if jwks_client is None:
        logger.warning(
            "JWKS client no configurado (AZURE_AD_JWKS_URI). "
            "Validación JWT omitida. Configura en producción."
        )
        return SecurityContext(
            is_valid=True,
            customer_id=customer_id,
            scopes=["banking.read"],
            error_reason="JWT_VALIDATION_SKIPPED",
        )

    try:
        signing_key = jwks_client.get_signing_key_from_jwt(jwt_token)
        payload = jwt.decode(
            jwt_token,
            signing_key.key,
            algorithms=["RS256"],
            audience=AZURE_AD_CLIENT_ID,
            options={"verify_exp": True},
        )

        # Verificar que el customer_id del token coincide con el solicitado
        token_subject = payload.get("sub", "")
        token_oid = payload.get("oid", "")
        token_customer = payload.get("extension_customerId", "")

        if customer_id not in (token_subject, token_oid, token_customer):
            logger.warning(
                "customer_id mismatch: claim=%s, solicitado=%s",
                token_customer or token_subject,
                customer_id,
            )
            return SecurityContext(
                is_valid=False,
                customer_id=customer_id,
                error_reason="CUSTOMER_ID_MISMATCH",
            )

        scopes = payload.get("scp", "banking.read").split(" ")

        logger.info(
            "Token JWT válido para customer_id=%s | scopes=%s | exp=%s",
            customer_id,
            scopes,
            payload.get("exp"),
        )

        return SecurityContext(
            is_valid=True,
            customer_id=customer_id,
            scopes=scopes,
            token_exp=payload.get("exp", 0),
        )

    except jwt.ExpiredSignatureError:
        logger.info("Token expirado para customer_id=%s", customer_id)
        return SecurityContext(
            is_valid=False,
            customer_id=customer_id,
            error_reason="TOKEN_EXPIRED",
        )
    except jwt.InvalidTokenError as exc:
        logger.warning("Token inválido para customer_id=%s: %s", customer_id, exc)
        return SecurityContext(
            is_valid=False,
            customer_id=customer_id,
            error_reason="INVALID_TOKEN",
        )
    except Exception as exc:
        logger.exception("Error inesperado validando JWT para customer_id=%s", customer_id)
        return SecurityContext(
            is_valid=False,
            customer_id=customer_id,
            error_reason="VALIDATION_ERROR",
        )
