"""
content_safety.py  +  security.py
==================================
Módulo de seguridad del agente bancario:
  1. ContentSafetyFilter  → Azure AI Content Safety
  2. SecurityContext      → Validación JWT + RBAC básico
"""

# ──────────────────────────────────────────────────────────────
# CONTENT SAFETY FILTER
# ──────────────────────────────────────────────────────────────

import logging
import os
from dataclasses import dataclass

from azure.ai.contentsafety.aio import ContentSafetyClient
from azure.ai.contentsafety.models import (
    AnalyzeTextOptions,
    TextCategory,
)
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import HttpResponseError

logger = logging.getLogger("content_safety")


@dataclass
class SafetyResult:
    """Resultado del análisis de seguridad de contenido."""
    is_blocked: bool
    reason: str
    severity: dict[str, int]  # {'hate': 0, 'violence': 0, 'self_harm': 0, 'sexual': 0}


class ContentSafetyFilter:
    """
    Filtro de contenido usando Azure AI Content Safety.
    
    Analiza tanto el INPUT del usuario como el OUTPUT del LLM
    para garantizar que ningún contenido dañino entre o salga
    del sistema bancario.
    
    Umbrales configurados conservadoramente para entorno bancario:
      - Severidad 2+ → BLOQUEADO (escala: 0-6)
    """

    # Umbral de severidad para entorno bancario (conservador)
    SEVERITY_THRESHOLD = 2
    BANKING_BLOCK_PATTERNS = [
        "número completo de tarjeta",
        "clave secreta",
        "pin bancario",
        "contraseña",
        "cvv",
        "clave dinámica",
    ]

    def __init__(self):
        self._client: ContentSafetyClient | None = None
        self.endpoint = os.environ.get("AZURE_CONTENT_SAFETY_ENDPOINT", "")
        self.api_key = os.environ.get("AZURE_CONTENT_SAFETY_API_KEY", "")
        self._enabled = bool(self.endpoint and self.api_key)

        if not self._enabled:
            logger.warning(
                "Azure Content Safety no configurado. "
                "Las variables AZURE_CONTENT_SAFETY_ENDPOINT y "
                "AZURE_CONTENT_SAFETY_API_KEY son requeridas en producción."
            )

    def _get_client(self) -> ContentSafetyClient:
        if self._client is None:
            self._client = ContentSafetyClient(
                endpoint=self.endpoint,
                credential=AzureKeyCredential(self.api_key),
            )
        return self._client

    def _check_banking_patterns(self, text: str) -> SafetyResult | None:
        """
        Bloqueo adicional de patrones específicos del dominio bancario.
        Previene que el LLM revele o solicite datos sensibles.
        """
        text_lower = text.lower()
        for pattern in self.BANKING_BLOCK_PATTERNS:
            if pattern in text_lower:
                return SafetyResult(
                    is_blocked=True,
                    reason=f"Patrón bancario sensible detectado: '{pattern}'",
                    severity={},
                )
        return None

    async def _analyze_with_azure(self, text: str) -> SafetyResult:
        """Analiza texto con Azure AI Content Safety."""
        try:
            client = self._get_client()
            request = AnalyzeTextOptions(
                text=text[:10_000],  # Límite de API
                categories=[
                    TextCategory.HATE,
                    TextCategory.VIOLENCE,
                    TextCategory.SELF_HARM,
                    TextCategory.SEXUAL,
                ],
                output_type="FourSeverityLevels",
            )
            response = await client.analyze_text(request)

            severities = {
                "hate": response.hate_result.severity if response.hate_result else 0,
                "violence": response.violence_result.severity if response.violence_result else 0,
                "self_harm": response.self_harm_result.severity if response.self_harm_result else 0,
                "sexual": response.sexual_result.severity if response.sexual_result else 0,
            }

            max_severity = max(severities.values())
            if max_severity >= self.SEVERITY_THRESHOLD:
                category = max(severities, key=severities.get)
                return SafetyResult(
                    is_blocked=True,
                    reason=f"Categoría '{category}' con severidad {max_severity}",
                    severity=severities,
                )

            return SafetyResult(is_blocked=False, reason="OK", severity=severities)

        except HttpResponseError as exc:
            logger.error("Error de Azure Content Safety API: %s", exc)
            # Fail-open en caso de error de servicio (decisión de negocio)
            return SafetyResult(is_blocked=False, reason="service_error", severity={})

    async def analyze_input(self, text: str, customer_id: str) -> SafetyResult:
        """
        Analiza el mensaje del usuario antes de procesarlo.
        
        Aplica dos capas:
          1. Patrones bancarios locales (sin latencia)
          2. Azure AI Content Safety (análisis profundo)
        """
        # Capa 1: patrones locales (rápido)
        local_result = self._check_banking_patterns(text)
        if local_result:
            logger.warning(
                "INPUT bloqueado por patrón local | customer=%s | razón=%s",
                customer_id,
                local_result.reason,
            )
            return local_result

        # Capa 2: Azure Content Safety
        if not self._enabled:
            return SafetyResult(is_blocked=False, reason="safety_disabled", severity={})

        result = await self._analyze_with_azure(text)
        if result.is_blocked:
            logger.warning(
                "INPUT bloqueado por Azure Safety | customer=%s | razón=%s | severidades=%s",
                customer_id,
                result.reason,
                result.severity,
            )
        return result

    async def analyze_output(self, text: str, customer_id: str) -> SafetyResult:
        """
        Analiza la respuesta del LLM antes de enviarla al cliente.
        Capa de seguridad adicional sobre el output del modelo.
        """
        if not self._enabled:
            return SafetyResult(is_blocked=False, reason="safety_disabled", severity={})

        result = await self._analyze_with_azure(text)
        if result.is_blocked:
            logger.error(
                "OUTPUT bloqueado por Azure Safety | customer=%s | razón=%s",
                customer_id,
                result.reason,
            )
        return result

    async def close(self):
        if self._client:
            await self._client.close()
