"""
cache_manager.py
================
Gestión eficiente de memoria y sesiones del agente bancario
usando Redis Cache for Azure.

Almacena:
  - Historial de chat por sesión (ChatHistory serializado)
  - Resultados de consultas bancarias frecuentes
  - Perfil de cliente enriquecido
"""

import json
import logging
import os
import pickle
from typing import Any

import redis.asyncio as aioredis
from semantic_kernel.contents import AuthorRole, ChatHistory, ChatMessageContent

logger = logging.getLogger("cache_manager")


class RedisCacheManager:
    """
    Manager de caché usando Redis Cache for Azure.
    
    Patrones de caché:
      - chat_history:{session_id}     → ChatHistory serializado (TTL: 30 min)
      - banking_data:{customer_id}:{type} → Datos bancarios cacheados (TTL: 5 min)
      - customer_profile:{customer_id}  → Perfil del cliente (TTL: 1 hora)
    """

    CHAT_HISTORY_PREFIX = "chat_history"
    BANKING_DATA_PREFIX = "banking_data"
    CUSTOMER_PROFILE_PREFIX = "customer_profile"

    # TTLs en segundos
    DEFAULT_CHAT_TTL = 1800      # 30 minutos
    BANKING_DATA_TTL = 300       # 5 minutos (datos financieros cambian)
    CUSTOMER_PROFILE_TTL = 3600  # 1 hora

    def __init__(self):
        self._redis: aioredis.Redis | None = None
        self._connection_string = os.environ.get("AZURE_REDIS_CONNECTION_STRING", "")
        self._enabled = bool(self._connection_string)

        if not self._enabled:
            logger.warning(
                "Redis no configurado (AZURE_REDIS_CONNECTION_STRING ausente). "
                "Usando caché en memoria como fallback."
            )
            self._memory_cache: dict[str, Any] = {}

    def _get_redis(self) -> aioredis.Redis:
        """Lazy init del cliente Redis."""
        if self._redis is None:
            self._redis = aioredis.from_url(
                self._connection_string,
                encoding="utf-8",
                decode_responses=False,  # Necesario para datos binarios (pickle)
                socket_connect_timeout=5,
                socket_timeout=3,
                retry_on_timeout=True,
                health_check_interval=30,
            )
        return self._redis

    # ── HISTORIAL DE CHAT ────────────────────────────────────────────────────

    def _serialize_history(self, history: ChatHistory) -> bytes:
        """Serializa ChatHistory a bytes para Redis."""
        messages = []
        for msg in history.messages:
            messages.append({
                "role": msg.role.value,
                "content": str(msg.content),
            })
        return pickle.dumps({"messages": messages})

    def _deserialize_history(self, data: bytes) -> ChatHistory:
        """Deserializa bytes de Redis a ChatHistory."""
        payload = pickle.loads(data)
        history = ChatHistory()
        for msg in payload.get("messages", []):
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                history.add_system_message(content)
            elif role == "user":
                history.add_user_message(content)
            elif role == "assistant":
                history.add_assistant_message(content)
        return history

    async def get_chat_history(self, session_id: str) -> ChatHistory | None:
        """
        Recupera el historial de chat de Redis por session_id.
        
        Returns:
            ChatHistory si existe, None si es nueva sesión.
        """
        key = f"{self.CHAT_HISTORY_PREFIX}:{session_id}"

        if not self._enabled:
            raw = self._memory_cache.get(key)
            if raw:
                logger.debug("Cache HIT (memoria) para session_id=%s", session_id)
                return self._deserialize_history(raw)
            return None

        try:
            redis = self._get_redis()
            raw = await redis.get(key)
            if raw:
                logger.debug("Cache HIT (Redis) para session_id=%s", session_id)
                return self._deserialize_history(raw)
            logger.debug("Cache MISS para session_id=%s", session_id)
            return None
        except Exception as exc:
            logger.warning(
                "Error leyendo historial de Redis para session_id=%s: %s",
                session_id,
                exc,
            )
            return None

    async def save_chat_history(
        self,
        session_id: str,
        history: ChatHistory,
        ttl_seconds: int = DEFAULT_CHAT_TTL,
    ) -> bool:
        """
        Guarda el historial de chat en Redis.
        Trunca automáticamente si supera los últimos 20 mensajes para
        controlar el uso de memoria sin perder contexto relevante.
        """
        # Truncar historial largo (mantener system + últimos 19 mensajes)
        if len(history.messages) > 20:
            system_msgs = [m for m in history.messages if m.role == AuthorRole.SYSTEM]
            recent_msgs = [m for m in history.messages if m.role != AuthorRole.SYSTEM][-19:]
            trimmed = ChatHistory()
            for msg in system_msgs:
                trimmed.add_system_message(str(msg.content))
            for msg in recent_msgs:
                if msg.role == AuthorRole.USER:
                    trimmed.add_user_message(str(msg.content))
                elif msg.role == AuthorRole.ASSISTANT:
                    trimmed.add_assistant_message(str(msg.content))
            history = trimmed
            logger.debug("Historial truncado a 20 mensajes para session_id=%s", session_id)

        key = f"{self.CHAT_HISTORY_PREFIX}:{session_id}"
        serialized = self._serialize_history(history)

        if not self._enabled:
            self._memory_cache[key] = serialized
            return True

        try:
            redis = self._get_redis()
            await redis.setex(key, ttl_seconds, serialized)
            return True
        except Exception as exc:
            logger.warning(
                "Error guardando historial en Redis para session_id=%s: %s",
                session_id,
                exc,
            )
            return False

    async def delete_chat_history(self, session_id: str) -> bool:
        """Elimina el historial de chat (logout o reset de sesión)."""
        key = f"{self.CHAT_HISTORY_PREFIX}:{session_id}"

        if not self._enabled:
            self._memory_cache.pop(key, None)
            return True

        try:
            redis = self._get_redis()
            await redis.delete(key)
            logger.info("Historial eliminado para session_id=%s", session_id)
            return True
        except Exception as exc:
            logger.warning("Error eliminando historial para session_id=%s: %s", session_id, exc)
            return False

    # ── CACHÉ DE DATOS BANCARIOS ─────────────────────────────────────────────

    async def get_banking_data(self, customer_id: str, data_type: str) -> dict | None:
        """
        Recupera datos bancarios cacheados (balance, tarjetas, etc).
        Reduce llamadas repetidas al core bancario.
        """
        key = f"{self.BANKING_DATA_PREFIX}:{customer_id}:{data_type}"

        if not self._enabled:
            return self._memory_cache.get(key)

        try:
            redis = self._get_redis()
            raw = await redis.get(key)
            if raw:
                logger.debug("Cache HIT bancario: %s para customer=%s", data_type, customer_id)
                return json.loads(raw)
            return None
        except Exception as exc:
            logger.warning("Error leyendo datos bancarios de Redis: %s", exc)
            return None

    async def set_banking_data(
        self,
        customer_id: str,
        data_type: str,
        data: dict,
        ttl_seconds: int = BANKING_DATA_TTL,
    ) -> bool:
        """Cachea datos bancarios con TTL corto (datos financieros cambian frecuentemente)."""
        key = f"{self.BANKING_DATA_PREFIX}:{customer_id}:{data_type}"

        if not self._enabled:
            self._memory_cache[key] = data
            return True

        try:
            redis = self._get_redis()
            await redis.setex(key, ttl_seconds, json.dumps(data, default=str))
            return True
        except Exception as exc:
            logger.warning("Error cacheando datos bancarios: %s", exc)
            return False

    async def invalidate_customer_cache(self, customer_id: str) -> int:
        """
        Invalida TODA la caché de un cliente.
        Llamar después de operaciones que modifican datos (transferencias, pagos).
        """
        if not self._enabled:
            keys_to_delete = [
                k for k in self._memory_cache
                if customer_id in k
            ]
            for k in keys_to_delete:
                del self._memory_cache[k]
            return len(keys_to_delete)

        try:
            redis = self._get_redis()
            pattern = f"*:{customer_id}:*"
            deleted = 0
            async for key in redis.scan_iter(match=pattern, count=100):
                await redis.delete(key)
                deleted += 1
            logger.info(
                "Caché invalidada: %d claves eliminadas para customer_id=%s",
                deleted,
                customer_id,
            )
            return deleted
        except Exception as exc:
            logger.warning("Error invalidando caché para customer_id=%s: %s", customer_id, exc)
            return 0

    # ── HEALTH CHECK ─────────────────────────────────────────────────────────

    async def health_check(self) -> dict:
        """Verifica la conectividad con Redis."""
        if not self._enabled:
            return {"status": "degraded", "backend": "memory", "message": "Redis no configurado"}

        try:
            redis = self._get_redis()
            await redis.ping()
            info = await redis.info("memory")
            return {
                "status": "healthy",
                "backend": "redis",
                "used_memory_human": info.get("used_memory_human", "N/A"),
                "connected_clients": info.get("connected_clients", "N/A"),
            }
        except Exception as exc:
            return {"status": "unhealthy", "backend": "redis", "error": str(exc)}

    async def close(self):
        """Cierra la conexión Redis."""
        if self._redis:
            await self._redis.close()
            logger.info("Conexión Redis cerrada.")
