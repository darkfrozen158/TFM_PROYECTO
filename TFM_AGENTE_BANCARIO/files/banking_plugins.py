"""
banking_plugins.py
==================
Plugins de Semantic Kernel con las funciones bancarias.
Cada método decorado con @kernel_function es auto-descubierto
por el LLM para function calling automático.
"""

import logging
import os
from typing import Annotated, Any

import httpx
from semantic_kernel.functions import kernel_function

logger = logging.getLogger("banking_plugins")

# URL del core bancario (microservicio interno)
CORE_API_URL = os.getenv("BANKING_CORE_API_URL", "https://core-banking-api.internal")
CORE_API_KEY = os.getenv("BANKING_CORE_API_KEY", "")


async def _call_core_api(
    endpoint: str,
    customer_id: str,
    params: dict | None = None,
) -> dict[str, Any]:
    """Cliente HTTP interno para el core bancario. Usa mTLS en producción."""
    headers = {
        "X-API-Key": CORE_API_KEY,
        "X-Customer-ID": customer_id,
        "Content-Type": "application/json",
    }
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(
            f"{CORE_API_URL}/{endpoint}",
            headers=headers,
            params=params or {},
        )
        resp.raise_for_status()
        return resp.json()


# ─────────────────────────────────────────────
# PLUGIN: CUENTAS
# ─────────────────────────────────────────────

class AccountPlugin:
    """Plugin para consultas de cuentas bancarias digitales."""

    @kernel_function(
        name="get_account_balance",
        description=(
            "Obtiene el saldo actual de las cuentas del cliente. "
            "Usar cuando el cliente pregunta por su saldo, disponible o movimientos recientes."
        ),
    )
    async def get_account_balance(
        self,
        customer_id: Annotated[str, "ID del cliente en el sistema bancario"],
        currency: Annotated[str, "Moneda de la cuenta: 'PEN' para soles, 'USD' para dólares"] = "PEN",
    ) -> str:
        """Consulta saldo de cuentas por moneda."""
        try:
            data = await _call_core_api(
                endpoint=f"customers/{customer_id}/accounts/balance",
                customer_id=customer_id,
                params={"currency": currency},
            )
            accounts = data.get("accounts", [])
            if not accounts:
                return f"No se encontraron cuentas en {currency} para el cliente."

            result_parts = [f"Saldos de cuentas en {currency}:"]
            for acc in accounts:
                account_number = acc.get("account_number", "")
                masked = f"****{account_number[-4:]}" if len(account_number) >= 4 else "****"
                result_parts.append(
                    f"- Cuenta {masked}: "
                    f"Disponible: {acc.get('available_balance', 0):,.2f} {currency} | "
                    f"Total: {acc.get('total_balance', 0):,.2f} {currency}"
                )
            return "\n".join(result_parts)

        except Exception as exc:
            logger.exception("Error al obtener saldo de cuenta para customer_id=%s", customer_id)
            return "No pude obtener el saldo en este momento. Por favor intenta más tarde."

    @kernel_function(
        name="get_account_movements",
        description=(
            "Obtiene los últimos movimientos o transacciones de la cuenta del cliente. "
            "Usar cuando preguntan por historial, últimas operaciones o movimientos."
        ),
    )
    async def get_account_movements(
        self,
        customer_id: Annotated[str, "ID del cliente"],
        currency: Annotated[str, "Moneda: 'PEN' o 'USD'"] = "PEN",
        limit: Annotated[int, "Número de movimientos a mostrar (máximo 10)"] = 5,
    ) -> str:
        """Obtiene los últimos movimientos de cuenta."""
        try:
            limit = min(limit, 10)  # Seguridad: máximo 10
            data = await _call_core_api(
                endpoint=f"customers/{customer_id}/accounts/movements",
                customer_id=customer_id,
                params={"currency": currency, "limit": limit},
            )
            movements = data.get("movements", [])
            if not movements:
                return f"No hay movimientos recientes en tu cuenta {currency}."

            result_parts = [f"Últimos {len(movements)} movimientos en {currency}:"]
            for mov in movements:
                sign = "+" if mov.get("type") == "credit" else "-"
                result_parts.append(
                    f"• {mov.get('date', 'N/A')} | {mov.get('description', 'Sin descripción')} | "
                    f"{sign}{abs(mov.get('amount', 0)):,.2f} {currency}"
                )
            return "\n".join(result_parts)

        except Exception as exc:
            logger.exception("Error al obtener movimientos para customer_id=%s", customer_id)
            return "No pude obtener los movimientos en este momento."


# ─────────────────────────────────────────────
# PLUGIN: TARJETAS DE CRÉDITO
# ─────────────────────────────────────────────

class CardPlugin:
    """Plugin para tarjetas de crédito digitales."""

    @kernel_function(
        name="get_credit_cards",
        description=(
            "Obtiene información de las tarjetas de crédito digitales del cliente: "
            "límite, saldo usado, fecha de corte y pago mínimo. "
            "Usar cuando preguntan por su tarjeta, deuda, límite o fecha de pago."
        ),
    )
    async def get_credit_cards(
        self,
        customer_id: Annotated[str, "ID del cliente"],
    ) -> str:
        """Consulta tarjetas de crédito digitales."""
        try:
            data = await _call_core_api(
                endpoint=f"customers/{customer_id}/cards",
                customer_id=customer_id,
            )
            cards = data.get("cards", [])
            if not cards:
                return "No tienes tarjetas de crédito digitales activas."

            result_parts = ["Tus tarjetas de crédito digitales:"]
            for card in cards:
                card_num = card.get("card_number", "")
                masked = f"**** **** **** {card_num[-4:]}" if len(card_num) >= 4 else "****"
                utilization = (
                    (card.get("used_balance", 0) / card.get("credit_limit", 1)) * 100
                    if card.get("credit_limit", 0) > 0
                    else 0
                )
                result_parts.append(
                    f"\n📱 Tarjeta: {masked} ({card.get('card_type', 'VISA')})\n"
                    f"   Estado: {card.get('status', 'ACTIVA')}\n"
                    f"   Límite: {card.get('credit_limit', 0):,.2f} {card.get('currency', 'PEN')}\n"
                    f"   Usado: {card.get('used_balance', 0):,.2f} ({utilization:.1f}%)\n"
                    f"   Disponible: {card.get('available_balance', 0):,.2f}\n"
                    f"   Fecha de corte: {card.get('cut_date', 'N/A')}\n"
                    f"   Pago mínimo: {card.get('minimum_payment', 0):,.2f}\n"
                    f"   Pago total: {card.get('total_payment', 0):,.2f}"
                )
            return "\n".join(result_parts)

        except Exception as exc:
            logger.exception("Error al obtener tarjetas para customer_id=%s", customer_id)
            return "No pude obtener información de tus tarjetas en este momento."

    @kernel_function(
        name="get_card_movements",
        description=(
            "Obtiene los últimos consumos o movimientos de la tarjeta de crédito. "
            "Usar cuando el cliente pregunta por sus compras o consumos recientes."
        ),
    )
    async def get_card_movements(
        self,
        customer_id: Annotated[str, "ID del cliente"],
        last_n_days: Annotated[int, "Días hacia atrás a consultar (máximo 90)"] = 30,
    ) -> str:
        """Consulta movimientos de tarjeta de crédito."""
        try:
            last_n_days = min(last_n_days, 90)
            data = await _call_core_api(
                endpoint=f"customers/{customer_id}/cards/movements",
                customer_id=customer_id,
                params={"days": last_n_days},
            )
            movements = data.get("movements", [])
            if not movements:
                return f"No hay movimientos en tus tarjetas en los últimos {last_n_days} días."

            result_parts = [f"Últimos movimientos de tarjeta ({last_n_days} días):"]
            for mov in movements:
                result_parts.append(
                    f"• {mov.get('date', 'N/A')} | {mov.get('merchant', 'Comercio')} | "
                    f"{mov.get('amount', 0):,.2f} {mov.get('currency', 'PEN')} | "
                    f"{mov.get('installments', 1)} cuota(s)"
                )
            return "\n".join(result_parts)

        except Exception as exc:
            logger.exception("Error al obtener movimientos de tarjeta para customer_id=%s", customer_id)
            return "No pude obtener los movimientos de tarjeta en este momento."


# ─────────────────────────────────────────────
# PLUGIN: PRÉSTAMOS
# ─────────────────────────────────────────────

class LoanPlugin:
    """Plugin para préstamos digitales."""

    @kernel_function(
        name="get_loans",
        description=(
            "Obtiene información de los préstamos digitales del cliente: "
            "saldo pendiente, cuota mensual, tasa de interés y próximas fechas de pago. "
            "Usar cuando preguntan por deuda, crédito personal, préstamo o cuotas."
        ),
    )
    async def get_loans(
        self,
        customer_id: Annotated[str, "ID del cliente"],
    ) -> str:
        """Consulta préstamos digitales activos."""
        try:
            data = await _call_core_api(
                endpoint=f"customers/{customer_id}/loans",
                customer_id=customer_id,
            )
            loans = data.get("loans", [])
            if not loans:
                return "No tienes préstamos digitales activos."

            result_parts = ["Tus préstamos digitales activos:"]
            for loan in loans:
                result_parts.append(
                    f"\n💰 Préstamo #{loan.get('loan_id', 'N/A')[:8]}...\n"
                    f"   Tipo: {loan.get('loan_type', 'Personal')}\n"
                    f"   Saldo pendiente: {loan.get('outstanding_balance', 0):,.2f} {loan.get('currency', 'PEN')}\n"
                    f"   Cuota mensual: {loan.get('monthly_payment', 0):,.2f}\n"
                    f"   Tasa TEA: {loan.get('annual_rate', 0):.2f}%\n"
                    f"   Cuotas restantes: {loan.get('remaining_installments', 0)}\n"
                    f"   Próximo vencimiento: {loan.get('next_due_date', 'N/A')}"
                )
            return "\n".join(result_parts)

        except Exception as exc:
            logger.exception("Error al obtener préstamos para customer_id=%s", customer_id)
            return "No pude obtener información de tus préstamos en este momento."

    @kernel_function(
        name="get_loan_eligibility",
        description=(
            "Consulta si el cliente es elegible para un nuevo préstamo digital "
            "y el monto máximo pre-aprobado. Usar cuando el cliente pregunta "
            "si puede acceder a un crédito o cuánto puede pedir prestado."
        ),
    )
    async def get_loan_eligibility(
        self,
        customer_id: Annotated[str, "ID del cliente"],
    ) -> str:
        """Consulta elegibilidad y monto pre-aprobado para nuevos préstamos."""
        try:
            data = await _call_core_api(
                endpoint=f"customers/{customer_id}/loans/eligibility",
                customer_id=customer_id,
            )
            if data.get("eligible"):
                return (
                    f"¡Tienes una oferta pre-aprobada!\n"
                    f"   Monto máximo: {data.get('max_amount', 0):,.2f} {data.get('currency', 'PEN')}\n"
                    f"   Tasa desde: {data.get('min_rate', 0):.2f}% TEA\n"
                    f"   Plazo máximo: {data.get('max_term_months', 0)} meses\n"
                    f"   ¿Deseas solicitar el préstamo ahora? Puedo guiarte en el proceso."
                )
            else:
                return (
                    f"En este momento no calificas para un nuevo préstamo digital. "
                    f"Motivo: {data.get('reason', 'No disponible')}. "
                    f"Si tienes dudas, puedes contactar a un asesor."
                )
        except Exception as exc:
            logger.exception("Error al consultar elegibilidad para customer_id=%s", customer_id)
            return "No pude verificar tu elegibilidad para préstamos en este momento."


# ─────────────────────────────────────────────
# PLUGIN: TRANSFERENCIAS
# ─────────────────────────────────────────────

class TransferPlugin:
    """Plugin para consultar límites y últimas transferencias (solo lectura por seguridad)."""

    @kernel_function(
        name="get_transfer_limits",
        description=(
            "Obtiene los límites de transferencia diaria y mensual del cliente. "
            "Usar cuando preguntan cuánto pueden transferir o sus límites operacionales."
        ),
    )
    async def get_transfer_limits(
        self,
        customer_id: Annotated[str, "ID del cliente"],
    ) -> str:
        """Consulta límites de transferencia del cliente."""
        try:
            data = await _call_core_api(
                endpoint=f"customers/{customer_id}/transfers/limits",
                customer_id=customer_id,
            )
            return (
                f"Tus límites de transferencia:\n"
                f"   Diario usado: {data.get('daily_used', 0):,.2f} / {data.get('daily_limit', 0):,.2f} PEN\n"
                f"   Mensual usado: {data.get('monthly_used', 0):,.2f} / {data.get('monthly_limit', 0):,.2f} PEN\n"
                f"   Disponible hoy: {data.get('daily_available', 0):,.2f} PEN"
            )
        except Exception as exc:
            logger.exception("Error al obtener límites para customer_id=%s", customer_id)
            return "No pude obtener tus límites de transferencia en este momento."
