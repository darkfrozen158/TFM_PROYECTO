# lakeflow_pipeline/gold/gold_kpis.py
import dlt
from pyspark.sql import functions as F
from pyspark.sql.window import Window

# ── KPI: Saldos diarios por tipo de producto ──────────────────────
@dlt.table(
    name="gold_saldos_diarios",
    comment="Saldos EOD agregados por producto y segmento cliente",
    table_properties={"quality": "gold"},
    partition_cols=["fecha_corte"]
)
def gold_saldos_diarios():
    cuentas = spark.read.table("banking_silver.silver_cuentas")
    txns    = spark.read.table("banking_silver.silver_transacciones")

    # Saldo corriente: saldo_inicial + suma de movimientos del día
    daily_movements = (
        txns.groupBy("cuenta_id", F.to_date("fecha_transaccion").alias("fecha_corte"))
        .agg(
            F.sum(F.when(F.col("tipo_transaccion_desc") == "CREDITO",  F.col("monto")).otherwise(0)).alias("total_creditos"),
            F.sum(F.when(F.col("tipo_transaccion_desc") == "DEBITO",  -F.col("monto")).otherwise(0)).alias("total_debitos"),
            F.count("*").alias("num_transacciones")
        )
    )

    return (
        cuentas.select("cuenta_id", "tipo_producto", "segmento_cliente", "moneda", "saldo_inicial")
        .join(daily_movements, "cuenta_id", "left")
        .withColumn("saldo_cierre",
            F.col("saldo_inicial") + F.coalesce("total_creditos", F.lit(0)) + F.coalesce("total_debitos", F.lit(0))
        )
        .withColumn("fecha_corte", F.coalesce("fecha_corte", F.current_date()))
        .select(
            "fecha_corte", "cuenta_id", "tipo_producto", "segmento_cliente",
            "moneda", "saldo_cierre", "total_creditos", "total_debitos", "num_transacciones"
        )
    )

# ── KPI: Uso de tarjetas mensual (para riesgo y marketing) ────────
@dlt.table(name="gold_uso_tarjetas_mensual", table_properties={"quality": "gold"})
def gold_uso_tarjetas_mensual():
    tarjetas = spark.read.table("banking_silver.silver_tarjetas")
    txns     = spark.read.table("banking_silver.silver_transacciones")

    return (
        txns.filter("tipo_transaccion_desc IN ('DEBITO', 'PAGO_SERVICIO')")
        .withColumn("anio_mes", F.date_format("fecha_transaccion", "yyyy-MM"))
        .groupBy("tarjeta_id", "anio_mes")
        .agg(
            F.count("*").alias("num_compras"),
            F.sum("monto").alias("monto_total_consumido"),
            F.avg("monto").alias("ticket_promedio"),
            F.countDistinct(F.to_date("fecha_transaccion")).alias("dias_activos")
        )
        .join(tarjetas.select("tarjeta_id", "tipo_tarjeta", "estado_tarjeta", "cliente_id"), "tarjeta_id")
        .withColumn("tasa_actividad", F.col("dias_activos") / 30.0)
    )