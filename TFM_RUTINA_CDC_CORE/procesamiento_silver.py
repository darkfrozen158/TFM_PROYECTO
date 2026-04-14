# lakeflow_pipeline/silver/silver_cuentas.py
import dlt
from pyspark.sql import functions as F
from pyspark.sql.types import DecimalType, TimestampType, StringType

# ── APPLY CHANGES: CDC para cuentas (SCD Type 1 + historial Type 2) ──
dlt.create_streaming_table(
    name="silver_cuentas",
    comment="Cuentas limpias con CDC aplicado - deduplicated",
    table_properties={"quality": "silver", "delta.enableChangeDataFeed": "true"},
    partition_cols=["tipo_producto"]
)

dlt.apply_changes(
    target="silver_cuentas",
    source="raw_cuentas",
    keys=["cuenta_id"],                          # PK de mainframe
    sequence_by="_ingest_timestamp",             # orden temporal
    apply_as_deletes=F.expr("operacion = 'D'"),  # filas con DELETE en CDC
    apply_as_truncates=F.expr("operacion = 'TRUNCATE'"),
    except_column_list=["_ingest_timestamp", "_batch_id", "_source_system"],
    stored_as_scd_type=2                         # historial completo
)

# ── TRANSACCIONES: validación con Expectations ─────────────────────
@dlt.table(
    name="silver_transacciones",
    comment="Transacciones validadas y normalizadas",
    table_properties={"quality": "silver"}
)
@dlt.expect_or_drop("monto_valido", "monto IS NOT NULL AND monto != 0")
@dlt.expect_or_drop("cuenta_existe", "cuenta_id IS NOT NULL")
@dlt.expect("moneda_valida", "moneda IN ('PEN', 'USD', 'EUR')")  # warn, no drop
def silver_transacciones():
    return (
        spark.read.table("banking_bronze.raw_transacciones")
        .withColumn("monto", F.col("monto").cast(DecimalType(18, 2)))
        .withColumn("fecha_transaccion", F.col("fecha_transaccion").cast(TimestampType()))
        # Normalizar tipo_transaccion que viene con códigos legacy del mainframe
        .withColumn("tipo_transaccion_desc",
            F.when(F.col("tipo_tx_cod") == "CR", "CREDITO")
             .when(F.col("tipo_tx_cod") == "DB", "DEBITO")
             .when(F.col("tipo_tx_cod") == "TF", "TRANSFERENCIA")
             .when(F.col("tipo_tx_cod") == "PG", "PAGO_SERVICIO")
             .otherwise("OTROS")
        )
        # Enmascarar PAN de tarjetas (PCI-DSS compliance)
        .withColumn("numero_tarjeta_masked",
            F.concat(F.lit("****-****-****-"), F.substring("numero_tarjeta", -4, 4))
        )
        .drop("numero_tarjeta")  # nunca persiste el PAN completo
        .withColumn("_silver_timestamp", F.current_timestamp())
    )

# ── TARJETAS con deduplicación ─────────────────────────────────────
@dlt.table(name="silver_tarjetas", table_properties={"quality": "silver"})
@dlt.expect_or_fail("tarjeta_id_not_null", "tarjeta_id IS NOT NULL")
def silver_tarjetas():
    # Deduplicamos quedándonos con el registro más reciente por tarjeta
    from pyspark.sql.window import Window
    w = Window.partitionBy("tarjeta_id").orderBy(F.desc("_ingest_timestamp"))
    return (
        spark.read.table("banking_bronze.raw_tarjetas")
        .withColumn("_rn", F.row_number().over(w))
        .filter("_rn = 1")
        .drop("_rn")
        .withColumn("estado_tarjeta",
            F.when(F.col("estado_cod").isin("A", "01"), "ACTIVA")
             .when(F.col("estado_cod").isin("B", "02"), "BLOQUEADA")
             .when(F.col("estado_cod").isin("C", "03"), "CANCELADA")
             .otherwise("DESCONOCIDO")
        )
    )