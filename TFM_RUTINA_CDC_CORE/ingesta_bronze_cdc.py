# lakeflow_pipeline/bronze/ingest_bronze.py
import dlt
from pyspark.sql import functions as F
from pyspark.sql.types import *

JDBC_URL = "jdbc:sqlserver://<server>.database.windows.net:1433;database=<db>"
JDBC_OPTS = {
    "user": dbutils.secrets.get("kv-banking", "sql-user"),
    "password": dbutils.secrets.get("kv-banking", "sql-password"),
    "driver": "com.microsoft.sqlserver.jdbc.SQLServerDriver",
    # Paralelismo: particionamos por fecha para cargar en paralelo
    "partitionColumn": "fecha_transaccion",
    "lowerBound": "2020-01-01",
    "upperBound": "2025-12-31",
    "numPartitions": "48",   # ajustar según cluster
    "fetchsize": "50000"
}

# ── CUENTAS ────────────────────────────────────────────────────────
@dlt.table(
    name="raw_cuentas",
    comment="Carga inicial + CDC de cuentas desde Azure SQL (réplica mainframe)",
    table_properties={"quality": "bronze", "pipelines.reset.allowed": "true"},
    partition_cols=["year_partition"]
)
def raw_cuentas():
    return (
        spark.read.format("jdbc")
        .options(**JDBC_OPTS)
        .option("dbtable", "(SELECT *, YEAR(fecha_apertura) AS year_partition FROM dbo.CUENTAS) t")
        .load()
        .withColumn("_ingest_timestamp", F.current_timestamp())
        .withColumn("_source_system", F.lit("mainframe_sql_replica"))
        .withColumn("_batch_id", F.lit(dbutils.widgets.get("batch_id")))
    )

# ── TRANSACCIONES (tabla grande ~500M rows) ────────────────────────
@dlt.table(
    name="raw_transacciones",
    comment="Transacciones 5 años - carga incremental por watermark",
    table_properties={
        "quality": "bronze",
        "delta.autoOptimize.optimizeWrite": "true",
        "delta.autoOptimize.autoCompact": "true"
    },
    partition_cols=["year_partition", "month_partition"]
)
def raw_transacciones():
    # Para la carga incremental usamos el watermark de la última fecha procesada
    last_processed = spark.sql("""
        SELECT COALESCE(MAX(fecha_transaccion), '2020-01-01')
        FROM banking_bronze.raw_transacciones
    """).collect()[0][0]

    return (
        spark.read.format("jdbc")
        .options(**JDBC_OPTS)
        .option("dbtable", f"""
            (SELECT *,
                YEAR(fecha_transaccion) AS year_partition,
                MONTH(fecha_transaccion) AS month_partition
             FROM dbo.TRANSACCIONES
             WHERE fecha_transaccion > '{last_processed}'
               AND fecha_transaccion <= GETDATE()) t
        """)
        .load()
        .withColumn("_ingest_timestamp", F.current_timestamp())
        .withColumn("_source_system", F.lit("mainframe_sql_replica"))
    )