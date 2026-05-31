"""
Module: feature_generation
Description: Reads pre-aggregated customer vectors from Unity Catalog, applies final filtering/null imputation, and structure the Gold table.
"""

import sys
import re
import logging
from typing import Dict, Any
import yaml
from pyspark.sql import SparkSession, DataFrame
import pyspark.sql.functions as F
import pyspark.sql.types as T 

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def load_config(config_path: str) -> Dict[str, Any]:
    "Loads pipeline configurations dynamically based on runtime environment."
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def clean_and_isolate_features(df: DataFrame, target_col: str) -> DataFrame:
    """
    Filters out customer metadata and target leakage columns.
    Enforces null-safety across all numerical and categorical features.
    """
    logger.info("Executing production feature engineering and leakage controls...")

    # 1. TARGET LEAKAGE & METADATA CONTROL
    leakage_and_metadata = [
        "Customer Name",
        "Last Purchase Date", # redundant with Days Since Last Purchase
        "Synthetic True Propensity",
        "Next 30D Spend"
    ]

    df_filtered = df.drop(*[col for col in leakage_and_metadata if col in df.columns])

    # 2. SEPARATE FEATURE TYPES FOR DYNAMIC IMPUTATION
    ignore_from_imputation = ["Customer ID", target_col]
    numerical_cols = [
        f.name for f in df_filtered.schema.fields 
        if isinstance(f.dataType, (T.IntegerType, T.DoubleType, T.FloatType, T.LongType)) and f.name not in ignore_from_imputation
    ]

    categorical_cols = [
        f.name for f in df_filtered.schema.fields
        if isinstance(f.dataType, T.StringType) and f.name not in ignore_from_imputation
    ]

    logger.info(f"Identified numerical features: {numerical_cols}")
    logger.info(f"Identified categorical features: {categorical_cols}")

    # 3. NULL MANAGEMENT
    num_defaults = {
        col: 0.0 for col in numerical_cols
    } # fill continuous metric blank with 0

    cat_defaults = {
        col: "UNKNOWN" for col in categorical_cols
    } # fill categorical missing values with an explicit string bucket

    # Apply changes to the distributed dataframe
    df_imputed = df_filtered.fillna(num_defaults).fillna(cat_defaults)

    return df_imputed

def standardize_column_names(df: DataFrame) -> DataFrame:
    """
    Standardizes column names to remove invalid characters.
    """
    cleaned_cols = []
    for col in df.columns:
        clean_name = re.sub(r"[ ,;{}()\n\t=]+", "_", col)
        clean_name = clean_name.strip("_") 
        clean_name = clean_name.lower()

        cleaned_cols.append(F.col(col).alias(clean_name))
    
    return df.select(*cleaned_cols)

def run_feature_pipeline(config_path: str) -> None:
    "Main pipeline execution loop called by Databricks Asset Bundles."
    spark = SparkSession.builder.getOrCreate()

    config = load_config(config_path=config_path)
    catalog = config["unity_catalog"]["catalog_name"]
    schema = config["unity_catalog"]["schema_name"]
    target_label = config["ml_settings"]["target_column"]

    source_table = f"{catalog}.{schema}.{config["tables"]["training_data"]}"
    gold_table = f"{catalog}.{schema}.{config["tables"]["processed_training_data"]}"

    logger.info(f"Loading master aggregated dataset from Unity Catalog: {source_table}")

    raw_df = spark.table(source_table)

    # Process dataset
    ml_ready_matrix = clean_and_isolate_features(raw_df, target_col=target_label)

    # Standardize column names
    final_clean_matrix = standardize_column_names(ml_ready_matrix)

    # Save optimized table back to Unity Catalog gold layer
    logger.info(f"Saving final ML feature matrix to Gold Table: {gold_table}")
    final_clean_matrix.write.format("delta").mode("overwrite").saveAsTable(gold_table)
    
    # Apple storage clustering to speed up subsequent parallel training algorithms
    logger.info(f"Running index optimizations on {gold_table}")
    spark.sql(f"OPTIMIZE {gold_table} ZORDER BY (customer_id)")

    logger.info("Gold feature generation pipeline completed successfully.")

if __name__ == "__main__":
    run_feature_pipeline("../config/pipeline_config.yaml")