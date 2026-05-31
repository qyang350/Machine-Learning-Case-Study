"""
Module: batch_inference
Description: Loads the production model from unity catalog, applies transformations, runs high-throughput batch scoring, and writes results back to unity catalog.
"""
import sys
import logging 
import re
import yaml
import pandas as pd
import mlflow 
import numpy as np
from typing import Dict, Any
from pyspark.sql import SparkSession, DataFrame
import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from mlflow.tracking import MlflowClient
from datetime import datetime, timezone

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config(config_path:str) -> Dict[str, Any]:
    """
    Loads the configuration file from the given path.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        return config 
    
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

    source_table = f"{catalog}.{schema}.{config["tables"]["scoring_data"]}"
    gold_table = f"{catalog}.{schema}.{config["tables"]["processed_scoring_data"]}"

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

    logger.info("Gold feature generation pipeline for scoring data completed successfully.")

def run_batch_inference(config_path:str, best_model_meta_path:str) -> None:
    "Orchestrate the batch inference process."
    # Run feature generation pipeline for scoring data
    logger.info(f"Preprocessing scoring data...")
    run_feature_pipeline(config_path=config_path)

    # Initialize env and configs
    spark = SparkSession.builder.getOrCreate()
    config = load_config(config_path)

    catalog = config["unity_catalog"]["catalog_name"]
    schema = config["unity_catalog"]["schema_name"]
    target_label = config["ml_settings"]["target_column"].lower().replace(" ", "_")

    gold_ml_matrix = f'{catalog}.{schema}.{config["tables"]["processed_scoring_data"]}'
    prediction_output_table = f"{catalog}.{schema}.customer_propensity_predictions" 

    df_raw = spark.table(gold_ml_matrix)
    identity_col ="customer_id"
    customer_ids = df_raw.select(identity_col)
    df_filtered = df_raw.drop(*[c for c in [target_label] if c in df_raw.columns])

    # Load model
    best_model_meta = load_config(best_model_meta_path)
    model_uri = best_model_meta["best_model"]["model_uri"]
    model_version = best_model_meta["best_model"]["version"]
    model_name = best_model_meta["best_model"]["name"]
    logger.info(f"Loading registered model: {model_name} @ version on {model_version} {model_uri}")

    client = MlflowClient()
    version_info = client.get_model_version(model_name, str(model_version))
    if version_info.status != "READY":
        raise RuntimeError(
            f"Model {model_name} v{model_version} is not in READY state (current: {version_info.status}). "
            "Aborting inference to avoid scoring with a degraded model."
        )
    
    model = mlflow.xgboost.load_model(model_uri)
    logger.info("Model loaded successfully.")

    # Feature transformation
    pipeline_path = f"/Volumes/{catalog}/{schema}/pipeline_artifacts/feature_pipeline"
    logger.info(f"Loading fitted feature pipeline from: {pipeline_path}")
    pipeline_model = PipelineModel.load(pipeline_path)
    
    processed_df = pipeline_model.transform(df_filtered)
    inference_pdf = processed_df.select("features").toPandas()
    X_inference = np.array(inference_pdf["features"].apply(lambda x: x.toArray()).tolist())

    # Score
    logger.info("Running batch scoring...")
    propensity_scores = model.predict_proba(X_inference)[:, 1]
    propensity_labels = model.predict(X_inference)

    scored_at = datetime.now(timezone.utc).isoformat()
    scores_pdf = pd.DataFrame({
            "propensity_score": propensity_scores.astype(float),
            "propensity_label": propensity_labels.astype(int),
            "model_name": model_name,
            "model_version": model_version,
            "scored_at": scored_at
        })

    # Re-attach customer_id by positional index
    customer_ids_pdf = customer_ids.toPandas()
    scores_pdf.insert(0, "customer_id", customer_ids_pdf["customer_id"].values)
    output_df = spark.createDataFrame(scores_pdf)

    # Write to table
    logger.info(f"Writing scored records to: {prediction_output_table}")
    (
        output_df.write
        .format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(prediction_output_table)
    )
     
if __name__ == "__main__":
    run_batch_inference(
        config_path="../config/pipeline_config.yaml", 
        best_model_meta_path="../config/best_model_meta.yaml"
    )