"""
Module: train
Description: Trains the final production xgboost model using optimized parameters and registers the trained model directly into the unity catalog.
"""
import sys
import logging
import yaml
import numpy as np
from typing import Dict, Any
from pyspark.sql import SparkSession
import pyspark.sql.types as T
from pyspark.ml import Pipeline 
from pyspark.ml.feature import StringIndexer, VectorAssembler, OneHotEncoder
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import mlflow

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Loads the configuration file from the given path and returns it as a dictionary.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        return config 

def run_final_training(config_path: str, params_path: str) -> None:
    spark = SparkSession.builder.getOrCreate()
    config = load_config(config_path)
    catalog = config["unity_catalog"]["catalog_name"]
    schema = config["unity_catalog"]["schema_name"]
    target_label = config["ml_settings"]["target_column"].lower().replace(" ", "_")

    gold_ml_matrix = f'{catalog}.{schema}.{config["tables"]["processed_training_data"]}'
    logger.info(f"Loading data for final training from {gold_ml_matrix}")

    df_raw = spark.table(gold_ml_matrix)
    drop_cols = ["customer_id"]
    df_filtered = df_raw.drop(*[c for c in drop_cols if c in df_raw.columns])

    # Construct categorical and numerical feature transformation
    categorical_cols = [f.name for f in df_filtered.schema.fields if isinstance(f.dataType, T.StringType) and f.name != target_label]
    numerical_cols = [f.name for f in df_filtered.schema.fields if isinstance(f.dataType, (T.IntegerType, T.DoubleType, T.FloatType, T.LongType)) and f.name != target_label]

    stages = []
    if categorical_cols:
        indexed_cols = [c + "_indexed" for c in categorical_cols]
        # encoded_cols = [c + "_encoded" for c in categorical_cols]
        stages.append(StringIndexer(inputCols=categorical_cols, outputCols=indexed_cols, handleInvalid="keep"))
        # stages.append(OneHotEncoder(inputCols=indexed_cols, outputCols=encoded_cols))
        assembler_inputs = indexed_cols + numerical_cols
    else:
        assembler_inputs = numerical_cols

    stages.append(VectorAssembler(inputCols=assembler_inputs, outputCol="features", handleInvalid="skip"))

    # Pre-transform the data to speed up the tuning search iterations
    logger.info("Executing spark transformation stages...")

    train_raw, val_raw = df_filtered.randomSplit([0.8, 0.2], seed=42)
    pre_pipeline = Pipeline(stages=stages)
    pipeline_model = pre_pipeline.fit(train_raw)

    pipeline_save_path = f"/Volumes/{catalog}/{schema}/pipeline_artifacts/feature_pipeline"
    spark.sql(f"CREATE VOLUME IF NOT EXISTS {catalog}.{schema}.pipeline_artifacts")
    pipeline_model.save(pipeline_save_path)
    logger.info(f"Fitted feature pipeline saved to: {pipeline_save_path}")

    train_df = pipeline_model.transform(train_raw).toPandas()
    val_df = pipeline_model.transform(val_raw).toPandas()

    X_train = np.array(train_df["features"].apply(lambda x: x.toArray()).tolist())
    y_train = np.array(train_df[target_label])
    X_val = np.array(val_df["features"].apply(lambda x: x.toArray()).tolist())
    y_val = np.array(val_df[target_label])

    run_name = "XGBoost_Production_Propensity_Model"
    with mlflow.start_run(run_name=run_name) as run:
        logger.info("Training final XGBoost classifier with tuned hyper-parameters...")

        tuned_params = load_config(params_path)
        best_params = tuned_params["best_xgboost_params"]
        model_args = {**best_params, "eval_metric": "auc", "enable_categorical": True}
        xgb_model = XGBClassifier(**model_args)
        xgb_model.fit(X_train, y_train)

        # Calculate validations
        val_probs = xgb_model.predict_proba(X_val)[:, 1]
        val_preds = xgb_model.predict(X_val)
        final_roc_auc = roc_auc_score(y_val, val_probs)
        final_accuracy = accuracy_score(y_val, val_preds)
        final_f1 = f1_score(y_val, val_preds)
        final_precision = precision_score(y_val, val_preds)
        final_recall = recall_score(y_val, val_preds)
        # final_confusion_matrix = confusion_matrix(y_val, val_preds)

        # Log metrics
        logger.info("Logging metrics...")
        mlflow.log_metric("val_roc_auc", final_roc_auc)
        mlflow.log_metric("val_accuracy", final_accuracy)
        mlflow.log_metric("val_f1", final_f1)
        mlflow.log_metric("val_precision", final_precision)
        mlflow.log_metric("val_recall", final_recall)
        # mlflow.log_metric("val_confusion_matrix", final_confusion_matrix)

        # Log best params
        logger.info("Logging best params...")
        mlflow.log_params(best_params)

        # Log model
        logger.info("Logging model...")
        mlflow.xgboost.log_model(
            xgb_model=xgb_model,
            artifact_path="model",
            input_example=X_train[:3]
        )

        registered_model_name = f"{catalog}.{schema}.propensity_model_xgboost"
        logger.info(f"Registering production model inside Unity Catalog as: {registered_model_name}")
        
        model_version_details = mlflow.register_model(
            model_uri=f"runs:/{run.info.run_id}/model",
            name=registered_model_name
        )
        
        model_production_uri = f"models:/{registered_model_name}/{model_version_details.version}"
        output_metadata_path = "../config/best_model_meta.yaml"
        metadata_payload = {
            "best_model": {
                "name": registered_model_name,
                "version": int(model_version_details.version),
                "model_uri": model_production_uri,
                "run_id": run.info.run_id
            }
        }
        with open(output_metadata_path, "w") as f:
            yaml.dump(metadata_payload, f, default_flow_style=False)
        
        logger.info("Model registration completed successfully.")

if __name__ == "__main__":
    run_final_training(
    config_path="../config/pipeline_config.yaml", 
    params_path="../config/best_xgboost_params.yaml"
)