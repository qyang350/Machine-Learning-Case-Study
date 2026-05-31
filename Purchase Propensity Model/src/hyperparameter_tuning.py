"""
Module: hyperparameter_tuning
Description: Runs an automated hyperopt space search loop to discover optimized XGBoost parameters, logging all outputs to MLflow.
"""

import sys
import logging 
import yaml
import numpy as np
from typing import Dict, Any
from pyspark.sql import SparkSession
import pyspark.sql.types as T
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK, space_eval
import mlflow

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def run_hyperparameter_tuning(config_path: str) -> None:
    spark = SparkSession.builder.getOrCreate()
    config = load_config(config_path)

    catalog = config["unity_catalog"]["catalog_name"]
    schema = config["unity_catalog"]["schema_name"]
    target_label = config["ml_settings"]["target_column"].lower().replace(" ", "_")

    gold_ml_matrix = f'{catalog}.{schema}.{config["tables"]["processed_training_data"]}'

    logger.info(f"Loading training data for hyperparameter tuning from {gold_ml_matrix}")
    df_raw = spark.table(gold_ml_matrix)

    # Isolate valid modeling columns and drop identity metadata
    drop_cols = ["customer_id"]
    df_filtered = df_raw.drop(*[c for c in drop_cols if c in df_raw.columns])

    # Process features dynamically by data type
    categorical_cols = [f.name for f in df_filtered.schema.fields if isinstance(f.dataType, T.StringType) and f.name != target_label]
    numerical_cols = [f.name for f in df_filtered.schema.fields if isinstance(f.dataType, (T.IntegerType, T.DoubleType, T.FloatType, T.LongType)) and f.name != target_label]

    stages = []
    if categorical_cols:
        indexed_cols = [c + "_indexed" for c in categorical_cols]
        # encoded_cols = [c + "_encoded" for c in categorical_cols]
        stages.append(StringIndexer(inputCols=categorical_cols, outputCols=indexed_cols, handleInvalid="keep"))
        # stages.append(OneHotEncoder(inputCols=indexed_cols, outputCols=encoded_cols))
        # assembler_inputs = encoded_cols + numerical_cols
        assembler_inputs = indexed_cols + numerical_cols
    else:
        assembler_inputs = numerical_cols

    stages.append(VectorAssembler(inputCols=assembler_inputs, outputCol="features", handleInvalid="skip"))

    logger.info("Executing spark transformation stages...")
    train_raw, val_raw = df_filtered.randomSplit([0.8, 0.2], seed=42)
    pre_pipeline = Pipeline(stages=stages)
    pipeline_model = pre_pipeline.fit(train_raw)
    train_processed = pipeline_model.transform(train_raw)
    val_processed = pipeline_model.transform(val_raw)

    train_df = train_processed.toPandas()
    val_df = val_processed.toPandas()

    X_train = np.array(train_df["features"].apply(lambda x: x.toArray()).tolist())
    y_train = np.array(train_df[target_label])
    X_val = np.array(val_df["features"].apply(lambda x: x.toArray()).tolist())
    y_val = np.array(val_df[target_label])
 
    # Define searching space
    search_space = {
        "max_depth": hp.choice("max_depth", [4,5,6,8]),
        "learning_rate": hp.loguniform("learning_rate", np.log(0.0001), np.log(0.1)),
        "n_estimators": hp.choice("n_estimators", [50, 100, 150, 200])
    }

    # Define the objective function
    def objective(params: Dict[str, Any]):
        with mlflow.start_run(nested=True): # launch a child run inside an already active parent run
            xgb = XGBClassifier(
                eval_metric="auc",
                enable_categorical=True,
                **params
            )

            model = xgb.fit(X_train, y_train)
            probs = model.predict_proba(X_val)[:, 1]
            roc_auc = roc_auc_score(y_val, probs)

            return {"loss": -roc_auc, "status": STATUS_OK}
    
    # Enable MLflow tracking
    # mlflow.pyspark.ml.autolog(log_models=False) # Turn off model saving inside the loop to save space
    with mlflow.start_run(run_name="XGBoost_Hyperopt_Optimization_Propensity_Model") as parent_run:
        logger.info("Initializing hyperopt distributed parameter search space execution...")

        trials = Trials()
        raw_best_params = fmin(
            fn=objective,
            space=search_space,
            algo=tpe.suggest,
            max_evals=8, 
            trials=trials
        )
        
        resolved_best_params = space_eval(search_space, raw_best_params)
        logger.info(f"Hyperparameter optimization complete. Wining combinations discovered: {resolved_best_params}")
        
        # Clean numpy datatypes so they can safely serialize to standard YAML format
        cleaned_params = {}
        for key, val in resolved_best_params.items():
            if isinstance(val, (np.float32, np.float64)):
                cleaned_params[key] = float(val)
            elif isinstance(val, (np.int32, np.int64)):
                cleaned_params[key] = int(val)
            else:
                cleaned_params[key] = val

        output_yaml_path = "../config/best_xgboost_params.yaml"
        with open(output_yaml_path, "w") as f:
            yaml.dump({"best_xgboost_params": cleaned_params}, f, default_flow_style=False)
        
        logger.info(f"Successfully exported best parameters to file system at: {output_yaml_path}")
    
if __name__ == "__main__":
    run_hyperparameter_tuning("../config/pipeline_config.yaml")