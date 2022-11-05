import os
from typing import List
from py._path.local import LocalPath
from project.train_pipeline import train_pipeline
from project.entities import (
    FeatureParams,
    SplitParams,
    TrainParams,
    TrainingPipelineParams,
)


def test_train_e2e(
    tmpdir: LocalPath,
    dataset_path: str,
    model_path: str,
    categorical_features: List[str],
    numerical_features: List[str],
    target_col: str,
    features_to_drop: List[str],
):
    expected_metrics_path = tmpdir.join("metrics.json")
    params = TrainingPipelineParams(
        input_data_path=dataset_path,
        output_model_path=model_path,
        metrics_path=expected_metrics_path,
        split_params=SplitParams(val_size=0.2, random_state=4),
        feature_params=FeatureParams(
            numerical_features=numerical_features,
            categorical_features=categorical_features,
            target_col=target_col,
            features_to_drop=features_to_drop,
        ),
        train_params=TrainParams(model_type="RandomForestClassifier"),
    )
    real_model_path, metrics = train_pipeline(params)
    assert metrics["accuracy"] > 0
    assert os.path.exists(real_model_path)
    assert os.path.exists(params.metrics_path)
