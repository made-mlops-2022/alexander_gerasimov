import yaml
from dataclasses import dataclass
from marshmallow_dataclass import class_schema
from .feature_params import FeatureParams
from .split_params import SplitParams
from .train_params import TrainParams


@dataclass()
class TrainingPipelineParams:
    input_data_path: str
    output_model_path: str
    metrics_path: str
    split_params: SplitParams
    feature_params: FeatureParams
    train_params: TrainParams


TrainingPipelineParamsSchema = class_schema(TrainingPipelineParams)


def read_training_pipeline_params(path: str) -> TrainingPipelineParams:
    with open(path, "r") as input_stream:
        schema = TrainingPipelineParamsSchema()
        return schema.load(yaml.safe_load(input_stream))
