from dataclasses import dataclass, field


@dataclass()
class TrainParams:
    model_type: str = field(default="RandomForestClassifier")
    random_state: int = field(default=42)
