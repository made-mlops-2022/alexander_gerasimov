from project.data.make_dataset import read_data, split_train_val_data
from project.entities import SplitParams


def test_load_dataset(dataset_path: str, target_col: str):
    data = read_data(dataset_path)
    assert 40 == len(data)
    assert target_col in data.keys()


def test_split_dataset(dataset_path: str):
    val_size = 0.2
    splitting_params = SplitParams(
        random_state=42,
        val_size=val_size,
    )
    data = read_data(dataset_path)
    train, val = split_train_val_data(data, splitting_params)
    assert 32 == train.shape[0]
    assert 8 == val.shape[0]
