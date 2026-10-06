from src.data import load_data
from src.features import TARGET, clean_data, split_features_target


def test_clean_data_shape_and_no_missing():
    df = clean_data(load_data())
    assert df.shape == (7043, 22)
    assert df.isnull().sum().sum() == 0


def test_clean_data_new_features_exist():
    df = clean_data(load_data())
    assert "total_services" in df.columns
    assert "is_automatic_payment" in df.columns


def test_target_is_binary():
    df = clean_data(load_data())
    _, y = split_features_target(df)
    assert set(y.unique()) == {0, 1}