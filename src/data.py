from pathlib import Path

import pandas as pd

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "Telco_Customer_Churn.csv"


def load_data(path=DATA_PATH):
    return pd.read_csv(path)