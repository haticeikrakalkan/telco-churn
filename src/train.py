from pathlib import Path

import joblib
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV, train_test_split

from src.data import load_data
from src.features import build_preprocessor, clean_data, split_features_target

MODEL_PATH = Path(__file__).resolve().parent.parent / "model.joblib"

PARAM_GRID = {
    "model__n_estimators": [50, 100, 200, 300, 500],
    "model__learning_rate": [0.001, 0.01, 0.05, 0.1, 0.5, 1.0],
}


def main():
    df = clean_data(load_data())
    X, y = split_features_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=45, stratify=y
    )

    pipeline = ImbPipeline([
        ("prep", build_preprocessor(X)),
        ("smote", SMOTE(random_state=42)),
        ("model", AdaBoostClassifier(random_state=42)),
    ])

    search = RandomizedSearchCV(
        pipeline,
        param_distributions=PARAM_GRID,
        n_iter=20,
        cv=5,
        scoring="f1",
        n_jobs=-1,
        random_state=45,
    )
    search.fit(X_train, y_train)
    print("En iyi parametreler:", search.best_params_)

    best = search.best_estimator_
    y_pred = best.predict(X_test)
    print(classification_report(y_test, y_pred))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

    joblib.dump(best, MODEL_PATH)
    print(f"Model kaydedildi: {MODEL_PATH}")


if __name__ == "__main__":
    main()