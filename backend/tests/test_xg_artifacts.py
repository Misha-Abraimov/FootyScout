from pathlib import Path

import joblib
import pandas as pd
import pytest
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.artifacts


def test_production_xg_artifacts_load_and_oof_integrity() -> None:
    preprocessor = joblib.load(ROOT / "models" / "xg_preprocessor.joblib")
    model = XGBClassifier()
    model.load_model(ROOT / "models" / "xg_model.json")
    oof = pd.read_parquet(ROOT / "data" / "processed" / "shot_oof_predictions.parquet")
    assert hasattr(preprocessor, "transform")
    assert hasattr(model, "predict_proba")
    assert len(oof) == 5_545
    assert oof["shot_id"].is_unique
    assert oof["expected_goal"].notna().all()
    assert oof["expected_goal"].between(0, 1).all()
    assert oof["fold"].nunique() == 5
