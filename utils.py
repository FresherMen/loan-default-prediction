import os
import joblib
import numpy as np

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")
_CACHED_MODEL = None

def get_model():
    """Singleton getter to cache model in memory instead of reloading from disk on each request."""
    global _CACHED_MODEL
    if _CACHED_MODEL is None:
        _CACHED_MODEL = joblib.load(MODEL_PATH)
    return _CACHED_MODEL

def _safe_float(val, default: float = 0.0) -> float:
    """Safely converts input values to float, falling back to default if empty or non-numeric."""
    try:
        if val is None or str(val).strip() == "":
            return default
        return float(val)
    except (ValueError, TypeError):
        return default

def preprocessdata(Gender, Married, Education, Self_Employed, ApplicantIncome,
                   CoapplicantIncome, LoanAmount, Loan_Amount_Term, Credit_History,
                   Property_Area):
    """
    Transforms applicant inputs into feature vectors and runs inference
    against the serialized loan approval classifier.
    """
    test_data = [[
        _safe_float(Gender, 1.0),
        _safe_float(Married, 1.0),
        _safe_float(Education, 1.0),
        _safe_float(Self_Employed, 0.0),
        _safe_float(ApplicantIncome, 0.0),
        _safe_float(CoapplicantIncome, 0.0),
        _safe_float(LoanAmount, 0.0),
        _safe_float(Loan_Amount_Term, 360.0),
        _safe_float(Credit_History, 1.0),
        _safe_float(Property_Area, 1.0)
    ]]
    trained_model = get_model()
    prediction = trained_model.predict(test_data)
    return int(prediction[0]) if len(prediction) > 0 else 0