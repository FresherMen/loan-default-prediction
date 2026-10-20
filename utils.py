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

def preprocessdata(Gender, Married, Education, Self_Employed, ApplicantIncome,
                   CoapplicantIncome, LoanAmount, Loan_Amount_Term, Credit_History,
                   Property_Area):
    """
    Transforms applicant inputs into feature vectors and runs inference
    against the serialized loan approval classifier.
    """
    test_data = [[
        float(Gender),
        float(Married),
        float(Education),
        float(Self_Employed),
        float(ApplicantIncome),
        float(CoapplicantIncome),
        float(LoanAmount),
        float(Loan_Amount_Term),
        float(Credit_History),
        float(Property_Area)
    ]]
    trained_model = get_model()
    prediction = trained_model.predict(test_data)
    return int(prediction[0]) if len(prediction) > 0 else 0