import joblib
import numpy as np

def load_artifacts():
    model = joblib.load("thyroid_model.pkl")
    scaler = joblib.load("scaler.pkl")
    label_map = {0: "Negative", 1: "Positive"}
    return model, scaler, label_map

def predict_single(data_dict):
    model, scaler, label_map = load_artifacts()
    x = np.array(list(data_dict.values())).reshape(1, -1)
    x_scaled = scaler.transform(x)
    pred = model.predict(x_scaled)[0]
    prob = model.predict_proba(x_scaled).max() * 100
    return label_map[pred], round(prob, 2), prob
