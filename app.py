import os
from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load model and scaler
model = joblib.load("models/model.pkl")
scaler = joblib.load("models/scaler.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        # Extract features from form input
        age = float(request.form["age"])
        sex = int(request.form["sex"])
        cp = int(request.form["cp"])
        trestbps = float(request.form["trestbps"])
        chol = float(request.form["chol"])
        fbs = int(request.form["fbs"])
        restecg = int(request.form["restecg"])
        thalach = float(request.form["thalach"])
        exang = int(request.form["exang"])
        oldpeak = float(request.form["oldpeak"])
        slope = int(request.form["slope"])
        ca = int(request.form["ca"])
        thal = int(request.form["thal"])

        # Create input vector matching exact UCI feature order
        features = np.array(
            [
                [
                    age,
                    sex,
                    cp,
                    trestbps,
                    chol,
                    fbs,
                    restecg,
                    thalach,
                    exang,
                    oldpeak,
                    slope,
                    ca,
                    thal,
                ]
            ]
        )

        # Scale inputs and calculate probability
        features_scaled = scaler.transform(features)
        risk_probability = model.predict_proba(features_scaled)[0][1] * 100
        risk_score = round(risk_probability, 1)

        # Categorize output
        if risk_score >= 70:
            category = "High Risk"
            alert_class = "danger"
            advice = "Strong indicators detected. Please consult a cardiologist for an ECG/stress test evaluation."
        elif risk_score >= 35:
            category = "Moderate Risk"
            alert_class = "warning"
            advice = "Elevated metrics observed. Consider scheduling a routine checkup and reviewing lifestyle factors."
        else:
            category = "Low Risk"
            alert_class = "success"
            advice = "Your clinical markers are currently within normal ranges. Keep up healthy habits!"

        return render_template(
            "result.html",
            score=risk_score,
            category=category,
            alert_class=alert_class,
            advice=advice,
        )


if __name__ == "__main__":
    app.run(debug=True)