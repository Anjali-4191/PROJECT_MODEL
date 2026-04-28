import os
import pickle
import numpy as np
from flask import Flask, render_template, request   # ✅ correct (lowercase flask)

app = Flask(__name__)

# Load model (make sure file name is exactly same in your repo)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")

with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)


# Home route
@app.route("/")
def home():
    return render_template("index.html")


# Prediction route
@app.route("/predict", methods=["POST"])
def predict():
    try:
        respiratory_rate = float(request.form.get("respiratory_rate", 0))
        crp = float(request.form.get("crp", 0))
        age = float(request.form.get("age", 0))
        sgot = float(request.form.get("sgot", 0))
        urea = float(request.form.get("urea", 0))
        tlc = float(request.form.get("tlc", 0))
        corads = float(request.form.get("corads", 0))
        platelet = float(request.form.get("platelet", 0))
        bmi = float(request.form.get("bmi", 0))
        qsofa = float(request.form.get("qsofa", 0))

        # Prepare data
        data = np.array([
            respiratory_rate, crp, age, sgot, urea,
            tlc, corads, platelet, bmi, qsofa
        ]).reshape(1, -1)

        # Prediction
        prediction = model.predict(data)[0]
        result = "High Risk" if prediction == 1 else "Low Risk"

        return render_template("index.html", prediction_text=f"Result: {result}")

    except Exception as e:
        return f"Error occurred: {str(e)}"


# Run app (for local testing only)
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
    