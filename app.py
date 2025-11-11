from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib, os

app = Flask(__name__)

# Load model
model_path = os.path.join(os.path.dirname(__file__), "final_calories_model.joblib")
bundle = joblib.load(model_path)
model = bundle["model"]
scaler = bundle["scaler"]
features = bundle["features"]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_api():
    data = request.get_json()
    df = pd.DataFrame([data])
    X_scaled = scaler.transform(df[features])
    prediction = model.predict(X_scaled)[0]
    return jsonify({"Predicted_Calories": float(prediction)})

@app.route('/predict_form', methods=['POST'])
def predict_form():
    # Get data from HTML form
    data = {key: float(value) for key, value in request.form.items()}

    # Compute BMI dynamically if needed
    data['BMI'] = data['Weight'] / ((data['Height'] / 100) ** 2)

    # Convert to DataFrame
    df = pd.DataFrame([data])
    X_scaled = scaler.transform(df[features])
    prediction = model.predict(X_scaled)[0]

    # Render same page with result
    return render_template('index.html', prediction=round(prediction, 2))

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)
