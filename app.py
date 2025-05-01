from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load model and scaler
model, scaler = joblib.load("model/diabetes_model.pkl")


@app.route('/')
def home():
    return "Diabetes Prediction API is running!"


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json["features"]
        input_data = np.array(data).reshape(1, -1)
        input_data = scaler.transform(input_data)

        prediction = model.predict(input_data)[0]
        return jsonify({"prediction": int(prediction)})

    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == '__main__':
    app.run(port=5000, debug=True)
