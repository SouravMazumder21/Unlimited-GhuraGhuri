from flask import Flask, request, jsonify  # type: ignore
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained ML model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return jsonify({"message": "Welcome to Unlimited GhuraGhuri API!"})

# API Endpoint for Prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # Extract input features
        travel_time = float(data.get("travel_time", 0))
        time_to_reach = float(data.get("time_to_reach", 0))

        # Convert input to numpy array
        feature_vector = np.array([[travel_time, time_to_reach]])

        # Make prediction
        prediction = model.predict(feature_vector)[0]

        return jsonify({"recommended": bool(prediction)})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
