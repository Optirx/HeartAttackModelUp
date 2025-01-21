from flask import Flask, request, jsonify
import pickle
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the trained Random Forest model
model_path = 'Trained_Random_Forest_Model.pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)

# Try to get feature names from the model
try:
    feature_labels = model.feature_names_in_
except AttributeError:
    feature_labels = None

# Expected number of features
expected_features = len(feature_labels) if feature_labels is not None else None

@app.route('/', methods=['GET'])
def home():
    return "Welcome to the Heart Attack Prediction API! Use the '/predict' endpoint to make predictions and '/features' to view the feature labels."

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from the request
        data = request.get_json()

        # Check if data is provided
        if not data:
            return jsonify({"error": "No input data provided"}), 400

        # Ensure input is in the correct format (list of feature values)
        features = data.get('features')
        if not features or not isinstance(features, list):
            return jsonify({"error": "Invalid input format. 'features' should be a list."}), 400

        # Check feature length
        if expected_features and len(features) != expected_features:
            return jsonify({"error": f"Input has {len(features)} features, but {expected_features} are expected."}), 400

        # Convert features into a numpy array for prediction
        input_data = np.array(features).reshape(1, -1)

        # Make a prediction using the model
        prediction = model.predict(input_data)[0]

        # Return the prediction result
        result = {
            "prediction": int(prediction),  # Convert to int for JSON compatibility
            "message": "Heart Attack Risk" if prediction == 1 else "No Heart Attack Risk"
        }
        return jsonify(result), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/features', methods=['GET'])
def get_features():
    try:
        if feature_labels is None:
            return jsonify({"error": "Feature labels are not available in the model. Ensure the model supports 'feature_names_in_'."}), 400
        return jsonify({"features": list(feature_labels)}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
