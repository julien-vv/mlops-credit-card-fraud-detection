from flask import Flask, request, jsonify
from xgboost import XGBClassifier
from preprocess_data import *
import mlflow

## HELP ##
# cd src
# python app.py
# curl http://localhost:8501/
# curl -X POST http://localhost:5000/predict -H "Content-Type: application/json" --data @sample_class_1.json


# Initialize the Flask app
app = Flask(__name__)

# Load the model in local
model = XGBClassifier()
model.load_model("model.json")

@app.route('/', methods=['GET'])
def home():
    return "Flask app is running."

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()  # Get JSON data from the request
    if not data:
        return jsonify({'error': 'No input data provided'}), 400

    # Check if data is a dict (single observation)
    if isinstance(data, dict):
        # Convert single dict to a list of dicts to create DataFrame correctly
        input_data = pd.DataFrame([data])
    else:
        # Data is already a list of dicts
        input_data = pd.DataFrame(data)

    # Processed the data
    processed_data = preprocessing_data(input_data)


    # Make the prediction using the model
    prediction = model.predict(processed_data)

    # Return prediction as JSON
    return jsonify({'prediction': prediction.tolist()})


if __name__ == "__main__":
    app.run(debug=True)
