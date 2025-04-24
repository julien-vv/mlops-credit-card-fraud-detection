from flask import Flask, request, jsonify
import pandas as pd
import pickle
# Initialize the Flask app
app = Flask(__name__)

# Load the model from MLflow's registry (replace 'LogisticRegression_model' with your model name)
url = r'.\mlruns\0\fc6f65a104f9403a8b179d342b93d8ae\artifacts\RandomForest_model\model.pkl'
with open(url, 'rb') as f:
    model = pickle.load(f)


@app.route('/')
def home():
    return "Flask app is running."


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()  # Get JSON data from the request
    if not data:
        return jsonify({'error': 'No input data provided'}), 400

    # Convert data into a DataFrame for prediction
    input_data = pd.DataFrame(data)

    # Make the prediction using the model
    prediction = model.predict(input_data)

    # Return prediction as JSON
    return jsonify({'prediction': prediction.tolist()})


if __name__ == "__main__":
    app.run(debug=True)
