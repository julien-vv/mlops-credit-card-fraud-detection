from flask import Flask, request, jsonify
import pandas as pd
import mlflow.pyfunc

# Initialize the Flask app
app = Flask(__name__)

model_uri = "../mlartifacts/0/7afc90be810947a985a50adc590ad769/artifacts/model"
model = mlflow.pyfunc.load_model(model_uri)

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
