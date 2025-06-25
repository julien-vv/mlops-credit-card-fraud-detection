<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body>

<h1>MLOps Credit Card Fraud Detection</h1>

<p>The <strong>Card Fraud Detection</strong> project aims to build a machine learning model capable of detecting credit card fraud. The goal is to deploy a supervised learning model that analyzes transaction data and identifies suspicious activity. This project integrates various tools for continuous integration, deployment, data versioning, and model versioning to ensure efficient tracking of every step in deploying the model into production.</p>

<p>The project contains a <strong>test</strong> folder, which includes unit tests for the functions defined in the source files. These unit tests are automated through <strong>CI/CD</strong> using <strong>GitHub Actions</strong>. Data versioning is managed manually using <strong>DVC</strong>, while the model itself is versioned with <strong>MLflow</strong>.</p>

<h3>Folder Structure:</h3>
<p>The <code>src</code> folder contains the essential Python files for training, evaluating, and deploying the model:</p>
<ul>
    <li><strong>preprocess_data.py</strong>: Contains functions for preprocessing the data.</li>
    <li><strong>evaluate_model.py</strong>: Includes functions to evaluate the model on test data and generate metrics such as the confusion matrix, ROC AUC score, and AUPRC.</li>
    <li><strong>train_model.py</strong>: Contains functions to train the model with different algorithms (Logistic Regression, Random Forest, XGBoost).</li>
    <li><strong>main.py</strong>: The main script that ties together all the steps (data loading, preprocessing, model training, and evaluation).</li>
    <li><strong>app.py</strong>: Uses <strong>Flask</strong> to deploy an API hosting the model.</li>
    <li><strong>frontend.py</strong>: Uses <strong>Streamlit</strong> to deploy a web app to make a request to the API for prediction.</li>
    <li><strong>db.py</strong>: Uses <strong>SQLite3</strong> to store model predictions in a lightweight database.</li>
</ul>

<h3>Dataset</h3>
<p>You can download the dataset used for this project from the following Kaggle link: <a href="https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud" target="_blank">Credit Card Fraud Detection Dataset</a>.
<p>Once downloaded, place the <code>creditcard.csv</code> file into the <code>data</code> folder in the project directory. This is required for running the model and the full pipeline.</p>

<h3>Run the Main Script</h3>
<p>To run the main script, which ties all components together (data loading, preprocessing, model training, and evaluation), use the command:</p>
<pre><code>mlflow ui</code></pre>
<pre><code>python src/main.py</code></pre>

<h3>Run the Flask Backend</h3>
<p>To start the Flask backend, run the following command:</p>
<pre><code>python src/app.py</code></pre>
<p>This will launch the Flask API locally on <strong>http://localhost:5002</strong>.</p>

<h3>Run the Streamlit Frontend</h3>
<p>To run the Streamlit frontend, use the following command:</p>
<pre><code>streamlit run src/frontend.py</code></pre>
<p>This opens a web-based UI that interacts with the Flask backend.</p>

<h3>Testing the API with curl</h3>

<p>You can test the API endpoints using <strong>curl</strong>. Run these commands in Git Bash or any compatible terminal:</p>

<h4>Check if the service is up</h4>
<pre><code>curl http://localhost:5002/</code></pre>
<p><strong>Expected Output:</strong> "Flask app is running."</p>

<h4>Test with Valid Input</h4>
<p>To test with a valid input, use the following command (ensure that the <code>sample_predictions_1.json</code> file is present in the <code>data</code> folder):</p>
<pre><code>curl -X POST http://localhost:5002/predict -H "Content-Type: application/json" --data @data/sample_predictions_1.json</code></pre>
<p><strong>Expected Output:</strong> A JSON response indicating prediction.</p>

<h4>Additional Valid Tests</h4>
<p>Run these commands for additional valid tests:</p>
<pre><code>curl -X POST http://localhost:5002/predict -H "Content-Type: application/json" --data @data/sample_predictions_0.json</code></pre>

<pre><code>curl -X POST http://localhost:5002/predict -H "Content-Type: application/json" --data @data/sample_predictions_1_0.json</code></pre>

<h4>Test with Invalid Input</h4>
<p>To test with invalid input (empty JSON payload), run the following:</p>
<pre><code>curl -X POST http://localhost:5002/predict -H "Content-Type: application/json" --data '{}'</code></pre>
<p><strong>Expected Output:</strong> An error message indicating that the input data is missing or invalid.</p>

<h3>Docker</h3>

<p>The project uses <strong>Docker</strong> to build a containerized image of the application. This ensures that the code works consistently on any machine, whether running Linux or Windows. The <strong>Dockerfile</strong> defines the Docker image, which includes Python dependency installation and code cloning. The Docker image is built and validated through the CI/CD pipeline, ensuring a seamless image creation process across different environments.</p>

<h4>Docker Commands:</h4>
<p><strong>Build the Docker image:</strong></p>
<pre><code>docker build -t flask-fraud-app .</code></pre>

<p><strong>Run the Docker container to launch the application:</strong></p>
<p>Note that running the Docker container will automatically start the <code>src/app.py</code> script, which runs the Flask API.</p>
<pre><code>docker run --rm -it -p 5000:5000 -v "${PWD}\\data:/app/data" --name fraud-api flask-fraud-app</code></pre>

<h3>Points for Improvement:</h3>

<p>Although the project is complete, there are several areas that can be improved:</p>
<ul>
    <li>Adding a configuration file to make the application more flexible, allowing easy adjustments to the model's parameters for fraud detection.</li>
    <li>Expanding the unit tests to provide more comprehensive code coverage.</li>
    <li>Containerizing other parts of the code, such as preprocessing, training, evaluation, and the Flask app, to fully integrate the pipeline.</li>
    <li>Improving the performance of the model and optimizing the code for better efficiency.</li>
    <li>Enhancing the documentation to improve readability, maintainability, and ease of contribution for other developers.</li>
</ul>

</body>
</html>
