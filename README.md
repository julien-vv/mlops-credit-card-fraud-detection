<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body>

<h1>Card Fraud Detection</h1>

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
</ul>

<h3>Docker</h3>

<p>The project uses <strong>Docker</strong> to build a containerized image of the application. This ensures that the code works consistently on any machine, whether running Linux or Windows. The <strong>Dockerfile</strong> defines the Docker image, which includes Python dependency installation, code cloning, and execution of the <strong>main.py</strong> script. The Docker image is built and validated through the CI/CD pipeline, ensuring a seamless image creation process across different environments.</p>

<h4>Docker Commands:</h4>
<p><strong>Build the Docker image:</strong></p>
<pre><code>docker build -t myfirstpythonapp .</code></pre>

<p><strong>Run the Docker container to launch the application:</strong></p>
<p>Note that running the Docker container will automatically start the <code>src/main.py</code> script, which performs all the necessary steps (data preprocessing, model training, and evaluation).</p>
<pre><code>docker run myfirstpythonapp</code></pre>

<h3>Points for Improvement:</h3>

<p>Although the project is complete, there are several areas that can be improved:</p>
<ul>
    <li>Adding a configuration file to make the application more flexible, allowing easy adjustments to the model's parameters for fraud detection.</li>
    <li>Expanding the unit tests to provide more comprehensive code coverage.</li>
    <li>Containerizing other parts of the code, such as preprocessing, training, evaluation, and the Flask app, to fully integrate the pipeline.</li>
    <li>Improving the performance of the model and optimizing the code for better efficiency.</li>
    <li>Enhancing the Flask API to store predictions, and extend its functionalities.</li>
    <li>Enhancing the documentation to improve readability, maintainability, and ease of contribution for other developers.</li>
</ul>

</body>
</html>
