# mlops-credit-card-fraud-detection
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>mlops-credit-card-fraud-detection</title>
</head>
<body>

<h1>mlops-credit-card-fraud-detection</h1>

<p>The <strong>Card Fraud Detection</strong> project aims to build a machine learning model capable of detecting credit card fraud. It focuses on deploying a supervised learning model that analyzes transactions and identifies suspicious ones. This project integrates multiple tools for continuous integration, deployment, and data versioning to ensure smooth and efficient management of the code and the model in production.</p>

<p>The project contains a <strong>test</strong> folder with unit tests for the functions in the source files. These unit tests are automated in <strong>CI/CD</strong> using <strong>GitHub Actions</strong>. Data versioning is managed manually with <strong>DVC</strong>. The model is versioned using <strong>MLflow</strong>.</p>

<h3>Key Features:</h3>
<ul>
    <li><strong>Docker Image Construction</strong>: The project uses Docker to create a containerized image, ensuring consistent behavior across all machines.</li>
    <li><strong>Docker Image Testing</strong>: The image is tested by running a container and verifying that the API launches correctly.</li>
    <li><strong>Multi-platform Compatibility</strong>: To ensure compatibility across platforms, continuous integration runs in different environments, including both Linux and Windows, to ensure cross-platform compatibility. Moreover, the code is containerized using Docker, allowing it to run seamlessly on different machines.</li>
</ul>

<h3>Folder Structure:</h3>
<p>The <strong>src</strong> folder contains the necessary Python files for training, evaluating, and deploying the model:</p>
<ul>
    <li><strong>app.py</strong>: Uses <strong>Flask</strong> to deploy an API hosting the model.</li>
    <li><strong>preprocess_data.py</strong>: Contains functions for data preprocessing.</li>
    <li><strong>evaluate_model.py</strong>: Includes functions for evaluating the model on test data and generating metrics such as the confusion matrix, ROC AUC score, and AUPRC.</li>
    <li><strong>train_model.py</strong>: Contains functions to train the model with different algorithms (Logistic Regression, Random Forest, XGBoost).</li>
    <li><strong>main.py</strong>: The main script that ties together all the steps (data loading, preprocessing, model training, and evaluation).</li>
</ul>

<h3>Docker</h3>

<p>The project uses <strong>Docker</strong> to build a containerized image of the application. This ensures that the code works consistently on any machine. The <strong>Dockerfile</strong> defines the Docker image, which includes Python dependency installation, code cloning, and running the <strong>main.py</strong> file.</p>

<h4>Docker Commands:</h4>
<p><strong>Build the Docker image:</strong></p>
<pre><code>docker build -t myfirstpythonapp .</code></pre>

<p><strong>Run the Docker container:</strong></p>
<pre><code>docker run myfirstpythonapp</code></pre>

<h3>Points for Improvement:</h3>

<p>Although the project is complete, there are several areas that can be improved:</p>
<ul>
    <li>Adding a configuration file to make the application more flexible and configurable, allowing easy parameter adjustments for model predictions.</li>
    <li>Adding more unit tests for better code coverage.</li>
    <li>Dockerizing other parts of the code, such as preprocessing, training, evaluation, and the Flask app itself.</li>
    <li>Improving model performance and further optimizing the code.</li>
    <li>Better documentation to improve readability and maintainability of the project.</li>
</ul>

</body>
</html>
