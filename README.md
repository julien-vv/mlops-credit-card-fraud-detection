<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>mlops-credit-card-fraud-detection</title>
</head>
<body>

<h1>mlops-credit-card-fraud-detection</h1>

<p>The <strong>Card Fraud Detection</strong> project aims to build a machine learning model capable of detecting credit card fraud. The goal is to deploy a supervised learning model that analyzes transaction data and identifies suspicious activity. This project incorporates various tools for continuous integration, deployment, and data versioning to ensure efficient management of the code and the model in production.</p>

<p>The project contains a <strong>test</strong> folder, which includes unit tests for the functions in the source files. These unit tests are automated through <strong>CI/CD</strong> using <strong>GitHub Actions</strong>. Data versioning is managed manually with <strong>DVC</strong>, and the model itself is versioned using <strong>MLflow</strong>.</p>

<h3>Folder Structure:</h3>
<p>The <strong>src</strong> folder contains the necessary Python files for training, evaluating, and deploying the model:</p>
<ul>
    <li><strong>app.py</strong>: Uses <strong>Flask</strong> to deploy an API hosting the model.</li>
    <li><strong>preprocess_data.py</strong>: Contains functions for preprocessing the data.</li>
    <li><strong>evaluate_model.py</strong>: Includes functions to evaluate the model on test data and generate metrics such as the confusion matrix, ROC AUC score, and AUPRC.</li>
    <li><strong>train_model.py</strong>: Contains functions to train the model with different algorithms (Logistic Regression, Random Forest, XGBoost).</li>
    <li><strong>main.py</strong>: The main script that ties together all the steps (data loading, preprocessing, model training, and evaluation).</li>
</ul>

<h3>Docker</h3>

<p>The project uses <strong>Docker</strong> to build a containerized image of the application. This ensures that the code works consistently on any machine, whether running Linux or Windows. The <strong>Dockerfile</strong> defines the Docker image, which includes Python dependency installation, code cloning, and execution of the <strong>main.py</strong> script. The Docker image is built and verified through the CI/CD pipeline, ensuring that the image creation process works smoothly in different environments.</p>

<h4>Docker Commands:</h4>
<p><strong>Build the Docker image:</strong></p>
<pre><code>docker build -t myfirstpythonapp .</code></pre>

<p><strong>Run the Docker container:</strong></p>
<pre><code>docker run myfirstpythonapp</code></pre>

<h3>Points for Improvement:</h3>

<p>Although the project is complete, there are several areas that can be improved:</p>
<ul>
    <li>Adding a configuration file to make the application more flexible and configurable, allowing easy adjustments to the model's parameters for fraud prediction.</li>
    <li>Expanding the unit tests to provide more comprehensive code coverage.</li>
    <li>Dockerizing other parts of the code, such as the preprocessing, training, evaluation, and Flask app components, to further containerize the pipeline.</li>
    <li>Improving the performance of the model and optimizing the code for better efficiency.</li>
    <li>Enhancing the documentation to improve readability and maintainability of the project, making it easier for other developers to contribute.</li>
</ul>

</body>
</html>
