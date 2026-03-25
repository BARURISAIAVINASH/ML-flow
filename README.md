# MLflow Experiment Tracking for ANN-Based Wine Quality Prediction

This repository demonstrates a complete **MLflow-based machine learning workflow** for a simple **Artificial Neural Network (ANN)** regression task on the **Wine Quality** dataset.

The main focus of this project is not only training an ANN, but also showing how **MLflow** can be used for:

- experiment tracking
- parameter logging
- metric logging
- hyperparameter tuning
- model saving
- model loading
- model registration

This project combines **TensorFlow/Keras**, **Hyperopt**, and **MLflow** to build a reproducible and organized deep learning experimentation pipeline.

---

## Table of Contents

- [Project Objective](#project-objective)
- [Why MLflow is Used](#why-mlflow-is-used)
- [Project Workflow](#project-workflow)
- [Model Architecture](#model-architecture)
- [MLflow Workflow in This Project](#mlflow-workflow-in-this-project)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [How to Run](#how-to-run)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Tracking Experiments in MLflow](#tracking-experiments-in-mlflow)
- [Loading the Logged Model](#loading-the-logged-model)
- [Registering the Model](#registering-the-model)
- [Example Output](#example-output)
- [.gitignore](#gitignore)
- [Future Improvements](#future-improvements)
- [Author](#author)

---

## Project Objective

The objective of this project is to predict wine quality using an **ANN regression model** and manage the full experiment lifecycle using **MLflow**.

Instead of training a model once and printing results to the console, this project organizes the workflow so that each experiment is tracked properly. This makes it easy to answer questions such as:

- Which hyperparameters gave the best result?
- What was the validation RMSE for each run?
- Which model corresponds to the best run?
- Can the model be loaded later for inference?
- Can the best model be registered for future use?

---

## Why MLflow is Used

In machine learning projects, we often try multiple experiments with different settings. Without proper tracking, it becomes difficult to compare runs and reproduce results.

In this project, **MLflow** is used to solve that problem.

### MLflow is used for:

1. **Creating an experiment**
2. **Starting and managing runs**
3. **Logging hyperparameters**
4. **Logging evaluation metrics**
5. **Saving the trained ANN model**
6. **Loading the saved model later**
7. **Registering the final model**

Thus, MLflow acts as the **experiment management layer** of this project.

---

## Project Workflow

The complete pipeline followed in this repository is:

1. Load the wine quality dataset
2. Split data into train, validation, and test sets
3. Build an ANN model using TensorFlow/Keras
4. Train the model using SGD optimizer
5. Tune hyperparameters using Hyperopt
6. Track every trial using MLflow
7. Log the best hyperparameters and best RMSE
8. Save the best model using MLflow
9. Reload the model for prediction
10. Optionally register the model in MLflow

This creates a clean and reproducible workflow for experimentation.

---

## Model Architecture

The ANN used in this project is a simple feedforward neural network for regression.

### Architecture

- **Input layer**
- **Normalization layer**
- **Dense hidden layer** with ReLU activation
- **Final dense output layer** with one neuron

### Example model

```python
model = keras.Sequential([
    keras.Input([train_x.shape[1]]),
    keras.layers.Normalization(mean=mean, variance=var),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(1)
])
