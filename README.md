# C3NTR
AI based recommendation system using behaviour analysis for Cyber Exercises. This project implements an AI-based recommendation system using behavior analysis for Cyber Exercises. The system generates synthetic data, trains a machine learning model, and makes recommendations for participants based on their predicted exercise duration.

# Cybersecurity Data Analysis

This project uses machine learning to analyze cybersecurity data and make recommendations for participants.

## Overview

The project generates a synthetic dataset of cybersecurity exercises, trains a Random Forest regressor on the data, and makes predictions on the test set. It evaluates the model using Mean Squared Error, Root Mean Squared Error, Mean Absolute Error, and R-squared Score. The model is then applied to the entire dataset for prediction, and additional statistics are calculated. The predictions are aggregated per participant and displayed in the GUI.

## Dependencies

- Python
- Tkinter
- Pandas
- NumPy
- Scikit-learn

## Usage

- To run the script, simply go to https://replit.com/.
  
- Create a new file with the python extension.
  
- Copy paste the code from this github repo's `main.py` file to replit python file.
  
- Afterwards click on run.

- All the libraries will start installing alongside generating the synthetic dataset we generate in the `cybersecurity_data.csv` file.

- Run the script and click the "Load Data" button in the GUI to load your data. The script will train the model, make predictions, evaluate the model, and display the recommendations in the GUI.

- Voila the predicted the behaviour analysis will be printed successfully
  
## Code Structure

The code is structured as follows:

- Import necessary libraries
- Generate synthetic dataset
- Define a function to load and preprocess the dataset
- Train a Random Forest regressor
- Make predictions on the test set
- Evaluate the model
- Apply the model to the entire dataset for prediction
- Calculate additional statistics
- Aggregate predictions per participant
- Display recommendations in the GUI
- Create the main window
- Create a label
- Create a button for loading the dataset
- Create a text area for displaying the predictions
- Run the GUI
  
## Features

- **Synthetic Data Generation**: The system generates a synthetic dataset for a set number of participants and a set of cyber exercises. For each participant and exercise, it randomly generates values for exercise duration, success, knowledge level, confidence level, and engagement score.

- **Data Preprocessing**: The system loads the generated dataset and preprocesses it. It performs feature engineering and encodes categorical features. It also creates additional features like `PriorSuccessRate` and `ExerciseDifficulty`.

- **Machine Learning Model**: The system trains a Random Forest regressor on the training set. This model is used to predict the exercise duration based on the features.

- **Model Evaluation**: The system evaluates the performance of the trained model on the test set using Mean Squared Error (MSE) as the metric.

- **Recommendations**: The system applies the trained model to the entire dataset to predict the exercise duration. It then aggregates these predictions per participant to generate recommendations.

## License

This project is licensed under the terms of the MIT license.



