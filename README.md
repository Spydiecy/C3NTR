# C3NTR
AI based recommendation system using behaviour analysis for Cyber Exercises. This project implements an AI-based recommendation system using behavior analysis for Cyber Exercises. The system generates synthetic data, trains a machine learning model, and makes recommendations for participants based on their predicted exercise duration.

## Features

- **Synthetic Data Generation**: The system generates a synthetic dataset for a set number of participants and a set of cyber exercises. For each participant and exercise, it randomly generates values for exercise duration, success, knowledge level, confidence level, and engagement score.

- **Data Preprocessing**: The system loads the generated dataset and preprocesses it. It performs feature engineering and encodes categorical features. It also creates additional features like `PriorSuccessRate` and `ExerciseDifficulty`.

- **Machine Learning Model**: The system trains a Random Forest regressor on the training set. This model is used to predict the exercise duration based on the features.

- **Model Evaluation**: The system evaluates the performance of the trained model on the test set using Mean Squared Error (MSE) as the metric.

- **Recommendations**: The system applies the trained model to the entire dataset to predict the exercise duration. It then aggregates these predictions per participant to generate recommendations.

## Usage

- To run the script, simply go to https://replit.com/.
  
- Create a new file with the python extension.
  
- Copy paste the code from this github repo's `main.py` file to replit python file.
  
- Afterwards click on run.

- All the libraries will start installing alongside generating the synthetic dataset we generate in the `cybersecurity.csv` file.

- Voila the predicted the behaviour analysis will be printed successfully

## Note

This is a simplified example and the actual implementation of such a system could be much more complex, depending on the specific requirements and available data. Also, the synthetic dataset generated here is for demonstration purposes and may not reflect real-world data. In a real-world scenario, you would use actual data collected from cyber exercises.




