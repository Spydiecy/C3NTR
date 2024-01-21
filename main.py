# Import necessary libraries
import tkinter as tk
from tkinter import filedialog, Text, messagebox
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from math import sqrt

# Set pandas display options
pd.set_option('display.max_columns', None)


# Function to load and preprocess the dataset
def load_data():
  filename = filedialog.askopenfilename()
  data = pd.read_csv(filename)

  # Feature engineering and encoding categorical features
  features = pd.get_dummies(
      data[[
          'Exercise', 'DurationSeconds', 'KnowledgeLevel', 'ConfidenceLevel',
          'EngagementScore', 'PreviousExperience', 'LearningStyle',
          'OtherExercisePerformance', 'KnowledgeConfidenceInteraction',
          'EngagementOtherExerciseInteraction', 'TransformedKnowledgeLevel',
          'Noise'
      ]],
      columns=['Exercise', 'PreviousExperience', 'LearningStyle'])

  # Additional features
  features['PriorSuccessRate'] = data.groupby(
      'ParticipantID')['Success'].transform('mean')
  features['ExerciseDifficulty'] = np.random.uniform(1, 5, len(data))

  # Target variable (change to ExerciseCompletionTime)
  target = data['DurationSeconds']

  # Step 3: Train a Machine Learning Model
  # Split the data into training and testing sets
  X_train, X_test, y_train, y_test = train_test_split(features,
                                                      target,
                                                      test_size=0.2,
                                                      random_state=42)

  # Train a Random Forest regressor (since we are predicting a continuous variable)
  model = RandomForestRegressor(random_state=42)
  model.fit(X_train, y_train)

  # Make predictions on the test set
  predictions = model.predict(X_test)

  # Evaluate the model (using Mean Squared Error in this case)
  mse = mean_squared_error(y_test, predictions)
  rmse = sqrt(mse)
  mae = mean_absolute_error(y_test, predictions)
  r2 = r2_score(y_test, predictions)

  # Display the evaluation metrics
  text_area.insert(tk.END, f'Mean Squared Error: {mse}\n')
  text_area.insert(tk.END, f'Root Mean Squared Error: {rmse}\n')
  text_area.insert(tk.END, f'Mean Absolute Error: {mae}\n')
  text_area.insert(tk.END, f'R-squared Score: {r2}\n')

  # Step 4: Make Recommendations for Participants
  # Apply the model to the entire dataset for prediction
  data['PredictedDuration'] = model.predict(features)

  # Calculate additional statistics
  data['PredictedDurationStd'] = data.groupby(
      'ParticipantID')['PredictedDuration'].transform('std')
  data['PredictedDurationMax'] = data.groupby(
      'ParticipantID')['PredictedDuration'].transform('max')
  data['PredictedDurationMin'] = data.groupby(
      'ParticipantID')['PredictedDuration'].transform('min')

  # Aggregate predictions per participant
  recommendation_data = data.groupby('ParticipantID').agg({
      'PredictedDuration':
      'mean',
      'PredictedDurationMin':
      'mean',
      'PredictedDurationMax':
      'mean',
  }).reset_index()

  # Display recommendations in the GUI
  text_area.insert(tk.END, "Recommendations:\n")
  text_area.insert(tk.END, recommendation_data.to_string(index=False))


# Create the main window
root = tk.Tk()
root.title("C3NTR")
root.geometry("800x600")

# Create a label
label = tk.Label(root, text="Please load your data", font=("Arial", 14))
label.pack(pady=20)

# Create a button for loading the dataset
load_button = tk.Button(root,
                        text="Load Data",
                        command=load_data,
                        font=("Arial", 12))
load_button.pack()

# Create a text area for displaying the predictions
text_area = Text(root, width=90, height=20)
text_area.pack(pady=20)

# Run the GUI
root.mainloop()
