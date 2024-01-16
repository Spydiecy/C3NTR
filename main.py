# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Step 1: Generate Synthetic Dataset
np.random.seed(42)

num_participants = 200
exercises = ['Malware Analysis', 'Network Security', 'Social Engineering', 'Incident Response']

data = []

for participant_id in range(1, num_participants + 1):
    for exercise in exercises:
        # Randomly generate exercise duration, success, and additional behavioral features
        duration_seconds = np.random.randint(60, 600)
        success = np.random.choice([True, False])
        knowledge_level = np.random.uniform(1, 5)
        confidence_level = np.random.uniform(1, 5)
        engagement_score = np.random.uniform(0, 1)

        data.append({
            'ParticipantID': participant_id,
            'Exercise': exercise,
            'DurationSeconds': duration_seconds,
            'Success': success,
            'KnowledgeLevel': knowledge_level,
            'ConfidenceLevel': confidence_level,
            'EngagementScore': engagement_score
        })

# Create a DataFrame and save it to a CSV file
df = pd.DataFrame(data)
df.to_csv('cybersecurity_data.csv', index=False)

# Step 2: Load and Preprocess the Dataset
data = pd.read_csv('cybersecurity_data.csv')

# Feature engineering and encoding categorical features
features = pd.get_dummies(data[['Exercise', 'DurationSeconds', 'KnowledgeLevel', 'ConfidenceLevel', 'EngagementScore']], columns=['Exercise'])

# Additional features
features['PriorSuccessRate'] = data.groupby('ParticipantID')['Success'].transform('mean')
features['ExerciseDifficulty'] = np.random.uniform(1, 5, len(data))

# Target variable (change to ExerciseCompletionTime)
target = data['DurationSeconds']

# Step 3: Train a Machine Learning Model
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Train a Random Forest regressor (since we are predicting a continuous variable)
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
predictions = model.predict(X_test)

# Evaluate the model (using Mean Squared Error in this case)
mse = mean_squared_error(y_test, predictions)
print(f'Mean Squared Error: {mse}')

# Step 4: Make Recommendations for Participants
# Apply the model to the entire dataset for prediction
data['PredictedDuration'] = model.predict(features)

# Aggregate predictions per participant
recommendation_data = data.groupby('ParticipantID')['PredictedDuration'].mean().reset_index()

# Display recommendations
print("Recommendations:")
print(recommendation_data[['ParticipantID', 'PredictedDuration']])
