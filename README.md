IPL Win Probability Predictor 🏏
A machine learning project that predicts the real-time win probability of a team during the second innings of an Indian Premier League (IPL) cricket match. The model updates the probability after every ball, providing a dynamic forecast of the match outcome.

Author: Divyansh Mathur

📋 Table of Contents
[Project Overview]

[Features]

[Dataset]

[Methodology]

[Technologies Used]

[Setup & Installation]

[How to Use the Model]

[Model Performance]

📝 Project Overview
The goal of this project is to build a robust machine learning model that can predict the win probability for the chasing team in an IPL match. The prediction is based on the state of the game at any given point, including the current score, wickets remaining, balls left, and required run rate. The final output is a trained LogisticRegression model saved as pipe.pkl, which can be integrated into other applications for real-time predictions.

✨ Features
Data Cleaning & Preprocessing: Merges and cleans two separate IPL datasets (matches.csv and deliveries.csv).

Feature Engineering: Creates crucial in-game features like runs_left, balls_left, wickets_left, current_run_rate (CRR), and required_run_rate (RRR).

Machine Learning Pipeline: Uses scikit-learn's Pipeline to streamline the process of one-hot encoding categorical features and training the model.

Model Training: Implements a Logistic Regression model, which is well-suited for binary classification tasks and provides probabilistic outputs.

Model Persistence: The final trained pipeline is saved as pipe.pkl for easy deployment and reuse.

📊 Dataset
This project uses two datasets from Kaggle's IPL Complete Dataset (2008-2020):

matches.csv: Contains match-level information, including teams, city, toss winner, and the final winner.

deliveries.csv: Provides ball-by-ball data for each match, including batsman, bowler, runs scored, and dismissals.

🛠️ Methodology
The model was built using the ipl_data_train.ipynb notebook, following these key steps:

Data Loading and Merging: The matches.csv and deliveries.csv files are loaded. The total runs for each innings are calculated and merged to create a comprehensive DataFrame.

Filtering for Second Innings: The data is filtered to include only the second innings of each match, as the model predicts the outcome for the chasing team.

Feature Engineering:

current_score: Calculated by taking the cumulative sum of runs scored.

runs_left: Calculated as total_runs_x - current_score.

balls_left: Calculated as 120 - (overs * 6 + balls).

wickets_left: Calculated as 10 - player_dismissed.

crr (Current Run Rate): current_score * 6 / (120 - balls_left).

rrr (Required Run Rate): runs_left * 6 / balls_left.

Final DataFrame Creation: A final DataFrame is created with the most relevant features: batting_team, bowling_team, city, runs_left, balls_left, wickets_left, total_runs_x, crr, rrr, and the target variable result (1 for win, 0 for loss).

Model Training:

The data is split into training and testing sets.

A ColumnTransformer is used to apply one-hot encoding to categorical features (batting_team, bowling_team, city).

A Logistic Regression model is trained on the transformed data.

The entire preprocessing and modeling pipeline is saved to pipe.pkl.

💻 Technologies Used
Python 3.10+

NumPy: For numerical operations.

Pandas: For data manipulation and analysis.

Scikit-learn: For building the machine learning pipeline and model.

Jupyter Notebook: For interactive development and training.

🚀 Setup & Installation
To run this project locally, follow these instructions:

Clone the repository:

git clone https://github.com/divyanshmathur004/IPL-Win-Probability-Predictor.git
cd IPL-Win-Probability-Predictor

Create and activate a virtual environment (recommended):

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

Install the required dependencies:
(First, create a requirements.txt file with numpy, pandas, and scikit-learn listed inside.)

pip install -r requirements.txt

Run the Jupyter Notebook:
To see how the model was trained, launch Jupyter Notebook and open ipl_data_train.ipynb.

jupyter notebook

🧠 How to Use the Model
The saved pipe.pkl file can be loaded to make predictions on new data. The input should be a Pandas DataFrame containing the required features.

Here is a sample Python script to make a prediction:

import pickle
import pandas as pd

# Load the trained model
with open('pipe.pkl', 'rb') as f:
    pipe = pickle.load(f)

# Create a sample input DataFrame (one row for one match state)
# Example: MI vs CSK in Mumbai, 80 runs left, 30 balls left, 7 wickets left, target 190
# CRR = (190-80)*6 / (120-30) = 7.33
# RRR = 80*6 / 30 = 16.0
input_data = pd.DataFrame({
    'batting_team': ['Mumbai Indians'],
    'bowling_team': ['Chennai Super Kings'],
    'city': ['Mumbai'],
    'runs_left': [80],
    'balls_left': [30],
    'wickets_left': [7],
    'total_runs_x': [190],
    'crr': [7.33],
    'rrr': [16.0]
})

# Make prediction
result = pipe.predict_proba(input_data)
win_prob = result[0][1]  # Probability of winning
loss_prob = result[0][0] # Probability of losing

print(f"Win Probability for {input_data['batting_team'].values[0]}: {round(win_prob*100, 2)}%")
print(f"Loss Probability for {input_data['batting_team'].values[0]}: {round(loss_prob*100, 2)}%")

🎯 Model Performance
The Logistic Regression model was chosen for its reliability and ability to output clear probabilities. After training, the model achieved an accuracy of 80.7% on the test set.
