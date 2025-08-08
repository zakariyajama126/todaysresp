EPL Match Result Predictor
A machine learning model that predicts English Premier League match outcomes using XGBoost classification.
🎯 Performance

46% Accuracy (vs 33% random chance)
Beats random guessing by 13%
Comparable to casual betting tipsters

How It Works
The model uses 4 key features:

Team Form: Rolling 5-game win streaks for home/away teams
Goal Averages: Historical scoring patterns
Time Weighting: Recent games matter more (95% decay rate)
Walk-Forward Validation: No future data leakage

Features

Individual match predictions with confidence scores
Gameweek-by-gameweek analysis
Breakdown by result type (Home Win/Draw/Away Win)
Shows most confident correct/incorrect predictions
Visual indicators (✅❌) for each prediction

Quick Start

Install requirements:

bashpip install pandas numpy xgboost

Run the predictor:

bashpython app.py
📁 Files

app.py - Main prediction script
E1.csv - Premier League match data (2022-23 season)
E0.csv - Additional season data
README.md - This file

Sample Output
GAMEWEEK 15
--------------------------------------------------
✅ Arsenal vs Chelsea
    Score: 2-1 | Actual: Home Win
    Predicted: Home Win (67.3% confidence)

❌ Liverpool vs Man City  
    Score: 1-1 | Actual: Draw
    Predicted: Home Win (52.8% confidence)
🎲 Model Details

Algorithm: XGBoost Multi-class Classification
Classes: Home Win (0), Draw (1), Away Win (2)
Training: Time-weighted samples with exponential decay
Validation: Chronological train/test splits by gameweek
Features: 4 engineered features per match
