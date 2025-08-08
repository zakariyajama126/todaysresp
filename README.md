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

📊 Features

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

📈 Sample Output
🏆 GAMEWEEK 15
--------------------------------------------------
✅ Arsenal vs Chelsea
    Score: 2-1 | Actual: Home Win
    Predicted: Home Win (67.3% confidence)

❌ Liverpool vs Man City  
    Score: 1-1 | Actual: Draw
    Predicted: Home Win (52.8% confidence)

Gameweek 15 Accuracy: 70.0% (7/10)


SUMMARY STATISTICS


📊 Overall Accuracy: 46.2% (175/379)

📈 Accuracy by Result Type:
              Total  Correct  Accuracy%
Home Wins       156      89       57.1
Draws            89      25       28.1
Away Wins       134      61       45.5

🎯 Most Confident Correct Predictions:
   Chelsea vs Wolves (2-0) - 89.3%
   Man City vs Bournemouth (4-1) - 87.1% 
   Newcastle vs Leeds (2-1) - 84.7%

💡 Average Confidence: 58.4%

🎉 SEASON COMPLETE!
The model achieved 46.2% accuracy - ABOVE the 45% professional tipster benchmark!
👍 Solid performance! Competitive with professional tipsters!

Model Details

Algorithm: XGBoost Multi-class Classification
Classes: Home Win (0), Draw (1), Away Win (2)
Training: Time-weighted samples with exponential decay
Validation: Chronological train/test splits by gameweek
Features: 4 engineered features per match

