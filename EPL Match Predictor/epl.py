import pandas as pd
import numpy as np
from xgboost import XGBClassifier

# 1. Load data
df = pd.read_csv("E1.csv")
df = df[['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR']].dropna()
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
df = df.sort_values('Date').reset_index(drop=True)
df['Result'] = df['FTR'].map({'H': 0, 'D': 1, 'A': 2})
df['Gameweek'] = (df.index // 10) + 1


# 2. Feature engineering
def add_features(df):
    df = df.copy()
    df['HomeForm'] = (
        df.assign(HomeWin=(df['Result'] == 0).astype(int))
        .groupby('HomeTeam')['HomeWin']
        .transform(lambda x: x.shift().rolling(5).sum())
    )
    df['AwayForm'] = (
        df.assign(AwayWin=(df['Result'] == 2).astype(int))
        .groupby('AwayTeam')['AwayWin']
        .transform(lambda x: x.shift().rolling(5).sum())
    )
    df['HomeGoalsAvg'] = df.groupby('HomeTeam')['FTHG'].transform(lambda x: x.shift().expanding().mean())
    df['AwayGoalsAvg'] = df.groupby('AwayTeam')['FTAG'].transform(lambda x: x.shift().expanding().mean())
    return df.dropna()


df = add_features(df)

# 3. Train/test by gameweek with weighted training
predictions = []
features = ['HomeForm', 'AwayForm', 'HomeGoalsAvg', 'AwayGoalsAvg']


# Helper function to convert result codes back to readable format
def result_to_text(code):
    return {0: 'Home Win', 1: 'Draw', 2: 'Away Win'}[code]


print("=" * 80)
print("INDIVIDUAL MATCH PREDICTIONS")
print("=" * 80)

for gw in range(6, df['Gameweek'].max() + 1):
    train = df[df['Gameweek'] < gw]
    test = df[df['Gameweek'] == gw]
    if len(train['Result'].unique()) < 3: continue

    X_train, y_train = train[features], train['Result']
    X_test, y_test = test[features], test['Result']

    weights = np.power(0.95, df['Gameweek'].max() - train['Gameweek'].values)
    model = XGBClassifier(
        objective='multi:softmax',
        num_class=3,
        eval_metric='mlogloss',
        use_label_encoder=False,
        n_estimators=30,
        verbosity=0
    )
    model.fit(X_train, y_train, sample_weight=weights)
    y_pred = model.predict(X_test)

    # Get prediction probabilities for confidence
    y_prob = model.predict_proba(X_test)

    print(f"\n🏆 GAMEWEEK {gw}")
    print("-" * 50)

    gw_correct = 0
    for i in range(len(test)):
        row = test.iloc[i]
        actual = int(y_test.iloc[i])
        predicted = int(y_pred[i])
        confidence = y_prob[i][predicted] * 100

        # Status indicator
        status = "✅" if actual == predicted else "❌"

        print(f"{status} {row['HomeTeam']} vs {row['AwayTeam']}")
        print(f"    Score: {int(row['FTHG'])}-{int(row['FTAG'])} | Actual: {result_to_text(actual)}")
        print(f"    Predicted: {result_to_text(predicted)} ({confidence:.1f}% confidence)")
        print()

        if actual == predicted:
            gw_correct += 1

        predictions.append({
            'Gameweek': int(gw),
            'HomeTeam': row['HomeTeam'],
            'AwayTeam': row['AwayTeam'],
            'Score': f"{int(row['FTHG'])}-{int(row['FTAG'])}",
            'Actual': int(actual),
            'Predicted': int(predicted),
            'Correct': actual == predicted,
            'Confidence': confidence
        })

    gw_accuracy = gw_correct / len(test) * 100 if len(test) > 0 else 0
    print(f"Gameweek {gw} Accuracy: {gw_accuracy:.1f}% ({gw_correct}/{len(test)})")

# 4. Overall Results
print("\n" + "=" * 80)
print("SUMMARY STATISTICS")
print("=" * 80)

pred_df = pd.DataFrame(predictions)
correct = (pred_df['Actual'] == pred_df['Predicted']).sum()
total = len(pred_df)
overall_accuracy = correct / total * 100

print(f"\n📊 Overall Accuracy: {overall_accuracy:.1f}% ({correct}/{total})")

# Breakdown by result type
result_breakdown = pred_df.groupby('Actual').agg({
    'Correct': ['count', 'sum']
}).round(3)
result_breakdown.columns = ['Total', 'Correct']
result_breakdown['Accuracy%'] = (result_breakdown['Correct'] / result_breakdown['Total'] * 100).round(1)
result_breakdown.index = ['Home Wins', 'Draws', 'Away Wins']

print(f"\n📈 Accuracy by Result Type:")
print(result_breakdown)

# Best and worst predictions
print(f"\n🎯 Most Confident Correct Predictions:")
confident_correct = pred_df[pred_df['Correct'] == True].nlargest(3, 'Confidence')
for _, row in confident_correct.iterrows():
    print(f"   {row['HomeTeam']} vs {row['AwayTeam']} ({row['Score']}) - {row['Confidence']:.1f}%")

print(f"\n🤔 Most Confident Wrong Predictions:")
confident_wrong = pred_df[pred_df['Correct'] == False].nlargest(3, 'Confidence')
for _, row in confident_wrong.iterrows():
    result_names = {0: 'Home Win', 1: 'Draw', 2: 'Away Win'}
    print(
        f"   {row['HomeTeam']} vs {row['AwayTeam']} ({row['Score']}) - Predicted {result_names[row['Predicted']]} with {row['Confidence']:.1f}%")

print(f"\n💡 Average Confidence: {pred_df['Confidence'].mean():.1f}%")