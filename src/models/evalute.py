import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import json
import os

X_test_scaled = pd.read_csv('./data/processed_data/X_test_scaled.csv')
y_test = pd.read_csv('./data/processed_data/y_test.csv').values.ravel()

with open('./models/gbr_model.pkl', 'rb') as f:
    model = pickle.load(f)

y_pred = model.predict(X_test_scaled)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

predictions_df = pd.DataFrame({'valeur_Actuel': y_test, 'valeur_Predicte': y_pred})

os.makedirs('./data/processed_data', exist_ok=True)
predictions_df.to_csv('./data/processed_data/prediction.csv', index=False)
os.makedirs('./metrics', exist_ok=True)
metrics = {
    'mse': mse,
    'r2': r2
}

with open('./metrics/scores.json', 'w') as f:
    json.dump(metrics, f)

print("Les Prédictions se trouvent dans ./data/processed_data/prediction.csv")
print("Les Métriques se trouvent dans ./metrics/scores.json")
print(f"MSE: {mse}")
print(f"R2 Score: {r2}")
