import pandas as pd
from sklearn.svm import SVR
import pickle
import os

X_train_scaled = pd.read_csv('./data/processed_data/X_train_scaled.csv')
y_train = pd.read_csv('./data/processed_data/y_train.csv').values.ravel()

with open('./models/best_params.pkl', 'rb') as f:
    best_params = pickle.load(f)

print("Les meilleurs paramètres qui ont étés chargés:", best_params)

model_svr = SVR(**best_params)
model_svr.fit(X_train_scaled, y_train)

os.makedirs('./models', exist_ok=True)

nom_du_model = './models/gbr_model.pkl'
with open(nom_du_model, 'wb') as f:
    pickle.dump(model_svr, f)

print(f"Le modèle a été entraîné et se trouve dans {nom_du_model}")
