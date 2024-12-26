import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
import pickle
import os

X_train_scaled = pd.read_csv('./data/processed_data/X_train_scaled.csv')
y_train = pd.read_csv('./data/processed_data/y_train.csv').values.ravel()

param_grid = {
    'kernel': ['rbf', 'linear'],
    'C': [0.1, 1, 10, 100],
    'epsilon': [0.01, 0.1, 0.5],
    'gamma': ['scale', 'auto', 0.1, 1]
}
svr = SVR()
grid_search = GridSearchCV(svr, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, verbose=2)

grid_search.fit(X_train_scaled, y_train)
best_params = grid_search.best_params_

os.makedirs('./models', exist_ok=True)

with open('./models/best_params.pkl', 'wb') as f:
    pickle.dump(best_params, f)

print("Meilleurs paramètres:", best_params)
print("Les meilleurs paramètres ont étés trouvés et sont dans ./models/best_params.pkl")
