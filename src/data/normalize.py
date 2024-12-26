import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

output_dir = os.path.expanduser('./data/processed_data')

X_train = pd.read_csv(os.path.join(output_dir, 'X_train.csv'))
X_test = pd.read_csv(os.path.join(output_dir, 'X_test.csv'))

scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

X_train_scaled.to_csv(os.path.join(output_dir, 'X_train_scaled.csv'), index=False)
X_test_scaled.to_csv(os.path.join(output_dir, 'X_test_scaled.csv'), index=False)

print("Les datasets normalisés ont été créés et se trouvent dans", output_dir)
