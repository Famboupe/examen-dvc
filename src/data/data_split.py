import pandas as pd
from sklearn.model_selection import train_test_split
import os

input_file = os.path.expanduser('./data/raw_data/raw.csv')
output_dir = os.path.expanduser('./data/processed_data')
os.makedirs(output_dir, exist_ok=True)

df = pd.read_csv(input_file, sep=',')

X = df.drop(['date','silica_concentrate'], axis=1)
y = df['silica_concentrate']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

X_train.to_csv(os.path.join(output_dir, 'X_train.csv'), index=False)
X_test.to_csv(os.path.join(output_dir, 'X_test.csv'), index=False)
y_train.to_csv(os.path.join(output_dir, 'y_train.csv'), index=False)
y_test.to_csv(os.path.join(output_dir, 'y_test.csv'), index=False)

print("Les datasets ont été splités et conservés dans", output_dir)
