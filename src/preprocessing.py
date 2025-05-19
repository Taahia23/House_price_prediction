import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

df = pd.read_csv('../data/house_price_regression_dataset.csv')  

X = df.drop('House_Price', axis=1)
y = df['House_Price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

joblib.dump(scaler, 'models/scaler.joblib')

joblib.dump(X_train_scaled, 'models/X_train.joblib')
joblib.dump(X_test_scaled, 'models/X_test.joblib')
joblib.dump(y_train, 'models/y_train.joblib')
joblib.dump(y_test, 'models/y_test.joblib')

