import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

data = pd.read_csv(r"C:\Users\Aryan\Downloads\archive (1)\car data.csv")

print(data.head())

print(data.info())

print(data.isnull().sum())

data = data.dropna()

encoder = LabelEncoder()

data['Car_Name'] = encoder.fit_transform(data['Car_Name'])
data['Fuel_Type'] = encoder.fit_transform(data['Fuel_Type'])
data['Selling_type'] = encoder.fit_transform(data['Selling_type'])
data['Transmission'] = encoder.fit_transform(data['Transmission'])

X = data.drop('Selling_Price', axis=1)

y = data['Selling_Price']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = LinearRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)

r2 = r2_score(y_test, y_pred)

print("Mean Absolute Error:", mae)

print("R2 Score:", r2)

plt.figure(figsize=(8,5))

plt.scatter(y_test, y_pred)

plt.xlabel("Actual Prices")

plt.ylabel("Predicted Prices")

plt.title("Actual Price vs Predicted Price")

plt.show()

sample = X.iloc[0:1]

prediction = model.predict(sample)

print("Predicted Car Price:", prediction[0])