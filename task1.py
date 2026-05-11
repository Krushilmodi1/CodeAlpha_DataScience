import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

data = pd.read_csv(r"C:\Users\Aryan\Downloads\Iris.csv")

print(data.head())

X = data.drop(["Species", "Id"], axis=1)   
y = data["Species"]                      

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = KNeighborsClassifier(n_neighbors=3)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy:", accuracy)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

sample = pd.DataFrame([[5.1, 3.5, 1.4, 0.2]],
columns=X.columns)

prediction = model.predict(sample)

print("\nPredicted Flower Species:", prediction[0])