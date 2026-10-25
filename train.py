import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

print("Loading dataset...")

# Load dataset
df = pd.read_csv("creditcard.csv")

print("Dataset Loaded Successfully!")
print(df.head())

# Features and Target
X = df.drop("Class", axis=1)
y = df["Class"]

# Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create Model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

print("Training model...")

model.fit(X_train, y_train)

print("Training Completed!")

# Prediction
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy)
import joblib

joblib.dump(model, "fraud_model.pkl")

print("Model Saved Successfully!")