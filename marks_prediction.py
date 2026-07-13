import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# ----------------------------
# Load Dataset
# ----------------------------
data = pd.read_csv("student_marks.csv")

print("\nStudent Dataset\n")
print(data)

# ----------------------------
# Input and Output
# ----------------------------
X = data[['Hours']]
y = data['Marks']

# ----------------------------
# Train-Test Split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ----------------------------
# Model Training
# ----------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# ----------------------------
# Model Evaluation
# ----------------------------
accuracy = model.score(X_test, y_test)
prediction_test = model.predict(X_test)
mae = mean_absolute_error(y_test, prediction_test)

print("\nModel Evaluation")
print("------------------------")
print(f"R² Score : {accuracy:.2f}")
print(f"Mean Absolute Error : {mae:.2f}")

# ----------------------------
# User Prediction
# ----------------------------
hours = float(input("\nEnter Study Hours : "))

prediction = model.predict(pd.DataFrame({"Hours":[hours]}))

print(f"\nPredicted Marks : {prediction[0]:.2f}")

# ----------------------------
# Visualization
# ----------------------------
plt.figure(figsize=(8,5))

plt.scatter(X, y, label="Actual Data")

plt.plot(X, model.predict(X), label="Regression Line")

plt.xlabel("Study Hours")
plt.ylabel("Marks")
plt.title("Student Marks Prediction using Linear Regression")

plt.legend()

plt.savefig("student_marks_prediction.png", dpi=300)

print("\nGraph saved as student_marks_prediction.png")