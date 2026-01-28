import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib

# Load dataset
df = pd.read_csv("C:/Users/K. Lakhan/OneDrive/Desktop/COE/LOGISTIC REGRESSION/heart.csv (1).xls")

# Split features and target
X = df.drop(columns=['target', 'age_group'], errors='ignore')
y = df['target']

# Split into train and test sets (optional)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Logistic Regression
model = LogisticRegression(max_iter=1000)  # Increase iterations if needed
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, "C:/Users/K. Lakhan/OneDrive/Desktop/COE/LOGISTIC REGRESSION/logistic_regression_model.pkl")

print("✅ Model trained and saved successfully!")
