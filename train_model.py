import pandas as pd # type: ignore
import numpy as np
import pickle
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.ensemble import RandomForestClassifier # type: ignore

# Sample dataset (Replace with Kaggle dataset)
data = {
    'travel_time': [30, 60, 90, 120, 45, 50, 70, 85, 95],
    'time_to_reach': [15, 30, 45, 60, 20, 25, 40, 50, 55],
    'label': [1, 0, 1, 0, 1, 1, 0, 1, 0]  # 1 = Recommended, 0 = Not Recommended
}

df = pd.DataFrame(data)

# Splitting data into train & test sets
X = df[['travel_time', 'time_to_reach']]
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("✅ Model saved as model.pkl")
