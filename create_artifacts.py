"""
Simple script to create model.pkl and preprocessor.pkl artifacts
"""

import pickle
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
import xgboost as xgb
from sklearn.model_selection import train_test_split

# Create artifacts directory
os.makedirs('artifacts', exist_ok=True)

print("Creating model and preprocessor artifacts...")

# Generate simple synthetic data
np.random.seed(42)
n_samples = 1000

# Generate diamond features
carat = np.random.uniform(0.2, 3.0, n_samples)
cut = np.random.choice(['Fair', 'Good', 'Very Good', 'Premium', 'Ideal'], n_samples)
color = np.random.choice(['J', 'I', 'H', 'G', 'F', 'E', 'D'], n_samples)
clarity = np.random.choice(['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF', 'FL'], n_samples)
depth = np.random.uniform(55, 70, n_samples)
table = np.random.uniform(50, 65, n_samples)
x = 6.5 * (carat ** (1/3)) + np.random.normal(0, 0.2, n_samples)
y = x + np.random.normal(0, 0.1, n_samples)
z = x * 0.6 + np.random.normal(0, 0.1, n_samples)

# Generate prices based on features
base_price = 3000 * (carat ** 1.8)
cut_mult = {'Fair': 0.9, 'Good': 0.95, 'Very Good': 1.0, 'Premium': 1.05, 'Ideal': 1.1}
price = base_price * np.array([cut_mult[c] for c in cut])
price = price + np.random.normal(0, 500, n_samples)
price = np.clip(price, 300, 15000)

# Create DataFrame
data = pd.DataFrame({
    'carat': carat, 'cut': cut, 'color': color, 'clarity': clarity,
    'depth': depth, 'table': table, 'x': x, 'y': y, 'z': z, 'price': price
})

print(f"Generated {len(data)} samples")
print(f"Price range: ${data['price'].min():.0f} - ${data['price'].max():.0f}")

# Prepare features and target
X = data.drop('price', axis=1)
y = data['price']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Creating preprocessor...")

class SimplePreprocessor:
    def __init__(self):
        self.cut_encoder = LabelEncoder()
        self.color_encoder = LabelEncoder()
        self.clarity_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.fitted = False
    
    def fit(self, X):
        # Encode categorical features
        self.cut_encoder.fit(X['cut'])
        self.color_encoder.fit(X['color'])
        self.clarity_encoder.fit(X['clarity'])
        
        # Transform for fitting scaler
        X_transformed = X.copy()
        X_transformed['cut'] = self.cut_encoder.transform(X['cut'])
        X_transformed['color'] = self.color_encoder.transform(X['color'])
        X_transformed['clarity'] = self.clarity_encoder.transform(X['clarity'])
        
        # Fit scaler
        self.scaler.fit(X_transformed)
        self.fitted = True
        return self
    
    def transform(self, X):
        if not self.fitted:
            raise ValueError("Must fit before transform")
        
        # Handle single prediction case
        if isinstance(X, dict):
            X = pd.DataFrame([X])
        
        X_transformed = X.copy()
        X_transformed['cut'] = self.cut_encoder.transform(X['cut'])
        X_transformed['color'] = self.color_encoder.transform(X['color'])
        X_transformed['clarity'] = self.clarity_encoder.transform(X['clarity'])
        
        return self.scaler.transform(X_transformed)

# Create and fit preprocessor
preprocessor = SimplePreprocessor()
X_train_processed = preprocessor.fit(X_train).transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print("Training XGBoost model...")

# Train model
model = xgb.XGBRegressor(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42
)

model.fit(X_train_processed, y_train)

# Evaluate
train_score = model.score(X_train_processed, y_train)
test_score = model.score(X_test_processed, y_test)

print(f"Training R² score: {train_score:.3f}")
print(f"Test R² score: {test_score:.3f}")

# Save artifacts
print("Saving artifacts...")

with open('artifacts/model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('artifacts/preprocessor.pkl', 'wb') as f:
    pickle.dump(preprocessor, f)

print("✓ model.pkl saved")
print("✓ preprocessor.pkl saved")

# Test loading
print("Testing artifacts...")
with open('artifacts/model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

with open('artifacts/preprocessor.pkl', 'rb') as f:
    loaded_preprocessor = pickle.load(f)

# Test prediction
test_sample = {
    'carat': 1.0, 'cut': 'Premium', 'color': 'G', 'clarity': 'VS1',
    'depth': 61.5, 'table': 57.0, 'x': 6.3, 'y': 6.4, 'z': 3.9
}

X_processed = loaded_preprocessor.transform(test_sample)
prediction = loaded_model.predict(X_processed)[0]

print(f"Test prediction: ${prediction:.2f}")
print("✓ Artifacts created and tested successfully!")