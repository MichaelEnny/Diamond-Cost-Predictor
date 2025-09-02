"""
Create artifacts compatible with container numpy version (1.24.4)
"""

import pickle
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.model_selection import train_test_split

print(f"Using numpy version: {np.__version__}")

# Create artifacts directory
os.makedirs('artifacts', exist_ok=True)

print("Creating container-compatible model and preprocessor artifacts...")

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

# Encode categorical features manually to match expected format
cut_mapping = {'Fair': 0, 'Good': 1, 'Very Good': 2, 'Premium': 3, 'Ideal': 4}
color_mapping = {'J': 0, 'I': 1, 'H': 2, 'G': 3, 'F': 4, 'E': 5, 'D': 6}
clarity_mapping = {'I1': 0, 'SI2': 1, 'SI1': 2, 'VS2': 3, 'VS1': 4, 'VVS2': 5, 'VVS1': 6, 'IF': 7, 'FL': 8}

# Transform data
data_transformed = data.copy()
data_transformed['cut'] = data['cut'].map(cut_mapping)
data_transformed['color'] = data['color'].map(color_mapping)
data_transformed['clarity'] = data['clarity'].map(clarity_mapping)

# Prepare features and target
feature_columns = ['carat', 'cut', 'color', 'clarity', 'depth', 'table', 'x', 'y', 'z']
X = data_transformed[feature_columns]
y = data_transformed['price']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and fit preprocessor (StandardScaler)
print("Creating StandardScaler preprocessor...")
preprocessor = StandardScaler()
X_train_scaled = preprocessor.fit_transform(X_train)
X_test_scaled = preprocessor.transform(X_test)

print("Training XGBoost model...")

# Train model
model = xgb.XGBRegressor(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42
)

model.fit(X_train_scaled, y_train)

# Evaluate
train_score = model.score(X_train_scaled, y_train)
test_score = model.score(X_test_scaled, y_test)

print(f"Training R² score: {train_score:.3f}")
print(f"Test R² score: {test_score:.3f}")

# Save artifacts
print("Saving artifacts...")

with open('artifacts/model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('artifacts/preprocessor.pkl', 'wb') as f:
    pickle.dump(preprocessor, f)

print("model.pkl saved")
print("preprocessor.pkl saved")

# Create mapping files for reference
mappings = {
    'cut_mapping': cut_mapping,
    'color_mapping': color_mapping,
    'clarity_mapping': clarity_mapping
}

with open('artifacts/feature_mappings.pkl', 'wb') as f:
    pickle.dump(mappings, f)

# Test loading
print("Testing artifacts...")
with open('artifacts/model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

with open('artifacts/preprocessor.pkl', 'rb') as f:
    loaded_preprocessor = pickle.load(f)

print("Artifacts loaded successfully!")

# Test with sample data (need to encode manually since we're using basic StandardScaler)
test_data = [[1.0, 3, 4, 3, 61.5, 57.0, 6.3, 6.4, 3.9]]  # Already encoded
X_processed = loaded_preprocessor.transform(test_data)
prediction = loaded_model.predict(X_processed)[0]

print(f"Test prediction: ${prediction:.2f}")
print("Container-compatible artifacts created successfully!")