# AI-Powered Supply Chain Optimizer

## Overview
This project leverages AI to optimize Pfizer’s drug supply chain by predicting production bottlenecks and suggesting resource allocation. Using **Python** and **Scikit-learn**, it analyzes synthetic supply chain data to forecast potential delays and improve efficiency in drug delivery.

## Features
- Predicts **production delays** using a **Random Forest Regressor**.
- Provides **optimization suggestions** to prevent supply chain bottlenecks.
- Uses **synthetic supply chain data** that mimics real-world operations.
- Visualizes **feature importance** and **predicted vs. actual delays**.

## Installation
Ensure you have **Python 3.8+** installed, then install the dependencies:

```bash
pip install pandas numpy scikit-learn matplotlib
```

## Dataset Creation
Generate a synthetic dataset before running the model:

```python
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(42)
n_batches = 1000
start_date = datetime(2025, 1, 1)

data = {
    'batch_id': range(1, n_batches + 1),
    'production_date': [start_date + timedelta(days=np.random.randint(0, 365)) for _ in range(n_batches)],
    'material_stock': np.random.uniform(0, 100, n_batches),
    'demand': np.random.randint(50, 201, n_batches),
    'delay': np.random.uniform(0, 5, n_batches)
}

df = pd.DataFrame(data)
df.to_csv('supply_chain_data.csv', index=False)
print("Synthetic dataset saved as 'supply_chain_data.csv'")
```

## Running the Model
Once the dataset is created, run the supply chain optimizer:

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv('supply_chain_data.csv')

data['days_to_year_end'] = (pd.to_datetime('2025-12-31') - pd.to_datetime(data['production_date'])).dt.days
X = data[['material_stock', 'demand', 'days_to_year_end']]
y = data['delay']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error: {mae:.2f} days")

# Optimization suggestion
threshold = 2  # Days
high_delay_idx = y_pred > threshold
suggestions = ["Increase material stock" if X_test[i, 0] < 0 else "Adjust schedule" for i in range(len(y_pred)) if high_delay_idx[i]]
print("Sample Optimization Suggestions:", suggestions[:5])

# Visualization
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([0, 5], [0, 5], 'r--')
plt.title("Predicted vs Actual Delays")
plt.xlabel("Actual Delay (days)")
plt.ylabel("Predicted Delay (days)")
plt.show()

# Feature Importance
importance = model.feature_importances_
features = X.columns
plt.bar(features, importance)
plt.title("Feature Importance in Supply Chain Delays")
plt.xlabel("Features")
plt.ylabel("Importance")
plt.xticks(rotation=45)
plt.show()

# Example prediction
new_batch = np.array([[50, 150, 100]])  # Material stock, demand, days to year-end
new_batch_scaled = scaler.transform(new_batch)
prediction = model.predict(new_batch_scaled)
print(f"Predicted Delay: {prediction[0]:.2f} days")
print("Suggestion:", "Increase material stock" if prediction[0] > threshold else "No action needed")
```

## Results & Insights
- **Mean Absolute Error (MAE):** Displays how accurate the predictions are.
- **Optimization Suggestions:** Identifies high-delay batches and suggests preventive actions.
- **Visualizations:** Helps understand model accuracy and feature importance.

## Future Improvements
- Implement **deep learning models** (LSTMs) for better forecasting.
- Integrate **real Pfizer supply chain data** for higher accuracy.
- Deploy as a **Flask API** for real-time use.

## License
This project is open-source under the MIT License.

---
### 🔗 Connect with Me
For suggestions or contributions, feel free to **open an issue** or **submit a pull request**!

🚀 **Happy Coding!**

