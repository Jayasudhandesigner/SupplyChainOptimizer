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
    'delay': np.random.uniform(0, 5, n_batches)  # Random delays
}

df = pd.DataFrame(data)
df.to_csv('supply_chain_data.csv', index=False)
print("Synthetic dataset saved as 'supply_chain_data.csv'")
print(df.head())
