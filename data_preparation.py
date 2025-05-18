import numpy as np
import pandas as pd

def generate_dummy_fraud_data(num_samples=10000, fraud_ratio=0.05, random_state=42):
    np.random.seed(random_state)
    
    # Features
    transaction_amount = np.random.exponential(scale=100, size=num_samples)  # amounts skewed positive
    transaction_time = np.random.randint(0, 24, size=num_samples)  # hour of day (0-23)
    user_id = np.random.randint(1000, 2000, size=num_samples)  # dummy user IDs
    location = np.random.choice(['US', 'EU', 'ASIA', 'OTHER'], size=num_samples, p=[0.5, 0.2, 0.2, 0.1])
    
    # Fraud label: imbalanced with fraud_ratio positive class
    is_fraud = np.random.choice([0, 1], size=num_samples, p=[1-fraud_ratio, fraud_ratio])
    
    # Create DataFrame
    df = pd.DataFrame({
        'transaction_amount': transaction_amount,
        'transaction_time': transaction_time,
        'user_id': user_id,
        'location': location,
        'is_fraud': is_fraud
    })
    
    return df

# Generate and save
df = generate_dummy_fraud_data()
print(df.head())

# Save to CSV for future use
df.to_csv('dummy_fraud_data.csv', index=False)
