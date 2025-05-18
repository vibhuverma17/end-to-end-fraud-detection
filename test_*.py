from generate_dummy_data import generate_dummy_fraud_data

def test_data_shape():
    df = generate_dummy_fraud_data(1000)
    assert df.shape[0] == 1000
    assert 'is_fraud' in df.columns

def test_fraud_ratio():
    df = generate_dummy_fraud_data(10000, fraud_ratio=0.1)
    fraud_ratio = df['is_fraud'].mean()
    assert 0.08 <= fraud_ratio <= 0.12
