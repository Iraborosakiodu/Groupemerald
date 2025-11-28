import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

def train_house_model():
    # 1. Load Data
    try:
        df = pd.read_csv('House Price Prediction Dataset.csv')
    except FileNotFoundError:
        print("Error: 'House Price Prediction Dataset.csv' not found.")
        return

    # 2. Preprocessing
    # Drop ID as it's just a counter, not a feature
    if 'Id' in df.columns:
        df = df.drop(columns=['Id'])
    
    # Store encoders to handle text columns (Location, Condition, Garage)
    encoders = {}
    
    # Identify categorical columns
    cat_cols = ['Location', 'Condition', 'Garage']
    
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    # 3. Define Features (X) and Target (y)
    target_col = 'Price'
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # 4. Train Model (Regressor because Price is a number)
    print("Training House Price Predictor...")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    # 5. Save Model and Encoders
    data_to_save = {
        'model': model,
        'encoders': encoders
    }
    
    with open('house_model.pkl', 'wb') as f:
        pickle.dump(data_to_save, f)

    print("Success! 'house_model.pkl' has been saved.")

if __name__ == "__main__":
    train_house_model()
