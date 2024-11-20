# src/preprocess.py
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path, low_memory=False)
    df = df.dropna(subset=['cancer_type_detailed'])
    X = df.drop(columns=['cancer_type_detailed', 'patient_id'])
    y = df['cancer_type_detailed']
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    categorical_columns = X.select_dtypes(include=['object']).columns
    X = pd.get_dummies(X, columns=categorical_columns, drop_first=True)
    X = X.fillna(X.mean()) 
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test
