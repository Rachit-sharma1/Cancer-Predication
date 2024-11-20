
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from joblib import Parallel, delayed 
import joblib 
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Directory for output images
IMAGE_DIR = "output_images"
os.makedirs(IMAGE_DIR, exist_ok=True)

def load_and_preprocess_data_breast(file_path):
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

# Load data function for different datasets
def load_and_preprocess_data(file_path, cancer_type):
    # Load the dataset
    df = pd.read_csv(file_path)
    y = df['target']
    X = df.drop(columns=['target'])
    if cancer_type=='Cervical':
        df = df.replace('?', np.nan)
        df = df.drop(['STDs: Time since first diagnosis', 'STDs: Time since last diagnosis'], axis=1)
        df = df.apply(pd.to_numeric)
        df =  df.fillna(df.mean())
        y = df['Biopsy']
        X = df.drop(['Biopsy'], axis=1)

    # Define the target and features
    

    # # Detect categorical columns
    # categorical_columns = X.select_dtypes(include=['object']).columns

    # # Apply label encoding to categorical columns
    # label_encoders = {}
    # for col in categorical_columns:
    #     le = LabelEncoder()
    #     X[col] = le.fit_transform(X[col].astype(str))
    #     label_encoders[col] = le

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    if cancer_type=="breast":
        X_train, X_test, y_train, y_test = load_and_preprocess_data_breast(file_path)
    # print("Feature shape of X:",X_train.shape)
    return X_train, X_test, y_train, y_test
# Model training function
# def train_model(X_train, y_train, model_type='random_forest'):
#     if model_type == 'random_forest':
#         model = RandomForestClassifier(random_state=42)
#     elif model_type == 'logistic_regression':
#         model = LogisticRegression(random_state=42, max_iter=1000)
#     elif model_type == 'xgboost':
#         model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    
#     model.fit(X_train, y_train)
#     return model

# Evaluation function with confusion matrix and classification report
def load_model(Model_name,dataset):
    model = ''
    if Model_name =="RandomForest":
        if dataset =="Cervical":
            model = joblib.load('F:/Project/Machine Learning/Cancer Predication/Cervical-Cancer/rf_model.joblib')
        else :
            model = joblib.load('F:/Project/Machine Learning/Cancer Predication/Breast-Cancer/cancer detection project/Cmodel_rf.joblib')
    if Model_name =="logistic_regression":
        if dataset =="Cervical":
            model = joblib.load('F:/Project/Machine Learning/Cancer Predication/Cervical-Cancer/lr_model.joblib')
        else :
            model = joblib.load('F:/Project/Machine Learning/Cancer Predication/Breast-Cancer/cancer detection project/Cmodel_lr.joblib')
    
    if Model_name =="xgboost":
        if dataset =="Cervical":
            model = joblib.load('F:/Project/Machine Learning/Cancer Predication/Cervical-Cancer/model.joblib')
        else :
            model = joblib.load('F:/Project/Machine Learning/Cancer Predication/Breast-Cancer/cancer detection project/Cmodel_xgb.joblib')
    
    return model

def evaluate_model(model, X_test, y_test, title_prefix):
    y_pred = model.predict(X_test)
    conf_matrix = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f"{title_prefix} Confusion Matrix")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.savefig(f"{IMAGE_DIR}/{title_prefix}_confusion_matrix.png")
    plt.close()

    # Plot classification report
    report_df = pd.DataFrame(report).T.iloc[:-1, :-1]
    report_df.plot(kind='bar', figsize=(10, 6))
    plt.title(f"{title_prefix} Classification Report")
    plt.xlabel("Metrics")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.savefig(f"{IMAGE_DIR}/{title_prefix}_classification_report.png")
    plt.close()
    print(report)

# Main function
def main():
    cancer_type = input("Enter cancer type (Cervical/breast): ")
    model_type = input("Enter model type (RandomForest/logistic_regression/xgboost): ")
    file_path = ""
    if cancer_type=="Cervical":
        file_path = "F:/Project/Machine Learning/Cancer Predication/Cervical-Cancer/data/cervical_cancer.csv"
    else:
        file_path = "F:/Project/Machine Learning/Cancer Predication/Breast-Cancer/cancer detection project/data/metabric.csv"
    # dataset_name = input("Enter the dataset Name :").strip()
    # Load and preprocess data
    X_train, X_test, y_train, y_test = load_and_preprocess_data(file_path, cancer_type)

    # Train model
    model = load_model(model_type,cancer_type)

    # Evaluate model
    evaluate_model(model, X_test, y_test, title_prefix=cancer_type.capitalize())

if __name__ == "__main__":
    main()
