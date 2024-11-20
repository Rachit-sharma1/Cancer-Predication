# main.py
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
from src.preprocess import load_and_preprocess_data
from src.train_model import train_random_forest, train_logistic_regression, train_xgboost
from src.evaluate_model import evaluate_model
from joblib import Parallel, delayed 
import joblib 


IMAGE_DIR = "output_images"
os.makedirs(IMAGE_DIR, exist_ok=True)

def plot_confusion_matrix(conf_matrix, title, filename):
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(title)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.savefig(filename)
    plt.close()

def plot_classification_report(report, title, filename):
    report_df = pd.DataFrame(report).T.iloc[:-1, :-1]  # Exclude 'accuracy' row and support column
    report_df.plot(kind='bar', figsize=(10, 6))
    plt.title(title)
    plt.xlabel("Metrics")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_pie_chart(y, filename):
    plt.figure(figsize=(8, 8))
    subtype_counts = y.value_counts()  # Get the count of each subtype
    plt.pie(subtype_counts, labels=subtype_counts.index, autopct='%1.1f%%', startangle=140)
    plt.title("Cancer Subtype Distribution")
    plt.savefig(filename)
    plt.close()

def print_evaluation_results(model_name, accuracy, conf_matrix, report):
    print(f"{model_name} Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print("Confusion Matrix:")
    print(conf_matrix)
    print("Classification Report:")
    report_df = pd.DataFrame(report).transpose()
    print(report_df.to_string())
    print("\n" + "="*50 + "\n")

def main():
    # Step 1: Load and preprocess data
    X_train, X_test, y_train, y_test = load_and_preprocess_data('C:/Users/HP/Desktop/PJT/NEW/cancer detection project/data/metabric.csv')
    print("Feature shape of X:",X_train.shape)
    # Generate and save pie chart for cancer subtype distribution
    plot_pie_chart(pd.concat([pd.Series(y_train), pd.Series(y_test)]), f"{IMAGE_DIR}/cancer_subtype_distribution.png")

    # Step 2: Train the models
    model_rf = train_random_forest(X_train, y_train)
    model_lr = train_logistic_regression(X_train, y_train)
    model_xgb = train_xgboost(X_train, y_train)  # Train XGBoost

    joblib.dump(model_rf, 'Cmodel_rf.joblib')
    joblib.dump(model_lr, 'Cmodel_lr.joblib')
    joblib.dump(model_xgb, 'Cmodel_xgb.joblib')

    # Step 3: Evaluate the models
    accuracy_rf, conf_matrix_rf, report_rf = evaluate_model(model_rf, X_test, y_test)
    accuracy_lr, conf_matrix_lr, report_lr = evaluate_model(model_lr, X_test, y_test)
    accuracy_xgb, conf_matrix_xgb, report_xgb = evaluate_model(model_xgb, X_test, y_test)  # Evaluate XGBoost

    # Print results to console in a structured format
    print_evaluation_results("Random Forest", accuracy_rf, conf_matrix_rf, report_rf)
    print_evaluation_results("Logistic Regression", accuracy_lr, conf_matrix_lr, report_lr)
    print_evaluation_results("XGBoost", accuracy_xgb, conf_matrix_xgb, report_xgb)  # Print XGBoost results

    # Step 4: Generate and save plots
    # Confusion Matrices
    plot_confusion_matrix(conf_matrix_rf, "Random Forest Confusion Matrix", f"{IMAGE_DIR}/conf_matrix_rf.png")
    plot_confusion_matrix(conf_matrix_lr, "Logistic Regression Confusion Matrix", f"{IMAGE_DIR}/conf_matrix_lr.png")
    plot_confusion_matrix(conf_matrix_xgb, "XGBoost Confusion Matrix", f"{IMAGE_DIR}/conf_matrix_xgb.png")  # XGBoost confusion matrix

    # Classification Report Bar Charts
    plot_classification_report(report_rf, "Random Forest Classification Report", f"{IMAGE_DIR}/classification_report_rf.png")
    plot_classification_report(report_lr, "Logistic Regression Classification Report", f"{IMAGE_DIR}/classification_report_lr.png")
    plot_classification_report(report_xgb, "XGBoost Classification Report", f"{IMAGE_DIR}/classification_report_xgb.png")  # XGBoost report



if __name__ == "__main__":
    main()
