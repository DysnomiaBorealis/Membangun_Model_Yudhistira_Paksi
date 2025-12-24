import os
os.environ['MLFLOW_TRACKING_INSECURE_TLS'] = 'true'
os.environ['PYTHONIOENCODING'] = 'utf-8'

import os
os.environ['MLFLOW_TRACKING_INSECURE_TLS'] = 'true'
os.environ['PYTHONIOENCODING'] = 'utf-8'

import pandas as pd
import numpy as np
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import dagshub

# Inisialisasi DagsHub
dagshub.init(repo_owner='DysnomiaBorealis', repo_name='Membangun_Model_Yudhistira_Paksi', mlflow=True)

# Muat data
print("Memuat data...")
df = pd.read_csv('indo_spam_preprocessing.csv')
vectorizer = joblib.load('vectorizer.joblib')

X = vectorizer.transform(df['cleaned_text'])
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Set eksperimen
mlflow.set_experiment("Spam_Detection_Hyperparameter_Tuning")

# Berbagai nilai alpha untuk diuji
alpha_values = [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]

print(f"\nMembuat {len(alpha_values)} MLflow runs...")
print("=" * 60)

for i, alpha in enumerate(alpha_values, 1):
    with mlflow.start_run(run_name=f"NB_alpha_{alpha}"):
        print(f"\n[{i}/{len(alpha_values)}] Melatih dengan alpha={alpha}...")
        
        # Log parameter
        mlflow.log_param("alpha", alpha)
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("random_state", 42)
        
        # Latih model
        model = MultinomialNB(alpha=alpha)
        model.fit(X_train, y_train)
        
        # Prediksi
        y_pred = model.predict(X_test)
        
        # Hitung metrik
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        
        print(f"      Accuracy: {accuracy:.4f}")
        print(f"      F1-Score: {f1:.4f}")

print("\n" + "=" * 60)
print(f"✓ Berhasil membuat {len(alpha_values)} MLflow runs!")
print("\nLihat runs:")
print("  Local MLflow UI: http://localhost:5000")
print("  DagsHub: https://dagshub.com/DysnomiaBorealis/Membangun_Model_Yudhistira_Paksi.mlflow/")
print("\nRefresh browser Anda untuk melihat semua runs!")
