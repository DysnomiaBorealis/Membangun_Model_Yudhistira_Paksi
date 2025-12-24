import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import os

def load_preprocessed_data():
    """
    Memuat dataset yang sudah dipreprocessing dan vectorizer
    
    Returns:
        tuple: (X_tfidf, y, vectorizer)
    """
    # Load preprocessed data
    df = pd.read_csv('indo_spam_preprocessing.csv')
    
    # Load the vectorizer
    vectorizer = joblib.load('vectorizer.joblib')
    
    # Ambil cleaned text dan transform menggunakan vectorizer
    X_text = df['cleaned_text']
    X_tfidf = vectorizer.transform(X_text)
    
    # Ambil labels
    y = df['label']
    
    return X_tfidf, y, vectorizer

def train_basic_model():
    """
    Melatih model Multinomial Naive Bayes dasar dengan MLflow autolog
    """
    
    print("=" * 60)
    print("PELATIHAN MODEL DASAR DENGAN MLFLOW AUTOLOG")
    print("=" * 60)
    
    # Load data
    print("\n[1/5] Memuat data yang sudah dipreprocessing...")
    X, y, vectorizer = load_preprocessed_data()
    print(f"      Dataset loaded! Shape: {X.shape[0]} samples dengan {X.shape[1]} features")
    
    # Split data
    print("\n[2/5] Memisahkan data menjadi train dan test set...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"      Training set: {X_train.shape[0]} samples")
    print(f"      Test set: {X_test.shape[0]} samples")
    
    # Set MLflow tracking ke lokal
    print("\n[3/5] Setup MLflow...")
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("Spam_Detection_Basic")
    print("      MLflow tracking URI: file:./mlruns")
    print("      Experiment: Spam_Detection_Basic")
    
    # Enable autolog
    print("\n[4/5] Mengaktifkan MLflow autolog...")
    mlflow.sklearn.autolog()
    print("      Autolog enabled untuk scikit-learn")
    
    # Latih model
    print("\n[5/5] Melatih model Multinomial Naive Bayes...")
    with mlflow.start_run(run_name="Basic_NB_Model"):
        # Train the model
        model = MultinomialNB()
        model.fit(X_train, y_train)
        
        # Prediksi
        y_pred = model.predict(X_test)
        
        # Hitung metrik (autolog akan capture secara otomatis)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        print(f"\n      Performa Model:")
        print(f"      Accuracy:  {accuracy:.4f}")
        print(f"      Precision: {precision:.4f}")
        print(f"      Recall:    {recall:.4f}")
        print(f"      F1-Score:  {f1:.4f}")
        
        # Dapatkan run ID
        run = mlflow.active_run()
        print(f"\n      MLflow Run ID: {run.info.run_id}")
    
    print("\n" + "=" * 60)
    print("PELATIHAN MODEL DASAR SELESAI!")
    print("=" * 60)
    print("\nUntuk melihat hasil:")
    print("  1. Jalankan: mlflow ui")
    print("  2. Buka: http://localhost:5000")
    print("=" * 60)

if __name__ == "__main__":
    train_basic_model()
