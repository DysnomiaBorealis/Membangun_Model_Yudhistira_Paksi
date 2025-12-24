import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    roc_auc_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import dagshub

def setup_dagshub():
    username = os.getenv('DAGSHUB_USERNAME', 'DysnomiaBorealis')
    token = os.getenv('DAGSHUB_TOKEN')
    
    if not token:
        raise ValueError(
        )
    
    dagshub.init(
        repo_owner=username,
        repo_name='Membangun_Model_Yudhistira_Paksi',
        mlflow=True
    )
    
    # Set MLflow tracking URI ke DagsHub
    tracking_uri = f'https://dagshub.com/{username}/Membangun_Model_Yudhistira_Paksi.mlflow'
    mlflow.set_tracking_uri(tracking_uri)
    
    # Set authentication
    os.environ['MLFLOW_TRACKING_USERNAME'] = username
    os.environ['MLFLOW_TRACKING_PASSWORD'] = token
    
    print(f"DagsHub initialized: {tracking_uri}")
    return tracking_uri

def load_preprocessed_data():
    """
    Memuat dataset yang sudah dipreprocessing
    
    Returns:
        tuple: (X_tfidf, y)
    """
    # Load preprocessed data
    df = pd.read_csv('indo_spam_preprocessing.csv')
    
    # Load the vectorizer
    import joblib
    vectorizer = joblib.load('vectorizer.joblib')
    
    # Ambil cleaned text dan transform menggunakan vectorizer
    X_text = df['cleaned_text']
    X_tfidf = vectorizer.transform(X_text)
    
    # Ambil labels
    y = df['label']
    
    return X_tfidf, y

def plot_confusion_matrix(y_true, y_pred, save_path='confusion_matrix.png'):
    """
    Membuat dan menyimpan heatmap confusion matrix
    
    Args:
        y_true: Label sebenarnya
        y_pred: Label prediksi
        save_path (str): Path untuk menyimpan plot
    
    Returns:
        str: Path file yang disimpan
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Ham', 'Spam'], 
                yticklabels=['Ham', 'Spam'])
    plt.title('Confusion Matrix - Spam Detection', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontweight='bold')
    plt.xlabel('Predicted Label', fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    
    return save_path

def plot_roc_curve(y_true, y_proba, save_path='roc_curve.png'):
    """
    Membuat dan menyimpan ROC curve
    
    Args:
        y_true: Label sebenarnya
        y_proba: Probabilitas prediksi
        save_path (str): Path untuk menyimpan plot
    
    Returns:
        tuple: (Path file yang disimpan, ROC AUC score)
    """
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontweight='bold')
    plt.ylabel('True Positive Rate', fontweight='bold')
    plt.title('ROC Curve - Spam Detection', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    
    return save_path, roc_auc

def save_classification_report(y_true, y_pred, save_path='classification_report.txt'):
    """
    Menyimpan classification report detail
    
    Args:
        y_true: Label sebenarnya
        y_pred: Label prediksi
        save_path (str): Path untuk menyimpan report
    
    Returns:
        str: Path file yang disimpan
    """
    report = classification_report(y_true, y_pred, 
                                   target_names=['Ham', 'Spam'],
                                   digits=4)
    
    with open(save_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("SPAM DETECTION - CLASSIFICATION REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write(report)
        f.write("\n" + "=" * 60 + "\n")
    
    return save_path

def plot_hyperparameter_comparison(cv_results, save_path='hyperparameter_comparison.png'):
    """
    Plot hasil hyperparameter tuning
    
    Args:
        cv_results: Hasil dari GridSearchCV
        save_path (str): Path untuk menyimpan plot
    
    Returns:
        str: Path file yang disimpan
    """
    alphas = cv_results['param_alpha'].data
    mean_scores = cv_results['mean_test_score']
    std_scores = cv_results['std_test_score']
    
    plt.figure(figsize=(10, 6))
    plt.errorbar(range(len(alphas)), mean_scores, yerr=std_scores, 
                 marker='o', linestyle='-', linewidth=2, markersize=8,
                 capsize=5, capthick=2)
    plt.xlabel('Alpha Values', fontweight='bold')
    plt.ylabel('Mean CV Score (Accuracy)', fontweight='bold')
    plt.title('Hyperparameter Tuning Results', fontsize=14, fontweight='bold')
    plt.xticks(range(len(alphas)), [f'{a:.4f}' for a in alphas], rotation=45)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    
    return save_path

def train_tuned_model():
    """
    Melatih model dengan hyperparameter tuning dan manual MLflow logging
    """
    
    print("=" * 60)
    print("PELATIHAN MODEL LANJUTAN DENGAN INTEGRASI DAGSHUB")
    print("=" * 60)
    
    # Setup DagsHub
    print("\n[1/8] Setup integrasi DagsHub...")
    try:
        tracking_uri = setup_dagshub()
    except Exception as e:
        print(f"\nError setup DagsHub: {e}")
        print("\nSilakan set DAGSHUB_TOKEN environment variable dan coba lagi.")
        return
    
    # Load data
    print("\n[2/8] Memuat data yang sudah dipreprocessing...")
    X, y = load_preprocessed_data()
    print(f"      Dataset loaded! {X.shape[0]} samples dengan {X.shape[1]} features")
    
    # Split data
    print("\n[3/8] Memisahkan data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"      Training: {X_train.shape[0]}, Test: {X_test.shape[0]}")
    
    # Set experiment
    print("\n[4/8] Setup MLflow experiment...")
    mlflow.set_experiment("Spam_Detection_Advanced")
    print("      Experiment: Spam_Detection_Advanced")
    
    # Hyperparameter tuning
    print("\n[5/8] Melakukan hyperparameter tuning...")
    param_grid = {
        'alpha': [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0]
    }
    
    grid_search = GridSearchCV(
        MultinomialNB(),
        param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    
    print(f"\n      Best hyperparameters: {grid_search.best_params_}")
    print(f"      Best CV score: {grid_search.best_score_:.4f}")
    
    # Start MLflow run dengan MANUAL LOGGING
    print("\n[6/8] Melatih final model dan logging ke MLflow...")
    with mlflow.start_run(run_name="Tuned_NB_Model_Manual_Logging"):
        
        # Log hyperparameters (MANUAL)
        mlflow.log_param("model_type", "MultinomialNB")
        mlflow.log_param("best_alpha", grid_search.best_params_['alpha'])
        mlflow.log_param("cv_folds", 5)
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("random_state", 42)
        mlflow.log_param("n_features", X_train.shape[1])
        mlflow.log_param("n_train_samples", X_train.shape[0])
        mlflow.log_param("n_test_samples", X_test.shape[0])
        
        # Prediksi
        y_pred = best_model.predict(X_test)
        y_proba = best_model.predict_proba(X_test)[:, 1]
        
        # Hitung metrik
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Log standard metrics (MANUAL)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("best_cv_score", grid_search.best_score_)
        
        # Log ADDITIONAL metrics (di luar autolog)
        roc_auc = roc_auc_score(y_test, y_proba)
        mlflow.log_metric("roc_auc", roc_auc)
        
        # Hitung dan log per-class metrics
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        mlflow.log_metric("true_negatives", int(tn))
        mlflow.log_metric("false_positives", int(fp))
        mlflow.log_metric("false_negatives", int(fn))
        mlflow.log_metric("true_positives", int(tp))
        mlflow.log_metric("specificity", tn / (tn + fp))
        
        print(f"\n      Performa Model:")
        print(f"      Accuracy:  {accuracy:.4f}")
        print(f"      Precision: {precision:.4f}")
        print(f"      Recall:    {recall:.4f}")
        print(f"      F1-Score:  {f1:.4f}")
        print(f"      ROC-AUC:   {roc_auc:.4f}")
        
        # Buat artifacts directory
        os.makedirs('artifacts', exist_ok=True)
        
        # ARTIFACT 1: Confusion Matrix (CUSTOM)
        print("\n[7/8] Membuat custom artifacts...")
        cm_path = plot_confusion_matrix(y_test, y_pred, 'artifacts/confusion_matrix.png')
        mlflow.log_artifact(cm_path)
        print("      Logged confusion matrix")
        
        # ARTIFACT 2: ROC Curve (CUSTOM)
        roc_path, _ = plot_roc_curve(y_test, y_proba, 'artifacts/roc_curve.png')
        mlflow.log_artifact(roc_path)
        print("      Logged ROC curve")
        
        # ARTIFACT 3: Classification Report (CUSTOM)
        report_path = save_classification_report(y_test, y_pred, 
                                                 'artifacts/classification_report.txt')
        mlflow.log_artifact(report_path)
        print("      Logged classification report")
        
        # ARTIFACT 4: Hyperparameter Comparison Plot (CUSTOM - BONUS)
        hp_path = plot_hyperparameter_comparison(grid_search.cv_results_, 
                                                 'artifacts/hyperparameter_comparison.png')
        mlflow.log_artifact(hp_path)
        print("      Logged hyperparameter comparison")
        
        # Log trained model
        mlflow.sklearn.log_model(best_model, "model")
        print("      Logged trained model")
        
        # Simpan model lokal
        joblib.dump(best_model, 'artifacts/best_model.joblib')
        mlflow.log_artifact('artifacts/best_model.joblib')
        print("      Saved model to artifacts/")
        
        run_id = mlflow.active_run().info.run_id
        print(f"\n      MLflow Run ID: {run_id}")
    
    print("\n[8/8] Pelatihan selesai!")
    print("\n" + "=" * 60)
    print("PELATIHAN MODEL LANJUTAN SELESAI!")
    print("=" * 60)
    print("\nHasil di-log ke DagsHub:")
    print(f"  {tracking_uri.replace('.mlflow', '')}")
    print("\nArtifacts lokal disimpan di: ./artifacts/")
    print("=" * 60)

if __name__ == "__main__":
    train_tuned_model()
