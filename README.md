# Membangun Model - MLflow with DagsHub Integration

Project untuk pelatihan model machine learning spam detection menggunakan MLflow dan DagsHub integration.

**Author:** Yudhistira Paksi

## Deskripsi Project

Repository ini berisi implementasi pelatihan model Multinomial Naive Bayes untuk deteksi spam pada teks berbahasa Indonesia. Model dilatih menggunakan MLflow untuk tracking eksperimen dan DagsHub untuk penyimpanan online.

## Struktur Folder

```
Membangun_Model/
├── modelling.py                     # Basic model training dengan MLflow autolog
├── modelling_tuning.py              # Advanced model dengan hyperparameter tuning
├── indo_spam_preprocessing.csv      # Dataset yang sudah dipreprocessing
├── vectorizer.joblib                # TF-IDF vectorizer yang sudah dilatih
├── requirements.txt                 # Dependencies Python
├── DagsHub.txt                      # Link ke repository DagsHub
├── artifacts/                       # Custom artifacts (confusion matrix, ROC curve, dll)
├── mlruns/                          # MLflow tracking files (local)
└── models/                          # Trained model files
```

## Prerequisites

- Python 3.12
- DagsHub account
- Dependencies yang terlist di requirements.txt

## Instalasi

1. Clone repository:
```bash
git clone https://github.com/DysnomiaBorealis/Membangun_Model_Yudhistira_Paksi.git
cd Membangun_Model_Yudhistira_Paksi
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set environment variables untuk DagsHub (untuk modelling_tuning.py):
```bash
# Windows PowerShell
$env:DAGSHUB_TOKEN = "your_dagshub_token_here"

# Windows CMD
set DAGSHUB_TOKEN=your_dagshub_token_here

# Linux/Mac
export DAGSHUB_TOKEN="your_dagshub_token_here"
```

## Cara Menggunakan

### Basic Model Training (Local MLflow)

Script ini menggunakan MLflow autolog untuk tracking lokal:

```bash
python modelling.py
```

**Output:**
- Model Multinomial Naive Bayes terlatih
- Metrics: accuracy, precision, recall, F1-score
- MLflow tracking di folder `mlruns/`

**Melihat hasil:**
```bash
mlflow ui
```
Kemudian buka: http://localhost:5000

### Advanced Model Training (DagsHub Integration)

Script ini menggunakan manual logging dengan hyperparameter tuning:

```bash
python modelling_tuning.py
```

**Features:**
- Hyperparameter tuning dengan GridSearchCV
- Manual MLflow logging (semua parameters dan metrics)
- Custom artifacts:
  - Confusion Matrix
  - ROC Curve
  - Classification Report
  - Hyperparameter Comparison Plot
- Online tracking ke DagsHub

**Output:**
- Model terbaik dari GridSearchCV
- Semua artifacts disimpan lokal dan di DagsHub
- Metrics lengkap termasuk ROC-AUC, specificity

## Model Details

### Algorithm
- **Model:** Multinomial Naive Bayes
- **Task:** Text Classification (Spam Detection)
- **Language:** Indonesian

### Features
- **Feature Extraction:** TF-IDF Vectorization
- **Max Features:** 3000
- **Min DF:** 2
- **Max DF:** 0.8

### Performance Metrics
Model dievaluasi menggunakan:
- Accuracy
- Precision (weighted)
- Recall (weighted)
- F1-Score (weighted)
- ROC-AUC
- Specificity

### Hyperparameter Tuning
GridSearchCV dengan:
- **Parameter:** alpha (Multinomial NB)
- **Values:** [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0]
- **CV Folds:** 5
- **Scoring:** Accuracy

## DagsHub Integration

Project ini terintegrasi dengan DagsHub untuk:
- Online experiment tracking
- Collaborative ML development
- Version control untuk models dan data

**DagsHub Repository:** https://dagshub.com/DysnomiaBorealis/Membangun_Model_Yudhistira_Paksi

## Artifacts

### Confusion Matrix
Visualisasi performa model dalam bentuk heatmap:
- True Positives / Negatives
- False Positives / Negatives

### ROC Curve
Kurva ROC dengan AUC score untuk evaluasi model performance.

### Classification Report
Detailed metrics per-class:
- Precision
- Recall
- F1-Score
- Support

### Hyperparameter Comparison
Plot comparison hasil GridSearchCV untuk semua nilai alpha yang ditest.

## Dataset

**Source:** Indonesian Spam Detection Dataset

**Preprocessing steps:**
1. Handle missing values
2. Remove duplicates
3. Text cleaning (lowercase, remove URLs, emails, phone numbers)
4. Categorical encoding (Spam=1, Ham=0)
5. Train-test split (80-20)
6. TF-IDF vectorization

**Final Dataset:**
- Total samples: ~1,898
- Features: TF-IDF vectors (3000 dimensions)
- Labels: Binary (0=Ham, 1=Spam)

## Dependencies

```
pandas==2.0.3
scikit-learn==1.7.2
mlflow==2.9.2
dagshub==0.3.17
matplotlib==3.7.2
seaborn==0.12.2
joblib==1.3.2
setuptools (for MLflow compatibility)
```

## Kriteria Penilaian yang Dipenuhi

**Advance (4 pts):**
- Melatih model ML menggunakan MLflow Tracking UI yang disimpan online dengan DagsHub
- Menggunakan manual logging (bukan autolog) di modelling_tuning.py
- Log metrics yang tercover autolog PLUS minimal 2 artifacts tambahan:
  - Standard metrics: accuracy, precision, recall, F1
  - Additional metrics: ROC-AUC, specificity, confusion matrix components
  - Custom artifacts: confusion matrix, ROC curve, classification report, hyperparameter comparison

## Troubleshooting

### Error: "DAGSHUB_TOKEN environment variable not set!"
**Solusi:** Set environment variable DAGSHUB_TOKEN sebelum menjalankan modelling_tuning.py

### Error: "ModuleNotFoundError"
**Solusi:** Install semua dependencies dengan `pip install -r requirements.txt`

### Error: "File not found: indo_spam_preprocessing.csv"
**Solusi:** Pastikan file dataset ada di folder yang sama dengan script

### MLflow UI tidak menampilkan experiments
**Solusi:** 
1. Pastikan sudah menjalankan modelling.py atau modelling_tuning.py
2. Jalankan `mlflow ui` di direktori yang sama dengan folder mlruns/
3. Buka http://localhost:5000

## Notes

- `modelling.py` menggunakan local MLflow tracking untuk testing cepat
- `modelling_tuning.py` menggunakan DagsHub untuk tracking online dan collaboration
- Kedua script memiliki Indonesian comments untuk kemudahan maintenance
- Model dan vectorizer disimpan dalam format joblib untuk portability

## License

Educational Project - Dicoding Machine Learning Operations

## Contact

Yudhistira Paksi
- GitHub: @DysnomiaBorealis
- DagsHub: https://dagshub.com/DysnomiaBorealis
