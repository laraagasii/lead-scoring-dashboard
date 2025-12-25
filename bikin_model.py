import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

def train_final_model():
    # 1. Load Data (Pastikan file CSV ada di folder yang sama)
    df = pd.read_csv('Lead Scoring.csv')
    df = df.replace('Select', np.nan)

    # 2. 9 Fitur terpilih yang paling berpengaruh
    selected_features = [
        'Total Time Spent on Website', 'Last Activity', 'What is your current occupation',
        'Lead Source', 'TotalVisits', 'Specialization', 'Lead Origin', 
        'Do Not Email', 'Page Views Per Visit'
    ]
    
    X = df[selected_features].copy()
    y = df['Converted']

    # 3. Handling Missing Values
    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = X[col].fillna('Not Specified')
        else:
            X[col] = X[col].fillna(X[col].median())

    # 4. Ambil opsi kategori untuk dropdown di Streamlit
    cat_options = {}
    for col in X.select_dtypes(include=['object']).columns:
        cat_options[col] = sorted([str(x) for x in X[col].unique()])

    # 5. Build Pipeline Preprocessing & Model
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), X.select_dtypes(exclude=['object']).columns.tolist()),
            ('cat', OneHotEncoder(handle_unknown='ignore'), X.select_dtypes(include=['object']).columns.tolist())
        ])

    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42))
    ])

    # 6. Training
    model.fit(X, y)

    # 7. Simpan Model (Perbaikan nama file dan ekstensi .joblib)
    joblib.dump({
        'model': model, 
        'features': selected_features, 
        'cat_options': cat_options
    }, 'best_lead_scoring_model.joblib')
    
    print("✅ Berhasil! File 'best_lead_scoring_model.joblib' telah disimpan.")

if __name__ == "__main__":
    train_final_model()