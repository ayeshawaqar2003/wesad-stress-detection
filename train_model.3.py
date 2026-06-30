Final Test Accuracy: 70.10%
import os
import pickle
import numpy as np
from scipy import stats
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from imblearn.combine import SMOTETomek
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

#CONFIGURATION
DATASET_PATH = r"C:\Users\ayesh\PycharmProjects\wesad\WESAD"
SUBJECTS = [f'S{i}' for i in range(2, 18) if i != 12]

WINDOW_SIZE = 7000 # 10 sec @ 700Hz
STEP_SIZE = 3500   # 5 sec overlap

def extract_stats(window):
    """Maintains 30-feature consistency: 10 stats * 3 signals."""
    return [
        np.mean(window), np.std(window), np.var(window),
        np.min(window), np.max(window), np.median(window),
        np.percentile(window, 25), np.percentile(window, 75),
        stats.skew(window), stats.kurtosis(window)
    ]

def get_key_case_insensitive(d, key):
    for k in d.keys():
        if k.lower() == key.lower(): return k
    raise KeyError(f"{key} not found")

def load_wesad_subject(subject_id):
    file_path = os.path.join(DATASET_PATH, subject_id, f'{subject_id}.pkl')
    with open(file_path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')

    chest = data['signal']['chest']
    labels = data['label']
    signals = {
        'eda': chest[get_key_case_insensitive(chest, 'eda')].flatten(),
        'temp': chest[get_key_case_insensitive(chest, 'temp')].flatten(),
        'resp': chest[get_key_case_insensitive(chest, 'resp')].flatten()
    }

    # Quantile Mapping: Forces subject data into a Normal distribution
    qt = QuantileTransformer(output_distribution='normal', n_quantiles=1000, random_state=42)
    norm_signals = {k: qt.fit_transform(v.reshape(-1, 1)).flatten() for k, v in signals.items()}

    X, y = [], []
    for i in range(0, len(labels) - WINDOW_SIZE, STEP_SIZE):
        window_labels = labels[i:i + WINDOW_SIZE]
        label = stats.mode(window_labels, keepdims=True)[0][0]

        if label in [1, 2, 3, 4]:
            features = []
            for sig in ['eda', 'temp', 'resp']:
                features.extend(extract_stats(norm_signals[sig][i:i + WINDOW_SIZE]))
            X.append(features)
            y.append(label)
    return np.array(X), np.array(y)

# 1. LOAD DATA
print(" Loading data with Quantile Mapping...")
X_all, y_all, groups = [], [], []
for s in SUBJECTS:
    try:
        X_s, y_s = load_wesad_subject(s)
        X_all.append(X_s)
        y_all.append(y_s)
        groups.extend([s] * len(y_s))
        print(f"Done: {s}")
    except Exception as e:
        print(f"Error {s}: {e}")

X = np.vstack(X_all)
y = np.concatenate(y_all)
groups = np.array(groups)

#  2. SPLIT (S6-S9)
test_subs = ['S6', 'S7', 'S8', 'S9']
train_idx = ~np.isin(groups, test_subs)
test_idx = np.isin(groups, test_subs)

X_train, y_train = X[train_idx], y[train_idx]
X_test, y_test = X[test_idx], y[test_idx]

#  3. RESAMPLING & BALANCING
print("⚖️ Applying SMOTE-Tomek to balance classes and remove noise...")
smt = SMOTETomek(sampling_strategy='auto', random_state=42)
X_train_bal, y_train_bal = smt.fit_resample(X_train, y_train)

# 4. TRAIN HIGH-CAPACITY MODEL
print("\n--- Training Final ExtraTrees Ensemble (2000 Trees) ---")
model = ExtraTreesClassifier(
    n_estimators=2000,        # Maximum voting stability
    max_depth=50,             # Deep enough for fine-grained separation
    min_samples_split=2,
    max_features=None,        # Use all 30 features for every decision
    bootstrap=True,
    class_weight='balanced_subsample',
    random_state=42,
    n_jobs=-1
)

model.fit(X_train_bal, y_train_bal)

# 5. EVALUATION
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\n✨ Final Test Accuracy: {acc:.2%}")

print("\nDetailed Report:")
print(classification_report(y_test, y_pred,
                            target_names=['Baseline', 'Stress', 'Amusement', 'Meditation']))

#6. VISUALIZATION
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='magma',
            xticklabels=['Baseline', 'Stress', 'Amusement', 'Meditation'],
            yticklabels=['Baseline', 'Stress', 'Amusement', 'Meditation'])
plt.title(f"Confusion Matrix - Final Accuracy: {acc:.2%}")
plt.show()

#7. SAVE
joblib.dump(model, 'wesad_stress_model.pkl')
print("\nModel saved. Dashboard is ready for update!")

