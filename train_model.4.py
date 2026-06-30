#Final Test Accuracy: 70.98%
import os
import pickle
import numpy as np
from scipy import stats
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# CONFIGURATION
DATASET_PATH = r"C:\Users\ayesh\PycharmProjects\wesad\WESAD"
SUBJECTS = [f'S{i}' for i in range(2, 18) if i != 12]

WINDOW_SIZE = 7000
STEP_SIZE = 3500


def extract_stats(window):
    """30-feature consistency for dashboard compatibility."""
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

    # ACCURACY BREAKTHROUGH: QUANTILE NORMALIZATION
    # This transforms each subject's data into a uniform distribution
    # before mapping it to a normal curve. It removes individual 'baselines' entirely.
    qt = QuantileTransformer(output_distribution='normal', n_quantiles=1000, random_state=42)

    normalized_signals = {}
    for k, v in signals.items():
        # Reshape for the transformer
        v_reshaped = v.reshape(-1, 1)
        normalized_signals[k] = qt.fit_transform(v_reshaped).flatten()

    X, y = [], []
    for i in range(0, len(labels) - WINDOW_SIZE, STEP_SIZE):
        window_labels = labels[i:i + WINDOW_SIZE]
        label = stats.mode(window_labels, keepdims=True)[0][0]

        if label in [1, 2, 3, 4]:
            features = []
            for sig in ['eda', 'temp', 'resp']:
                # Using the already transformed signals
                features.extend(extract_stats(normalized_signals[sig][i:i + WINDOW_SIZE]))
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

# 2. SPLIT (S6-S9)
test_subs = ['S6', 'S7', 'S8', 'S9']
train_idx = ~np.isin(groups, test_subs)
test_idx = np.isin(groups, test_subs)

# 3. TRAIN EXTRA TREES (High Capacity)
print("\n--- Training Extra Trees Classifier ---")
model = ExtraTreesClassifier(
    n_estimators=1200,  # More trees to smooth out prediction errors
    max_depth=40,  # High depth for high-dimensional feature separation
    min_samples_leaf=1,  # Maximize detail capture
    bootstrap=True,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

model.fit(X[train_idx], y[train_idx])

# 4. EVALUATION
y_pred = model.predict(X[test_idx])
print(f"\nFinal Test Accuracy: {accuracy_score(y[test_idx], y_pred):.2%}")

# 5. VISUALIZATION
cm = confusion_matrix(y[test_idx], y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges')
plt.show()

# 6. SAVE
joblib.dump(model, 'wesad_stress_model.pkl')
