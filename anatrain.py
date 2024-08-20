import os
import numpy as np
import librosa
import joblib
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score, RepeatedStratifiedKFold
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.over_sampling import SMOTE
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import logging
import warnings

warnings.filterwarnings('ignore')

logging.basicConfig(filename='audio_processing.log', level=logging.INFO)

def get_label(file_name):
    return os.path.basename(file_name).split('_')[-1].split('.')[0]

def extract_features(file_name):
    try:
        y, sr = librosa.load(file_name, duration=8)
        features = []

        # MFCC
        mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
        features.extend(mfccs)

        # Spectral Centroid
        spectral_centroids = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr).T, axis=0)
        features.extend(spectral_centroids)

        # Spectral Contrast
        spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr).T, axis=0)
        features.extend(spectral_contrast)

        # Chroma Features
        chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
        features.extend(chroma)

        # Zero Crossing Rate
        zcr = np.mean(librosa.feature.zero_crossing_rate(y).T, axis=0)
        features.extend(zcr)

        # Spectral Rolloff
        rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr).T, axis=0)
        features.extend(rolloff)

        # Spectral Flux
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        spectral_flux = np.mean(onset_env)
        features.append(spectral_flux)

        # Harmonic-to-Noise Ratio (HNR)
        harmonic, percussive = librosa.effects.hpss(y)
        hnr = np.mean(librosa.effects.harmonic(y) / (np.abs(percussive) + 1e-6))
        features.append(hnr)

        # Energy
        energy = np.mean(librosa.feature.rms(y=y).T, axis=0)
        features.extend(energy)

        # Root Mean Square Energy (RMS Energy)
        rms = np.mean(librosa.feature.rms(y=y).T, axis=0)
        features.extend(rms)

        # Pitch
        pitches, magnitudes = librosa.core.piptrack(y=y, sr=sr)
        pitch = np.mean(pitches[pitches > 0]) if np.any(pitches > 0) else 0
        features.append(pitch)

        return np.array(features)
    except Exception as e:
        logging.error(f"Error processing {file_name}: {e}")
        return None

audio_files_path = r'C:\Users\karad\Desktop\duyguanalizi\sesler'
features = []
labels = []

for root, _, filenames in os.walk(audio_files_path):
    for file_name in tqdm(filenames, desc="Processing audio files"):
        if file_name.endswith('.wav'):
            full_path = os.path.join(root, file_name)
            label = get_label(full_path)
            features_extracted = extract_features(full_path)
            if features_extracted is not None:
                features.append(features_extracted)
                labels.append(label)

feature_lengths = [len(f) for f in features]
if len(set(feature_lengths)) > 1:
    raise ValueError(f"Inconsistent feature lengths detected: {set(feature_lengths)}")

label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

X = np.array(features)
y = np.array(labels_encoded)

if np.any(np.isnan(X)):
    logging.info("NaN değerler bulundu, bunları dolduruyoruz...")
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)

scaler = RobustScaler()
X_normalized = scaler.fit_transform(X)

selector = SelectKBest(f_classif, k=20)
X_selected = selector.fit_transform(X_normalized, y)

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_selected, y)

X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled)

logging.info("Starting RandomizedSearchCV...")
param_distributions = {
    'C': [0.01, 0.1, 1, 10, 100],
    'gamma': [0.001, 0.01, 0.1, 1, 10],
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid']
}
random_search = RandomizedSearchCV(SVC(probability=True), param_distributions, n_iter=100, cv=RepeatedStratifiedKFold(n_splits=5, n_repeats=3), verbose=2, n_jobs=-1)
random_search.fit(X_train, y_train)
logging.info("RandomizedSearchCV finished.")

logging.info(f"Best parameters: {random_search.best_params_}")
logging.info(f"Best cross-validation score: {random_search.best_score_}")

try:
    cv_scores = cross_val_score(random_search.best_estimator_, X_train, y_train, cv=RepeatedStratifiedKFold(n_splits=5, n_repeats=3))
    logging.info(f"Cross-validation scores: {cv_scores}")
    logging.info(f"Mean CV score: {np.mean(cv_scores)}")
    logging.info(f"Std CV score: {np.std(cv_scores)}")
except ValueError as e:
    logging.error(f"Error during cross-validation: {e}")

for class_label in np.unique(y_resampled):
    class_mask = y_resampled == class_label
    if np.sum(class_mask) > 1:
        try:
            class_cv_scores = cross_val_score(random_search.best_estimator_, X_resampled[class_mask], y_resampled[class_mask], cv=RepeatedStratifiedKFold(n_splits=5, n_repeats=3))
            logging.info(f"CV scores for {label_encoder.inverse_transform([class_label])[0]}: "
                         f"mean = {np.mean(class_cv_scores):.4f}, std = {np.std(class_cv_scores):.4f}")
        except ValueError as e:
            logging.error(f"Error during cross-validation for class {label_encoder.inverse_transform([class_label])[0]}: {e}")

y_pred = random_search.predict(X_test)
logging.info(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
logging.info(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

f1 = f1_score(y_test, y_pred, average='weighted')
logging.info(f"F1 Score (weighted): {f1:.4f}")

y_pred_proba = random_search.predict_proba(X_test)
roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
logging.info(f"ROC AUC Score (weighted): {roc_auc:.4f}")

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig(r'C:\Users\karad\Desktop\duyguanalizi\confusion_matrix.png')
plt.close()

joblib.dump(random_search, r'C:\Users\karad\Desktop\duyguanalizi\emotion_model.pkl')
joblib.dump(label_encoder, r'C:\Users\karad\Desktop\duyguanalizi\label_encoder.pkl')
joblib.dump(scaler, r'C:\Users\karad\Desktop\duyguanalizi\scaler.pkl')
joblib.dump(selector, r'C:\Users\karad\Desktop\duyguanalizi\feature_selector.pkl')

class_distribution = np.bincount(y_resampled)
class_names = label_encoder.classes_

plt.figure(figsize=(10,6))
plt.bar(class_names, class_distribution)
plt.title('Class Distribution')
plt.xlabel('Emotion')
plt.ylabel('Count')
plt.savefig(r'C:\Users\karad\Desktop\duyguanalizi\class_distribution.png')
plt.close()

tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X_resampled)

plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_resampled, cmap='viridis')
plt.colorbar(scatter)
plt.title('t-SNE visualization of the dataset')
plt.savefig(r'C:\Users\karad\Desktop\duyguanalizi\tsne_visualization.png')
plt.close()
