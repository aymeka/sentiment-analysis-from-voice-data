import numpy as np
import librosa
import sounddevice as sd
import joblib
import logging
from sklearn.impute import SimpleImputer
from scipy import stats
from sklearn.feature_selection import SelectKBest, f_classif

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

model = joblib.load(r'C:\Users\karad\Desktop\duyguanalizi\emotion_model.pkl')
label_encoder = joblib.load(r'C:\Users\karad\Desktop\duyguanalizi\label_encoder.pkl')
scaler = joblib.load(r'C:\Users\karad\Desktop\duyguanalizi\scaler.pkl')
selector = joblib.load(r'C:\Users\karad\Desktop\duyguanalizi\feature_selector.pkl')

def record_audio(duration=5, fs=22050):
    try:
        print("Recording...")
        recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
        sd.wait()
        print("Recording finished.")
        return recording.flatten()
    except Exception as e:
        logging.error(f"Error during recording: {e}")
        return None

def extract_features_from_audio(audio, sr=22050):
    try:
        features = []

        # MFCC
        mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40).T, axis=0)
        features.extend(mfccs)

        # Spectral Centroid
        spectral_centroids = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr).T, axis=0)
        features.extend(spectral_centroids)

        # Spectral Contrast
        spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=audio, sr=sr).T, axis=0)
        features.extend(spectral_contrast)

        # Chroma Features
        chroma = np.mean(librosa.feature.chroma_stft(y=audio, sr=sr).T, axis=0)
        features.extend(chroma)

        # Zero Crossing Rate
        zcr = np.mean(librosa.feature.zero_crossing_rate(y=audio).T, axis=0)
        features.extend(zcr)

        # Spectral Rolloff
        rolloff = np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sr).T, axis=0)
        features.extend(rolloff)

        # Spectral Flux
        onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
        spectral_flux = np.mean(onset_env)
        features.append(spectral_flux)

        # Harmonic-to-Noise Ratio (HNR)
        harmonic, percussive = librosa.effects.hpss(audio)
        hnr = np.mean(librosa.effects.harmonic(audio) / (np.abs(percussive) + 1e-6))
        features.append(hnr)

        # Energy
        energy = np.mean(librosa.feature.rms(y=audio).T, axis=0)
        features.extend(energy)

        # Root Mean Square Energy (RMS Energy)
        rms = np.mean(librosa.feature.rms(y=audio).T, axis=0)
        features.extend(rms)

        # Pitch
        pitches, magnitudes = librosa.core.piptrack(y=audio, sr=sr)
        pitch = np.mean(pitches[pitches > 0]) if np.any(pitches > 0) else 0
        features.append(pitch)

        # Ekstra Özellikler
        features.append(np.max(features))  # Max
        features.append(np.min(features))  # Min
        features.append(np.mean(features))  # Mean
        features.append(np.ptp(features))  # Peak-to-Peak (range)

        # Regresyon
        x = np.arange(len(features))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, features)
        features.append(slope)
        features.append(intercept)

        # Momentler
        features.append(np.var(features))  # Variance
        features.append(np.std(features))  # Standard deviation
        features.append(stats.skew(features))  # Skewness
        features.append(stats.kurtosis(features))  # Kurtosis

        # Yüzdelikler
        features.append(np.percentile(features, 25))  # 25th percentile
        features.append(np.percentile(features, 50))  # Median (50th percentile)
        features.append(np.percentile(features, 75))  # 75th percentile

        return np.array(features)
    except Exception as e:
        logging.error(f"Error during feature extraction: {e}")
        return None

def segment_audio(audio, segment_length=0.02, sr=22050):
    segment_samples = int(segment_length * sr)
    num_segments = len(audio) // segment_samples
    segments = [audio[i*segment_samples:(i+1)*segment_samples] for i in range(num_segments)]
    return segments

def extract_features_from_segments(segments, sr=22050):
    all_features = []
    for segment in segments:
        features = extract_features_from_audio(segment, sr=sr)
        if features is not None:
            all_features.append(features)
    return np.array(all_features)

def predict_emotions_from_segments(features_segments):
    predictions = []
    try:
        imputer = SimpleImputer(strategy='mean')
        for features in features_segments:
            features_imputed = imputer.fit_transform(features.reshape(1, -1))
            features_normalized = scaler.transform(features_imputed)
            features_selected = selector.transform(features_normalized)
            predicted_label = model.predict(features_selected)
            predicted_emotion = label_encoder.inverse_transform(predicted_label)
            predictions.append(predicted_emotion[0])
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
    return predictions

def main():
    duration = 5
    audio = record_audio(duration=duration)
    if audio is not None:
        segments = segment_audio(audio, segment_length=0.02)
        features_segments = extract_features_from_segments(segments)
        if features_segments.size > 0:
            predicted_emotions = predict_emotions_from_segments(features_segments)
            if predicted_emotions:
                most_common_emotion = max(set(predicted_emotions), key=predicted_emotions.count)
                print(f"Predicted Emotion: {most_common_emotion}")
            else:
                print("Error predicting emotion.")
        else:
            print("Error extracting features from segments.")
    else:
        print("Error recording audio.")

if __name__ == "__main__":
    main()
