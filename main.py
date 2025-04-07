#this is a single python code with all the functions for denoising using dnn for major project
import numpy as np
import librosa
import soundfile as sf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

SAMPLE_RATE = 16000  
FRAME_LENGTH = 512  
HOP_LENGTH = 128    
N_FFT = 512        
N_MEL = 128        
EPOCHS = 50
BATCH_SIZE = 32
TEST_SIZE = 0.2
RANDOM_STATE = 42

#please give path
def extract_features(audio_path): 
    try:
        y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
        if len(y) == 0:
            return None
        mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=N_FFT,
                                                       hop_length=HOP_LENGTH, n_mels=N_MEL)
        mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
        return mel_spectrogram_db.T
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None

def load_and_preprocess_data(noisy_files, clean_files):
    noisy_features = [extract_features(f) for f in noisy_files]
    clean_features = [extract_features(f) for f in clean_files]

    noisy_features = [f for f in noisy_features if f is not None]
    clean_features = [f for f in clean_features if f is not None]

    min_len = min(len(noisy_features), len(clean_features))
    noisy_features = noisy_features[:min_len]
    clean_features = clean_features[:min_len]

    X = []
    y = []
    for noisy, clean in zip(noisy_features, clean_features):
        min_frames = min(noisy.shape[0], clean.shape[0])
        X.extend(noisy[:min_frames])
        y.extend(clean[:min_frames])

    X = np.array(X)
    y = np.array(y)

    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    scaler_y = StandardScaler()
    y_scaled = scaler_y.fit_transform(y)

    X_reshaped = X_scaled.reshape(-1, N_MEL)
    y_reshaped = y_scaled.reshape(-1, N_MEL)

    return X_reshaped, y_reshaped, scaler_X, scaler_y

def create_dnn_model(input_shape):
    model = keras.Sequential([
        keras.layers.Dense(256, activation='relu', input_shape=input_shape),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(input_shape[0])
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def train_model(model, X_train, y_train, X_val, y_val, epochs, batch_size):
    history = model.fit(X_train, y_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=(X_val, y_val),
                        callbacks=[keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)])
    return model, history

def denoise_audio(audio_path, model, scaler_X, scaler_y):
    features = extract_features(audio_path)
    if features is None:
        print("Could not extract features from the input audio.")
        return None

    features_scaled = scaler_X.transform(features)
    features_reshaped = features_scaled.reshape(-1, N_MEL)
    denoised_features_scaled = model.predict(features_reshaped)
    denoised_features = scaler_y.inverse_transform(denoised_features_scaled)

    denoised_mel_spectrogram_db = denoised_features.T
    denoised_mel_spectrogram = librosa.db_to_power(denoised_mel_spectrogram_db)
    denoised_audio = librosa.feature.inverse.mel_to_audio(denoised_mel_spectrogram,
                                                        sr=SAMPLE_RATE,
                                                        n_fft=N_FFT,
                                                        hop_length=HOP_LENGTH)
    return denoised_audio

if __name__ == "__main__":

    noisy_audio_files = ['noisy_speech1.wav', 'noisy_speech2.wav']
    clean_audio_files = ['clean_speech1.wav', 'clean_speech2.wav']

    X, y, scaler_X, scaler_y = load_and_preprocess_data(noisy_audio_files, clean_audio_files)

    if X.shape[0] == 0:
        print("No valid training data found. Please check your audio files.")
    else:
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

        input_shape = (X_train.shape[1],)
        model = create_dnn_model(input_shape)
        model.summary()

        trained_model, history = train_model(model, X_train, y_train, X_val, y_val, EPOCHS, BATCH_SIZE)

        model.save('denoising_model.h5')
        import joblib
        joblib.dump(scaler_X, 'scaler_X.pkl')
        joblib.dump(scaler_y, 'scaler_y.pkl')
        print("Trained model and scalers saved.")

        input_audio_path = 'noisy_input.wav'  #
        denoised_output = denoise_audio(input_audio_path, trained_model, scaler_X, scaler_y)

        if denoised_output is not None:
            output_audio_path = 'denoised_output.wav'
            sf.write(output_audio_path, denoised_output, SAMPLE_RATE)
            print(f"Denoised audio saved to {output_audio_path}")
