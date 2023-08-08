#extract_spectrogram.py

import os
import warnings
import numpy as np
import joblib
import librosa
import argparse
warnings.filterwarnings("ignore", category=UserWarning)

SR = 22050
N_FFT = 2048
N_HOP = 512
N_MEL = 128

data_path = 'feature_folder'


def load_labels(label_list):
    labels = {}
    round_labels = {}
    for label_joblib in label_list:
        labels.update(joblib.load(label_joblib))
    for key, label in labels.items():
        round_labels[key] = []
        for row in label:
            round_labels[key] += [[round(float(row[0])), round(float(row[1]))]]
    return round_labels


def load_features(files):
    features = {}
    for feature_file in files:
        features.update(joblib.load(feature_file))
    return features


def extract_features(file):

    y, _ = librosa.load(file, sr=SR)
    data = librosa.feature.melspectrogram(y=y, sr=SR, n_fft=N_FFT, hop_length=N_HOP,
                                          power=1, n_mels=N_MEL, fmin=20, fmax=5000)
    data = np.reshape(data, (data.shape[0], data.shape[1], 1))
    return data


def folder_to_joblib(base_folder, feature_file, label_file):
    feature_set = {}
    label_set = {}
    keys = ""
    for (dirpath, _, filenames) in os.walk(base_folder):
        for file in [f for f in filenames if f.endswith('.mp3') or f.endswith('.wav')]:
            key = '{}'.format(file.replace('.mp3', ''))

            features = extract_features(os.path.join(dirpath, file))
            feature_set[key] = features
            label_set[key] = [[0,0]]

            print(key, np.shape(features))
            keys += key + '\n'

    joblib.dump(feature_set, feature_file)
    print(len(feature_set), 'features saved to', feature_file)
    joblib.dump(label_set, label_file)
    print(len(feature_set), 'features saved to', label_file)
    with open("keys.txt", "w") as f:
        f.write(keys.strip())
    print("Succesfully wrote key names to keys.txt")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    args = parser.parse_args()

    folder_to_joblib(args.path, 'feature_name.joblib', 'label_name.joblib')
