import numpy as np
from numpy.typing import NDArray
import pandas as pd
import librosa

from .data import SR
from .data import load_audio
from .preprocessing import get_labels


def get_spectrogram(audio, ref=np.max, n_fft=2048):
    # TODO: automatically set hop_length
    S = librosa.stft(audio, n_fft=n_fft, hop_length=512)
    frequencies = librosa.fft_frequencies(sr=SR, n_fft=n_fft)
    times = librosa.frames_to_time(
        np.arange(S.shape[1]), sr=SR, n_fft=n_fft, hop_length=512
    )
    S_db = librosa.amplitude_to_db(np.abs(S), ref=ref)
    return S_db, frequencies, times


def get_features(audio: NDArray) -> pd.Series:
    """Collect all features for one audio"""
    # TODO: just an example for now, collect all the features here
    # mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    # features = mfcc.mean(axis=1)
    # S = librosa.stft(y)
    # S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)
    S = np.abs(librosa.stft(audio))
    S_db = librosa.amplitude_to_db(S)  # TODO: use ref

    return pd.Series({"spec_min": S_db.min(), "spec_max": S_db.max()})


def prepare_data(df, df_taxonomy, sample_idx):

    df = df.iloc[sample_idx]
    y_class, y_primary = get_labels(df, df_taxonomy)

    features = [get_features(load_audio(sample)) for _, sample in df.iterrows()]

    return {
        "X": pd.DataFrame(features, index=sample_idx),
        "y_primary": y_primary,
        "y_class": y_class,
    }
