import numpy as np
from numpy.typing import NDArray
import pandas as pd
import librosa

def get_features(audio: NDArray)-> pd.Series:
    S = np.abs(librosa.stft(audio))
    S_db = librosa.amplitude_to_db(S)
    return pd.Series({"spec_min": S_db.min(), "spec_max": S_db.max()})
    