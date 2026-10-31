from pathlib import Path

target_file = Path("src/features_extra.py")
text = target_file.read_text(encoding="utf-8")

old_block = """    if len(waveform) == 0:
        raise ValueError("Cannot extract features from empty waveform.")

    # 1. MFCC Statistics (13 coefficients)
    try:
        mfcc = librosa.feature.mfcc(y=waveform, sr=sr, n_mfcc=13)"""

new_block = """    if len(waveform) == 0:
        raise ValueError("Cannot extract features from empty waveform.")

    if len(waveform) < 512:
        return ExtraAudioFeatures(
            track_id=track_id,
            mfcc_mean=[0.0] * 13,
            mfcc_std=[0.0] * 13,
            onset_strength_mean=0.0,
            tempo_bpm=120.0,
            rhythm_regularity=1.0,
            spectral_contrast_mean=[0.0] * 7,
            harmonic_ratio=0.0,
            percussive_ratio=0.0,
            spectral_flatness_mean=0.0,
            spectral_rolloff_hz=0.0,
            spectral_bandwidth_hz=0.0,
            extraction_version=EXTRA_FEATURE_VERSION,
        )

    n_fft = min(2048, len(waveform))

    # 1. MFCC Statistics (13 coefficients)
    try:
        mfcc = librosa.feature.mfcc(y=waveform, sr=sr, n_mfcc=13, n_fft=n_fft)"""

old_spectral = """    # 5. Additional Spectral Descriptors
    try:
        flatness = float(np.mean(librosa.feature.spectral_flatness(y=waveform)))
        rolloff = float(np.mean(librosa.feature.spectral_rolloff(y=waveform, sr=sr, roll_percent=0.85)))
        bandwidth = float(np.mean(librosa.feature.spectral_bandwidth(y=waveform, sr=sr)))"""

new_spectral = """    # 5. Additional Spectral Descriptors
    try:
        flatness = float(np.mean(librosa.feature.spectral_flatness(y=waveform, n_fft=n_fft)))
        rolloff = float(np.mean(librosa.feature.spectral_rolloff(y=waveform, sr=sr, roll_percent=0.85, n_fft=n_fft)))
        bandwidth = float(np.mean(librosa.feature.spectral_bandwidth(y=waveform, sr=sr, n_fft=n_fft)))"""

# Normalize newlines for search and replace
norm_text = text.replace("\r\n", "\n")
norm_old_block = old_block.replace("\r\n", "\n")
norm_old_spectral = old_spectral.replace("\r\n", "\n")

if norm_old_block in norm_text and norm_old_spectral in norm_text:
    norm_text = norm_text.replace(norm_old_block, new_block.replace("\r\n", "\n"), 1)
    norm_text = norm_text.replace(norm_old_spectral, new_spectral.replace("\r\n", "\n"), 1)
    target_file.write_text(norm_text.replace("\n", "\r\n"), encoding="utf-8")
    print("SUCCESS: features_extra.py updated cleanly.")
else:
    print("ERROR: blocks not found in features_extra.py")
