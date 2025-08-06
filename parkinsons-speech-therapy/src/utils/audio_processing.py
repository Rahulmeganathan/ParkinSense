import librosa
import numpy as np
import parselmouth
from parselmouth.praat import call
import torch

class AudioProcessor:
    def __init__(self):
        self.sample_rate = 16000
        
    def extract_features(self, audio_path):
        """Extract acoustic features from audio file
        
        Args:
            audio_path (str): Path to audio file
            
        Returns:
            dict: Extracted features
            
        Raises:
            RuntimeError: If audio file cannot be loaded or processed
        """
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=self.sample_rate)
            if len(y) == 0:
                raise RuntimeError("Audio file is empty")
        except Exception as e:
            raise RuntimeError(f"Failed to load audio file: {str(e)}")
        
        # Create Praat Sound object
        sound = parselmouth.Sound(y, sr)
        
        # Extract features
        features = {}
        
        # Basic features
        features['duration'] = float(sound.duration)
        features['mean_pitch'] = float(sound.to_pitch().selected_array['frequency'].mean())
        features['pitch_std'] = float(sound.to_pitch().selected_array['frequency'].std())
        
        # Intensity features
        intensity = sound.to_intensity()
        features['mean_intensity'] = float(intensity.values.mean())
        features['intensity_std'] = float(intensity.values.std())
        
        # Jitter and Shimmer
        point_process = call(sound, "To PointProcess (periodic, cc)", 75, 600)
        
        features['jitter'] = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        features['shimmer'] = call([sound, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        
        # Harmonic to Noise Ratio
        harmonicity = call(sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
        features['harmonic_ratio'] = float(harmonicity.values.mean())
        
        # Spectral features
        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        features['spectral_centroid'] = float(spec_cent.mean())
        
        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        features['spectral_bandwidth'] = float(spec_bw.mean())
        
        # Mel-frequency features
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        features['mfccs'] = mfccs.mean(axis=1).tolist()
        
        # Time series features for visualization
        features['pitch_contour'] = sound.to_pitch().selected_array['frequency'].tolist()
        features['volume_envelope'] = librosa.feature.rms(y=y)[0].tolist()
        
        # Derived metrics
        features['clarity'] = self._calculate_clarity(features)
        features['stability'] = self._calculate_stability(features)
        features['volume'] = self._calculate_volume(features)
        
        return features
        
    def _calculate_clarity(self, features):
        """Calculate speech clarity score"""
        # Combine relevant features into clarity score
        harmonic_component = np.clip(features['harmonic_ratio'], 0, 20) / 20
        jitter_component = 1 - np.clip(features['jitter'], 0, 0.02) / 0.02
        shimmer_component = 1 - np.clip(features['shimmer'], 0, 0.1) / 0.1
        
        clarity = (harmonic_component + jitter_component + shimmer_component) / 3
        return float(clarity)
        
    def _calculate_stability(self, features):
        """Calculate voice stability score"""
        # Combine relevant features into stability score
        pitch_stability = 1 - np.clip(features['pitch_std'], 0, 50) / 50
        intensity_stability = 1 - np.clip(features['intensity_std'], 0, 10) / 10
        
        stability = (pitch_stability + intensity_stability) / 2
        return float(stability)
        
    def _calculate_volume(self, features):
        """Calculate volume adequacy score"""
        # Normalize intensity to 0-1 range
        volume = np.clip(features['mean_intensity'], 30, 90)
        volume = (volume - 30) / (90 - 30)
        
        return float(volume)