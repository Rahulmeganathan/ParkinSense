import os
import json
import numpy as np
if not hasattr(np, 'complex'):
    np.complex = complex
import librosa
import soundfile as sf
from pathlib import Path
from tqdm import tqdm
import argparse
import warnings
import opensmile
import tempfile
warnings.filterwarnings("ignore")


class ParkinsonFeatureExtractor:
    def __init__(self, sample_rate=16000, clip_duration=10.0, hop_length=160, n_fft=512):
        self.sample_rate = sample_rate
        self.clip_duration = clip_duration
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.clip_samples = int(sample_rate * clip_duration)

        self.output_dir = Path("processed_dataset")
        self.audio_dir = self.output_dir / "audio_clips"
        self.features_dir = self.output_dir / "acoustic_features"
        self.output_dir.mkdir(exist_ok=True)
        for label in ["PD", "Control"]:
            (self.audio_dir / label).mkdir(parents=True, exist_ok=True)
            (self.features_dir / label).mkdir(parents=True, exist_ok=True)

        # Initialize OpenSMILE
        self.smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.ComParE_2016,
            feature_level=opensmile.FeatureLevel.Functionals
        )

    def find_audio_files(self, base_folders):
        files = []
        for folder in base_folders:
            for label in ["PD", "HC"]:
                full = Path(folder) / label
                if not full.exists():
                    continue
                for f in full.glob("*.wav"):
                    files.append({
                        "file_path": f,
                        "label": "PD" if label == "PD" else "Control",
                        "source_folder": folder
                    })
        return files

    def extract_features(self, y, sr):
        f0 = librosa.yin(y, fmin=80, fmax=450, sr=sr)
        pitch_stats = {
            "pitch_mean": float(np.mean(f0)),
            "pitch_std": float(np.std(f0)),
            "pitch_min": float(np.min(f0)),
            "pitch_max": float(np.max(f0)),
            "pitch_var": float(np.var(f0))
        }

        zcr = float(librosa.feature.zero_crossing_rate(y)[0].mean())
        centroid = float(librosa.feature.spectral_centroid(y=y, sr=sr)[0].mean())

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        delta = librosa.feature.delta(mfcc)
        mfcc_stats = {
            f"mfcc{i}": float(mfcc[i].mean()) for i in range(mfcc.shape[0])
        }
        mfcc_stats.update({
            f"delta_mfcc{i}": float(delta[i].mean()) for i in range(delta.shape[0])
        })

        # Use OpenSMILE
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
            sf.write(temp_audio.name, y, sr)
            smile_feats = self.smile.process_file(temp_audio.name)
        os.unlink(temp_audio.name)

        smile_dict = smile_feats.iloc[0].to_dict()

        jitter_features = {
            "jitter_local": float(smile_dict.get("jitterLocal_sma[0]_amean", 0.0)),
            "jitter_rap": float(smile_dict.get("jitterDDP_sma[0]_amean", 0.0)),
            "jitter_ppq": float(smile_dict.get("jitterPPQ5_sma[0]_amean", 0.0)),
        }
        shimmer_features = {
            "shimmer_db": float(smile_dict.get("shimmerLocaldB_sma[0]_amean", 0.0)),
            "shimmer_apq3": float(smile_dict.get("shimmerAPQ3_sma[0]_amean", 0.0)),
            "shimmer_apq5": float(smile_dict.get("shimmerAPQ5_sma[0]_amean", 0.0)),
        }
        hnr_value = float(smile_dict.get("HNRdBACF_sma[0]_amean", 0.0))

        return {
            "zcr": zcr,
            "centroid": centroid,
            "hnr": hnr_value,
            **pitch_stats,
            **jitter_features,
            **shimmer_features,
            **mfcc_stats
        }

    def split_into_clips(self, audio):
        clips = []
        for i in range(0, len(audio) - self.clip_samples, self.clip_samples // 2):
            clip = audio[i:i + self.clip_samples]
            if len(clip) == self.clip_samples:
                clips.append(clip)
        return clips if clips else [audio[:self.clip_samples]]

    def build_instruction(self, features):
        return (
            f"Analyze this audio clip for signs of Parkinson's. Acoustic features: "
            f"pitch={features['pitch_mean']:.1f}Hz, zcr={features['zcr']:.3f}, "
            f"centroid={features['centroid']:.1f}Hz, jitter={features['jitter_local']:.3f}, "
            f"shimmer={features['shimmer_db']:.2f}dB, hnr={features['hnr']:.2f}."
        )

    def build_output(self, label):
        return ("Detected tremor and hesitations consistent with Parkinson's symptoms."
                if label == "PD" else
                "No significant signs of Parkinson's detected. Speech patterns appear normal.")

    def process_all(self, base_folders):
        all_messages = []
        audio_files = self.find_audio_files(base_folders)

        for f in tqdm(audio_files, desc="Processing"):
            y, sr = librosa.load(f["file_path"], sr=self.sample_rate)
            y = librosa.util.normalize(y)
            clips = self.split_into_clips(y)

            for i, clip in enumerate(clips):
                fname = f"{f['file_path'].stem}_clip{i+1}"
                label = f["label"]
                wav_path = self.audio_dir / label / f"{fname}.wav"
                json_path = self.features_dir / label / f"{fname}.json"

                sf.write(wav_path, clip, self.sample_rate)
                feats = self.extract_features(clip, self.sample_rate)
                with open(json_path, "w") as jf:
                    json.dump(feats, jf, indent=2)

                all_messages.append({
                    "messages": [
                        {
                            "role": "system",
                            "content": [
                                {"type": "text", "text": "You are an assistant that detects Parkinson’s disease from audio."}
                            ]
                        },
                        {
                            "role": "user",
                            "content": [
                                {"type": "audio", "audio": str(wav_path)},
                                {"type": "text", "text": self.build_instruction(feats)}
                            ]
                        },
                        {
                            "role": "assistant",
                            "content": [
                                {"type": "text", "text": self.build_output(label)}
                            ]
                        }
                    ]
                })

        # Save dataset
        dataset_path = self.output_dir / "messages_dataset.jsonl"
        with open(dataset_path, "w") as f:
            for m in all_messages:
                f.write(json.dumps(m) + "\n")
        print(f"\n✅ Final dataset saved to: {dataset_path}")
        print(f"Total examples: {len(all_messages)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--clip-duration", type=float, default=4.0)
    parser.add_argument("--n-fft", type=int, default=512)
    parser.add_argument("--hop-length", type=int, default=160)
    args = parser.parse_args()

    processor = ParkinsonFeatureExtractor(
        sample_rate=args.sample_rate,
        clip_duration=args.clip_duration,
        n_fft=args.n_fft,
        hop_length=args.hop_length
    )

    processor.process_all(base_folders=["ReadText", "SpontaneousDialogue"])
import os
import json
import numpy as np
if not hasattr(np, 'complex'):
    np.complex = complex
import librosa
import soundfile as sf
from pathlib import Path
from tqdm import tqdm
import argparse
import warnings
import opensmile
import tempfile
warnings.filterwarnings("ignore")


class ParkinsonFeatureExtractor:
    def __init__(self, sample_rate=16000, clip_duration=10.0, hop_length=160, n_fft=512):
        self.sample_rate = sample_rate
        self.clip_duration = clip_duration
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.clip_samples = int(sample_rate * clip_duration)

        self.output_dir = Path("processed_dataset")
        self.audio_dir = self.output_dir / "audio_clips"
        self.features_dir = self.output_dir / "acoustic_features"
        self.output_dir.mkdir(exist_ok=True)
        for label in ["PD", "Control"]:
            (self.audio_dir / label).mkdir(parents=True, exist_ok=True)
            (self.features_dir / label).mkdir(parents=True, exist_ok=True)

        # Initialize OpenSMILE
        self.smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.ComParE_2016,
            feature_level=opensmile.FeatureLevel.Functionals
        )

    def find_audio_files(self, base_folders):
        files = []
        for folder in base_folders:
            for label in ["PD", "HC"]:
                full = Path(folder) / label
                if not full.exists():
                    continue
                for f in full.glob("*.wav"):
                    files.append({
                        "file_path": f,
                        "label": "PD" if label == "PD" else "Control",
                        "source_folder": folder
                    })
        return files

    def extract_features(self, y, sr):
        f0 = librosa.yin(y, fmin=80, fmax=450, sr=sr)
        pitch_stats = {
            "pitch_mean": float(np.mean(f0)),
            "pitch_std": float(np.std(f0)),
            "pitch_min": float(np.min(f0)),
            "pitch_max": float(np.max(f0)),
            "pitch_var": float(np.var(f0))
        }

        zcr = float(librosa.feature.zero_crossing_rate(y)[0].mean())
        centroid = float(librosa.feature.spectral_centroid(y=y, sr=sr)[0].mean())

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        delta = librosa.feature.delta(mfcc)
        mfcc_stats = {
            f"mfcc{i}": float(mfcc[i].mean()) for i in range(mfcc.shape[0])
        }
        mfcc_stats.update({
            f"delta_mfcc{i}": float(delta[i].mean()) for i in range(delta.shape[0])
        })

        # Use OpenSMILE
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
            sf.write(temp_audio.name, y, sr)
            smile_feats = self.smile.process_file(temp_audio.name)
        os.unlink(temp_audio.name)

        smile_dict = smile_feats.iloc[0].to_dict()

        jitter_features = {
            "jitter_local": float(smile_dict.get("jitterLocal_sma[0]_amean", 0.0)),
            "jitter_rap": float(smile_dict.get("jitterDDP_sma[0]_amean", 0.0)),
            "jitter_ppq": float(smile_dict.get("jitterPPQ5_sma[0]_amean", 0.0)),
        }
        shimmer_features = {
            "shimmer_db": float(smile_dict.get("shimmerLocaldB_sma[0]_amean", 0.0)),
            "shimmer_apq3": float(smile_dict.get("shimmerAPQ3_sma[0]_amean", 0.0)),
            "shimmer_apq5": float(smile_dict.get("shimmerAPQ5_sma[0]_amean", 0.0)),
        }
        hnr_value = float(smile_dict.get("HNRdBACF_sma[0]_amean", 0.0))

        return {
            "zcr": zcr,
            "centroid": centroid,
            "hnr": hnr_value,
            **pitch_stats,
            **jitter_features,
            **shimmer_features,
            **mfcc_stats
        }

    def split_into_clips(self, audio):
        clips = []
        for i in range(0, len(audio) - self.clip_samples, self.clip_samples // 2):
            clip = audio[i:i + self.clip_samples]
            if len(clip) == self.clip_samples:
                clips.append(clip)
        return clips if clips else [audio[:self.clip_samples]]

    def build_instruction(self, features):
        return (
            f"Analyze this audio clip for signs of Parkinson's. Acoustic features: "
            f"pitch={features['pitch_mean']:.1f}Hz, zcr={features['zcr']:.3f}, "
            f"centroid={features['centroid']:.1f}Hz, jitter={features['jitter_local']:.3f}, "
            f"shimmer={features['shimmer_db']:.2f}dB, hnr={features['hnr']:.2f}."
        )

    def build_output(self, label):
        return ("Detected tremor and hesitations consistent with Parkinson's symptoms."
                if label == "PD" else
                "No significant signs of Parkinson's detected. Speech patterns appear normal.")

    def process_all(self, base_folders):
        all_messages = []
        audio_files = self.find_audio_files(base_folders)

        for f in tqdm(audio_files, desc="Processing"):
            y, sr = librosa.load(f["file_path"], sr=self.sample_rate)
            y = librosa.util.normalize(y)
            clips = self.split_into_clips(y)

            for i, clip in enumerate(clips):
                fname = f"{f['file_path'].stem}_clip{i+1}"
                label = f["label"]
                wav_path = self.audio_dir / label / f"{fname}.wav"
                json_path = self.features_dir / label / f"{fname}.json"

                sf.write(wav_path, clip, self.sample_rate)
                feats = self.extract_features(clip, self.sample_rate)
                with open(json_path, "w") as jf:
                    json.dump(feats, jf, indent=2)

                all_messages.append({
                    "messages": [
                        {
                            "role": "system",
                            "content": [
                                {"type": "text", "text": "You are an assistant that detects Parkinson’s disease from audio."}
                            ]
                        },
                        {
                            "role": "user",
                            "content": [
                                {"type": "audio", "audio": str(wav_path)},
                                {"type": "text", "text": self.build_instruction(feats)}
                            ]
                        },
                        {
                            "role": "assistant",
                            "content": [
                                {"type": "text", "text": self.build_output(label)}
                            ]
                        }
                    ]
                })

        # Save dataset
        dataset_path = self.output_dir / "messages_dataset.jsonl"
        with open(dataset_path, "w") as f:
            for m in all_messages:
                f.write(json.dumps(m) + "\n")
        print(f"\n✅ Final dataset saved to: {dataset_path}")
        print(f"Total examples: {len(all_messages)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--clip-duration", type=float, default=4.0)
    parser.add_argument("--n-fft", type=int, default=512)
    parser.add_argument("--hop-length", type=int, default=160)
    args = parser.parse_args()

    processor = ParkinsonFeatureExtractor(
        sample_rate=args.sample_rate,
        clip_duration=args.clip_duration,
        n_fft=args.n_fft,
        hop_length=args.hop_length
    )

    processor.process_all(base_folders=["ReadText", "SpontaneousDialogue"])
