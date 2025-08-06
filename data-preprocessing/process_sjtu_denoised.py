import os
import json
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from tqdm import tqdm
import argparse
import warnings
warnings.filterwarnings("ignore")

from opensmile import Smile, FeatureSet, FeatureLevel

class SJTUDenoisedProcessor:
    def __init__(self, sample_rate=16000, clip_duration=10.0):
        self.sample_rate = sample_rate
        self.clip_duration = clip_duration
        self.clip_samples = int(sample_rate * clip_duration)

        self.output_dir = Path("processed_sjtu_denoised")
        self.audio_dir = self.output_dir / "audio_clips"
        self.features_dir = self.output_dir / "acoustic_features"
        self.output_dir.mkdir(exist_ok=True)
        for label in ["PD", "Control"]:
            (self.audio_dir / label).mkdir(parents=True, exist_ok=True)
            (self.features_dir / label).mkdir(parents=True, exist_ok=True)

        self.smile = Smile(
            feature_set=FeatureSet.ComParE_2016,
            feature_level=FeatureLevel.Functionals
        )

    def find_audio_files(self):
        audio_files = []
        folders = [
            "Parkinson-Patient-Speech-Dataset/denoised-speech-dataset/DL",
            "Parkinson-Patient-Speech-Dataset/denoised-speech-dataset/Faces", 
            "Parkinson-Patient-Speech-Dataset/denoised-speech-dataset/LW",
            "Parkinson-Patient-Speech-Dataset/denoised-speech-dataset/Tessi",
            "Parkinson-Patient-Speech-Dataset/denoised-speech-dataset/emma"
        ]

        for folder in folders:
            if not os.path.exists(folder):
                continue
            for f in Path(folder).glob("*.wav"):
                audio_files.append({
                    "file_path": f,
                    "label": "PD",
                    "source_folder": Path(folder).name
                })
        return audio_files

    def extract_features(self, y, sr, temp_wav_path):
        # Save clip temporarily for OpenSMILE
        sf.write(temp_wav_path, y, sr)
        feats = self.smile.process_file(temp_wav_path)
        os.remove(temp_wav_path)

        # Subset of relevant OpenSMILE features
        def get(feat):
            return float(feats[feat].values[0]) if feat in feats.columns else 0.0

        jitter = get("jitterLocal_sma[0]_percentile1.0")
        shimmer = get("shimmerLocaldB_sma[0]_percentile1.0")
        hnr = get("HNRdBACF_sma[0]_amean")

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

        return {
            "zcr": zcr,
            "centroid": centroid,
            "jitter": jitter,
            "shimmer": shimmer,
            "hnr": hnr,
            **pitch_stats
        }

    def split_into_clips(self, audio):
        clips = []
        for i in range(0, len(audio) - self.clip_samples, self.clip_samples // 2):
            clip = audio[i:i + self.clip_samples]
            if len(clip) == self.clip_samples:
                clips.append(clip)
        return clips if clips else [audio[:self.clip_samples]]

    def build_instruction(self, f):
        return (
            f"Analyze this audio clip for signs of Parkinson's. Acoustic features: "
            f"pitch={f['pitch_mean']:.1f}Hz, zcr={f['zcr']:.3f}, "
            f"centroid={f['centroid']:.1f}Hz, jitter={f['jitter']:.3f}, "
            f"shimmer={f['shimmer']:.2f}dB, hnr={f['hnr']:.2f}."
        )

    def build_output(self, label):
        return ("Detected tremor and hesitations consistent with Parkinson's symptoms."
                if label == "PD"
                else "No significant signs of Parkinson's detected. Speech patterns appear normal.")

    def process_dataset(self):
        print("🔍 Searching for SJTU denoised audio files...")
        files = self.find_audio_files()
        print(f"✅ Found {len(files)} .wav files.\n")

        all_messages = []
        temp_wav_path = "_temp_clip.wav"

        for file in tqdm(files, desc="Processing audio"):
            y, sr = librosa.load(file["file_path"], sr=self.sample_rate)
            y = librosa.util.normalize(y)
            clips = self.split_into_clips(y)

            for i, clip in enumerate(clips):
                fname = f"{file['file_path'].stem}_clip{i+1}"
                label = file["label"]

                audio_path = self.audio_dir / label / f"{fname}.wav"
                feat_path = self.features_dir / label / f"{fname}.json"
                sf.write(audio_path, clip, self.sample_rate)

                feats = self.extract_features(clip, self.sample_rate, temp_wav_path)
                with open(feat_path, "w") as f:
                    json.dump(feats, f, indent=2)

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
                                {"type": "audio", "audio": str(audio_path)},
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

        out_path = self.output_dir / "messages_dataset.jsonl"
        with open(out_path, "w") as f:
            for m in all_messages:
                f.write(json.dumps(m) + "\n")

        print(f"\n📁 Final dataset written to: {out_path}")
        print(f"🧾 Total examples: {len(all_messages)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--clip-duration", type=float, default=10.0)
    args = parser.parse_args()

    processor = SJTUDenoisedProcessor(
        sample_rate=args.sample_rate,
        clip_duration=args.clip_duration
    )
    processor.process_dataset()
