from setuptools import setup, find_packages

setup(
    name="parkinsons-speech-therapy",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "streamlit>=1.32.0",
        "torch>=2.0.0",
        "transformers>=4.36.0",
        "numpy>=1.24.0",
        "librosa>=0.10.0",
        "sounddevice>=0.4.6",
        "scipy>=1.10.0",
        "matplotlib>=3.7.0",
        "pandas>=2.0.0",
        "plotly>=5.18.0",
        "praat-parselmouth>=0.4.3",
        "pyaudio>=0.2.13",
    ],
    python_requires=">=3.8",
)
