import streamlit as st
import sounddevice as sd
import numpy as np
import wave
import os
from datetime import datetime
from pathlib import Path

class AudioRecorder:
    def __init__(self):
        self.sample_rate = 16000
        self.channels = 1
        self.recordings_dir = Path("recordings")
        self.recordings_dir.mkdir(exist_ok=True)
        
    def record_audio(self, duration=5):
        """Record audio from the microphone
        
        Args:
            duration (int): Recording duration in seconds
            
        Returns:
            str: Path to the recorded audio file, or None if recording failed
        """
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Start Recording"):
                try:
                    # Show recording progress
                    progress_bar = st.progress(0)
                    st.markdown("Recording in progress...")
                    
                    # Record audio
                    recording = sd.rec(
                        int(duration * self.sample_rate),
                        samplerate=self.sample_rate,
                        channels=self.channels
                    )
                    
                    # Update progress bar
                    for i in range(duration):
                        progress_bar.progress((i + 1) / duration)
                        st.session_state.recording = True
                        sd.wait()
                    
                    # Save recording
                    filename = self.save_recording(recording)
                    st.success("Recording completed!")
                    
                    # Display audio playback
                    st.audio(str(filename))
                    
                    return str(filename)
                    
                except Exception as e:
                    st.error(f"Recording failed: {str(e)}")
                    return None
                    
        with col2:
            if duration > 10:
                st.info(
                    f"**Reading Time:** {duration} seconds\n\n"
                    "**Tips for best results:**\n"
                    "• Read at a normal, comfortable pace\n"
                    "• Speak clearly and naturally\n"
                    "• Take your time with each word\n"
                    "• Maintain consistent volume"
                )
            else:
                st.info(
                    f"Please speak for {duration} seconds when you click the "
                    "recording button"
                )
            
        return None
        
    def save_recording(self, recording):
        """Save the recorded audio to a WAV file
        
        Args:
            recording (np.ndarray): The recorded audio data
            
        Returns:
            str: Path to the saved audio file
        """
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.recordings_dir / f"recording_{timestamp}.wav"
        
        # Ensure recording is in the correct format
        recording = np.int16(recording * 32767)
        
        # Save as WAV file
        with wave.open(str(filename), 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(self.sample_rate)
            wf.writeframes(recording.tobytes())
            
        return str(filename)