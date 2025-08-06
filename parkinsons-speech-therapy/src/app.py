import streamlit as st
import os
import sys
import pandas as pd
from datetime import datetime
from pathlib import Path

# Add the src directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from components.audio_recorder import AudioRecorder
from components.model_prediction import ParkinsonsPrediction
from components.visualizations import VisualizationManager
from utils.audio_processing import AudioProcessor

# Create necessary directories
Path("recordings").mkdir(exist_ok=True)

def cleanup_audio_file(audio_file_path):
    """Clean up audio file after processing"""
    try:
        if audio_file_path and os.path.exists(audio_file_path):
            os.remove(audio_file_path)
            print(f"Cleaned up audio file: {audio_file_path}")
    except Exception as e:
        print(f"Failed to cleanup audio file {audio_file_path}: {e}")

@st.cache_resource
def load_parkinson_model():
    """Load and cache the Parkinson's prediction model"""
    from components.model_prediction import ParkinsonsPrediction
    return ParkinsonsPrediction()

@st.cache_resource  
def load_other_components():
    """Load and cache other components"""
    from components.audio_recorder import AudioRecorder
    from components.visualizations import VisualizationManager
    from utils.audio_processing import AudioProcessor
    
    return AudioRecorder(), VisualizationManager(), AudioProcessor()

class ParkinsonsTherapyApp:
    def __init__(self):
        st.set_page_config(
            page_title="Parkinson's Speech Therapy Assistant",
            page_icon="",
            layout="wide"
        )
        self.setup_session_state()
        self.load_components()
        
    def setup_session_state(self):
        """Initialize session state variables"""
        if 'conversation_history' not in st.session_state:
            st.session_state.conversation_history = []
        
        # Initialize model and components in session state for persistence
        if 'model' not in st.session_state:
            st.session_state.model = None
        if 'audio_recorder' not in st.session_state:
            st.session_state.audio_recorder = None
        if 'visualizer' not in st.session_state:
            st.session_state.visualizer = None
        if 'audio_processor' not in st.session_state:
            st.session_state.audio_processor = None
        if 'model_info' not in st.session_state:
            st.session_state.model_info = None
            
    def load_components(self):
        """Load all required components once and cache in session state"""
        # Only load if not already in session state
        if st.session_state.model is None:
            with st.spinner("Loading GGUF model and components... Please wait."):
                # Use cached functions for faster loading
                try:
                    st.session_state.model = load_parkinson_model()
                    audio_recorder, visualizer, audio_processor = load_other_components()
                    
                    st.session_state.audio_recorder = audio_recorder
                    st.session_state.visualizer = visualizer
                    st.session_state.audio_processor = audio_processor
                    
                    # Store model info for sidebar display
                    st.session_state.model_info = st.session_state.model.get_model_info()
                    
                    st.success("Model loaded successfully!")
                except Exception as e:
                    st.error(f"Failed to load components: {e}")
                    # Fallback to session state approach
                    st.session_state.model = None
        
        # Set instance variables from session state
        self.model = st.session_state.model
        self.audio_recorder = st.session_state.audio_recorder
        self.visualizer = st.session_state.visualizer
        self.audio_processor = st.session_state.audio_processor
        self.model_info = st.session_state.model_info
        
    def main(self):
        """Main application interface"""
        st.title("Parkinson's Speech Therapy Assistant")
        
        # Check if components are loaded, if not show loading message
        if st.session_state.model is None:
            st.info("Loading model for the first time... This may take a moment.")
            self.load_components()
        
        # Sidebar for mode selection
        with st.sidebar:
            st.header("Navigation")
            mode = st.radio(
                "Select Mode",
                ["Detection", "Speech Therapy", "Companion Mode"]
            )
            
            # Model Status Display
            st.header("Model Status")
            if self.model.is_simplified:
                st.warning("Using Acoustic Analysis")
                st.caption("Simplified feature-based detection")
            else:
                st.success("GGUF Model Active")
                st.caption("Fine-tuned Gemma 3n model loaded")
            
            st.info(f"Type: {self.model_info.get('model_type', 'Unknown')}")
                    
            # Force refresh button
            if st.button("Force Reload Model"):
                # Clear session state to force reload
                for key in ['model', 'audio_recorder', 'visualizer', 'audio_processor', 'model_info']:
                    if key in st.session_state:
                        del st.session_state[key]
                # Clear the cached functions
                load_parkinson_model.clear()
                load_other_components.clear()
                st.cache_data.clear()
                st.cache_resource.clear()
                st.rerun()
        
        # Main content area
        if mode == "Detection":
            self.detection_mode()
        elif mode == "Speech Therapy":
            self.therapy_mode()
        else:
            self.companion_mode()
            
    def detection_mode(self):
        """Parkinson's detection interface"""
        st.header("Parkinson's Detection")
        
        # Standardized paragraph for consistent analysis
        st.subheader("Standardized Reading Passage")
        st.markdown("""
        **Please read the following paragraph aloud for consistent analysis:**
        
        > "The quick brown fox jumps over the lazy dog. This pangram contains every letter of the alphabet at least once. 
        > It is commonly used to test typing speed and keyboard layouts. The sentence is often used in voice recognition 
        > systems and speech therapy exercises because it contains a good variety of sounds and phonetic elements."
        """)
        
        st.info("**Instructions:** Read the paragraph above clearly and at a normal pace. This standardized text helps ensure consistent and accurate analysis results.")
        
        # Explanation of why standardized text is important
        with st.expander("Why use a standardized paragraph?"):
            st.markdown("""
            **Consistency Benefits:**
            - **Same phonetic content** for all users
            - **Controlled variables** for accurate comparison
            - **Reproducible results** across different sessions
            - **Better baseline** for tracking progress over time
            - **Reduced variability** from different speech content
            
            **What we analyze:**
            - Voice tremor and stability
            - Pitch consistency
            - Volume control
            - Speech clarity
            - Voice quality characteristics
            """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            audio_file = self.audio_recorder.record_audio(duration=15)  # Increased duration for paragraph
            
        with col2:
            if audio_file:
                st.info("Processing audio...")
                
                # Debug: Show model status
                st.write(f"Model simplified: {self.model.is_simplified}")
                st.write(f"Model type: {type(self.model.model)}")
                
                # Get model prediction directly from audio file
                with st.spinner("Analyzing speech patterns..."):
                    prediction = self.model.predict(audio_file)
                
                # Debug: Show prediction details
                st.write("**Debug Info:**")
                st.write(f"Prediction keys: {list(prediction.keys())}")
                st.write(f"Raw output: {prediction.get('raw_output', 'N/A')}")
                
                # Extract features for visualization
                features = prediction.get('features', {})
                
                # Display results
                self.display_detection_results(prediction, features)
                
                # Clean up audio file after processing
                cleanup_audio_file(audio_file)
                
    def therapy_mode(self):
        """Speech therapy exercise interface"""
        st.header("Speech Therapy Exercises")
        
        exercises = {
            "Sustained Vowel": {
                "description": "Sustain the vowel 'ahhhh' for 5 seconds",
                "duration": 5,
                "type": "vowel_sounds"
            },
            "Word Repetition": {
                "description": "Repeat the following words clearly: 'blue, green, red'",
                "duration": 10,
                "type": "reading_aloud"
            },
            "Tongue Twisters": {
                "description": "Say 'Peter Piper picked a peck of pickled peppers' clearly",
                "duration": 8,
                "type": "tongue_twisters"
            },
            "Loudness Training": {
                "description": "Say 'Strong voice!' with increasing volume",
                "duration": 5,
                "type": "volume_control"
            },
            "Rate Control": {
                "description": "Count from 1 to 10 slowly and clearly",
                "duration": 15,
                "type": "reading_aloud"
            }
        }
        
        exercise = st.selectbox("Choose Exercise", list(exercises.keys()))
        
        st.info(exercises[exercise]["description"])
        
        audio_file = self.audio_recorder.record_audio(
            duration=exercises[exercise]["duration"]
        )
        
        if audio_file:
            # Get prediction first
            with st.spinner("Analyzing speech patterns..."):
                prediction = self.model.predict(audio_file)
            
            # Generate therapy feedback using the prediction
            feedback = self.model.generate_therapy_feedback(
                audio_file,
                prediction,
                exercise_type=exercises[exercise]["type"]
            )
            
            # Extract features for visualization
            features = prediction.get('features', {})
            
            self.display_therapy_feedback(feedback, features)
            
            # Clean up audio file after processing
            cleanup_audio_file(audio_file)
            
    def companion_mode(self):
        """AI companion chat interface"""
        st.header("AI Companion")
        st.info(
            "Chat with your AI companion about your progress, concerns, or just for "
            "motivation and support."
        )
        
        # Display chat history
        for message in st.session_state.conversation_history:
            with st.chat_message(message["role"]):
                st.write(message["content"])
                
        # Chat input
        user_input = st.chat_input("Type your message here...")
        
        if user_input:
            # Add user message to history
            st.session_state.conversation_history.append({
                "role": "user",
                "content": user_input
            })
            
            # Generate AI response
            response_data = self.model.generate_companion_response(
                user_input, 
                st.session_state.conversation_history
            )
            
            # Extract the actual response text
            response_text = response_data.get('response', 'I understand. How can I help you today?')
            
            # Add AI response to history
            st.session_state.conversation_history.append({
                "role": "assistant",
                "content": response_text
            })
            
            # Force rerun to update chat
            st.rerun()
            
    def display_detection_results(self, prediction, features):
        """Display detection results and visualizations"""
        st.subheader("Analysis Results")
        
        confidence = prediction['confidence']
        prediction_text = prediction['prediction']
        
        # Check if Parkinson's indicators were detected
        has_parkinsons = ("parkinson" in prediction_text.lower() and 
                         "detected" in prediction_text.lower()) or \
                        ("indicators" in prediction_text.lower() and 
                         "detected" in prediction_text.lower()) or \
                        ("possible parkinson" in prediction_text.lower()) or \
                        ("irregularities" in prediction_text.lower()) or \
                        ("abnormal" in prediction_text.lower())
        
        # Show as healthy only if explicitly classified as control/healthy
        is_healthy = ("control" in prediction_text.lower() and "healthy" in prediction_text.lower()) or \
                    ("no clear" in prediction_text.lower() and "indicators" in prediction_text.lower())
        
        if has_parkinsons and not is_healthy:
            st.warning(
                f"{prediction_text} (Confidence: {confidence*100:.1f}%)"
            )
        else:
            st.success(
                f"{prediction_text} (Confidence: {confidence*100:.1f}%)"
            )
        
        # Display additional information if available
        if 'risk_score' in prediction:
            st.info(f"Risk Score: {prediction['risk_score']}")
            
        if 'risk_factors' in prediction and prediction['risk_factors']:
            st.write("**Risk Factors Identified:**")
            for factor in prediction['risk_factors']:
                st.write(f"• {factor}")
        
        if 'analysis_type' in prediction:
            analysis_type = prediction['analysis_type']
            if analysis_type == 'acoustic_features':
                st.info("Analysis based on acoustic feature extraction")
            elif analysis_type == 'error':
                st.error("Analysis encountered errors")
            
        # Display feature visualizations
        self.visualizer.plot_audio_features(features)
        
    def display_therapy_feedback(self, feedback, features):
        """Display therapy exercise feedback and visualizations"""
        st.subheader("Exercise Feedback")
        
        # Display the overall assessment
        if 'overall_assessment' in feedback:
            st.info(feedback['overall_assessment'])
        
        # Display specific areas to work on
        if 'specific_areas' in feedback and feedback['specific_areas']:
            st.subheader("Areas to Focus On")
            for area in feedback['specific_areas']:
                st.write(f"• {area}")
        
        # Display recommended exercises
        if 'exercises' in feedback and feedback['exercises']:
            st.subheader("Recommended Exercises")
            for exercise in feedback['exercises']:
                st.write(f"• {exercise}")
        
        # Display encouragement
        if 'encouragement' in feedback:
            st.success(feedback['encouragement'])
        
        # Display progress note
        if 'progress_note' in feedback:
            st.write(f"**Progress Note:** {feedback['progress_note']}")
        
        # Show basic feature analysis if available
        if features:
            st.subheader("Voice Analysis")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                duration = features.get('duration', 0)
                st.metric("Duration", f"{duration:.1f}s")
            
            with col2:
                jitter = features.get('jitter', 0) * 100
                st.metric("Voice Tremor", f"{jitter:.2f}%")
            
            with col3:
                shimmer = features.get('shimmer', 0) * 100
                st.metric("Voice Stability", f"{shimmer:.2f}%")
        
        # Show feature visualizations if visualizer supports it
        try:
            self.visualizer.plot_audio_features(features)
        except Exception as e:
            st.warning("Feature visualization not available")

if __name__ == "__main__":
    app = ParkinsonsTherapyApp()
    app.main()