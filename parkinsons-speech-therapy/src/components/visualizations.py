import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

class VisualizationManager:
    def plot_audio_features(self, features):
        """Plot audio features analysis
        
        Args:
            features (dict): Dictionary of audio features
        """
        if not features or 'error' in features:
            st.warning("Could not extract audio features for visualization")
            return
            
        st.subheader("Audio Feature Analysis")
        
        # Create radar chart of key features
        fig = go.Figure()
        
        # Get available features and normalize them
        jitter = features.get('jitter', 0) * 100  # Convert to percentage
        shimmer = features.get('shimmer', 0) * 100  # Convert to percentage
        f0_std = features.get('f0_std', 0) / 50  # Normalize pitch variation
        speech_rate = features.get('speech_rate', 0) / 10  # Normalize speech rate
        pause_ratio = features.get('pause_ratio', 0)  # Already 0-1
        
        # Normalize features to 0-1 scale for radar chart
        feature_values = [
            min(jitter / 2, 1),  # Jitter as percentage, cap at 2%
            min(shimmer / 5, 1),  # Shimmer as percentage, cap at 5%
            min(f0_std, 1),  # Pitch variation
            min(speech_rate, 1),  # Speech rate
            pause_ratio  # Pause ratio
        ]
        
        feature_names = [
            'Jitter (%)',
            'Shimmer (%)',
            'Pitch Variation',
            'Speech Rate',
            'Pause Ratio'
        ]
        
        fig.add_trace(go.Scatterpolar(
            r=feature_values,
            theta=feature_names,
            fill='toself',
            name='Current Recording'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=False
        )
        
        st.plotly_chart(fig)
        
        # Display numerical values
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Jitter", f"{jitter:.3f}%")
            st.metric("Speech Rate", f"{features.get('speech_rate', 0):.1f} syll/sec")
            
        with col2:
            st.metric("Shimmer", f"{shimmer:.3f}%")
            st.metric("F0 Mean", f"{features.get('f0_mean', 0):.1f} Hz")
            
        with col3:
            st.metric("Pause Ratio", f"{pause_ratio:.2f}")
            st.metric("Duration", f"{features.get('duration', 0):.1f} sec")
        
    def plot_therapy_metrics(self, features, feedback):
        """Plot therapy exercise metrics
        
        Args:
            features (dict): Audio features from exercise
            feedback (dict): Model feedback including scores
        """
        if not features or 'error' in features:
            st.warning("Could not extract features for therapy metrics")
            return
            
        st.subheader("Exercise Performance Metrics")
        
        # Display key metrics for therapy
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Voice Quality Metrics")
            jitter = features.get('jitter', 0) * 100
            shimmer = features.get('shimmer', 0) * 100
            
            # Create bar chart for voice quality
            quality_fig = go.Figure(data=[
                go.Bar(name='Current', x=['Jitter (%)', 'Shimmer (%)'], 
                       y=[jitter, shimmer],
                       marker_color=['lightblue', 'lightgreen'])
            ])
            quality_fig.update_layout(
                title="Voice Stability Metrics",
                yaxis_title="Percentage"
            )
            st.plotly_chart(quality_fig, use_container_width=True)
            
        with col2:
            st.subheader("Speech Characteristics")
            
            # Display speech metrics
            st.metric("Speech Rate", f"{features.get('speech_rate', 0):.1f} syllables/sec")
            st.metric("Average Pitch", f"{features.get('f0_mean', 0):.1f} Hz")
            st.metric("Pitch Variation", f"{features.get('f0_std', 0):.1f} Hz")
            st.metric("Pause Ratio", f"{features.get('pause_ratio', 0):.2f}")
        
        # MFCC visualization
        st.subheader("Spectral Analysis")
        mfcc_values = []
        mfcc_names = []
        for i in range(13):
            if f'mfcc_{i}_mean' in features:
                mfcc_values.append(features[f'mfcc_{i}_mean'])
                mfcc_names.append(f'MFCC {i}')
        
        if mfcc_values:
            mfcc_fig = go.Figure(data=[
                go.Bar(x=mfcc_names, y=mfcc_values, marker_color='orange')
            ])
            mfcc_fig.update_layout(
                title="MFCC Features (Speech Characteristics)",
                xaxis_title="MFCC Coefficients",
                yaxis_title="Value"
            )
            st.plotly_chart(mfcc_fig, use_container_width=True)
    
    def plot_progress_history(self, history_df):
        """Plot historical progress data
        
        Args:
            history_df (pd.DataFrame): DataFrame of exercise history
        """
        st.subheader("Progress Over Time")
        
        # Check which score columns are available
        available_scores = []
        if 'clarity_score' in history_df.columns:
            available_scores.append('clarity_score')
        if 'stability_score' in history_df.columns:
            available_scores.append('stability_score')
        if 'volume_score' in history_df.columns:
            available_scores.append('volume_score')
        if 'overall_score' in history_df.columns:
            available_scores.append('overall_score')
            
        if not available_scores:
            st.warning("No score data available for plotting.")
            return
            
        # Create line plot of scores over time
        fig = px.line(
            history_df,
            x='timestamp',
            y=available_scores,
            title="Score Progress",
            labels={
                'timestamp': 'Date',
                'value': 'Score',
                'variable': 'Metric'
            }
        )
        
        fig.update_layout(
            hovermode='x unified',
            yaxis_range=[0, 100]  # Assuming scores are 0-100
        )
        
        st.plotly_chart(fig)
        
        # Show exercise type distribution
        st.subheader("Exercise Type Distribution")
        
        exercise_counts = history_df['exercise_type'].value_counts()
        
        fig = px.pie(
            values=exercise_counts.values,
            names=exercise_counts.index,
            title="Exercises Completed"
        )
        
        st.plotly_chart(fig)