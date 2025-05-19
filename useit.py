"""
Speech Emotion Classification Streamlit App

This application allows users to:
1. Upload audio files (.wav)
2. Process them with a trained speech emotion classification model
3. Visualize the predictions with confidence scores
4. Process multiple files in batch mode

Author: Armaan Mittal
Date: May 15, 2025
"""

import os
import io
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import librosa
import librosa.display
import matplotlib.pyplot as plt
import streamlit as st
import plotly.graph_objects as go
from pathlib import Path
from PIL import Image
from matplotlib.colors import LinearSegmentedColormap
import streamlit.components.v1 as components
import tempfile
import base64

st.set_page_config(
    page_title="Speech Emotion Classifier",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

def local_css():
    st.markdown("""
    <style>
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        h1, h2, h3 {
            margin-bottom: 1rem;
        }
        .stButton button {
            width: 100%;
            border-radius: 5px;
            height: 3em;
            font-weight: 500;
        }
        .stProgress .st-eb {
            background-color: #4CAF50;
        }
        .emotion-card {
            padding: 1.5rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            background-color: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        .emotion-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 15px rgba(0, 0, 0, 0.2);
        }
        .prediction-result {
            font-size: 1.5rem;
            font-weight: bold;
            padding: 1rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
            text-align: center;
        }
        .batch-table {
            width: 100%;
            margin-top: 1rem;
        }
        .audio-container {
            margin-top: 1rem;
            margin-bottom: 1rem;
            padding: 1rem;
            border-radius: 0.5rem;
            background-color: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(5px);
        }
        .stAudio {
            margin-top: 0.5rem;
        }
        @media (max-width: 768px) {
            .mobile-col {
                min-width: 100%;
            }
        }
        .stAlert {
            background-color: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
        }
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

EMOTION_CLASSES = {
    0: {'name': 'neutral', 'emoji': '😐', 'color': '#8884d8'},
    1: {'name': 'calm', 'emoji': '😌', 'color': '#83a6ed'},
    2: {'name': 'happy', 'emoji': '😄', 'color': '#ffc658'},
    3: {'name': 'sad', 'emoji': '😢', 'color': '#8dd1e1'},
    4: {'name': 'angry', 'emoji': '😠', 'color': '#ff6b6b'},
    5: {'name': 'fearful', 'emoji': '😨', 'color': '#a4de6c'},
    6: {'name': 'disgust', 'emoji': '🤢', 'color': '#d0ed57'},
    7: {'name': 'surprised', 'emoji': '😲', 'color': '#ffbb96'}
}

class SpeechEmotionCNN(nn.Module):
    def __init__(self, n_mels=64, num_classes=8):
        super(SpeechEmotionCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.adaptive_pool = nn.AdaptiveAvgPool2d((2, 2))
        
        self.fc_input_size = 512 * 2 * 2
        
        self.fc1 = nn.Linear(self.fc_input_size, 512)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, num_classes)
        
    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        
        x = self.adaptive_pool(x)
        
        x = x.view(-1, self.fc_input_size)
        
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        
        return x

@st.cache_resource
def load_model(model_path):
    """
    Load the trained PyTorch model from disk
    
    Args:
        model_path: Path to the saved model checkpoint
        
    Returns:
        Loaded PyTorch model in evaluation mode
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = SpeechEmotionCNN(n_mels=64, num_classes=len(EMOTION_CLASSES))
    
    checkpoint = torch.load(model_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    return model, device

def preprocess_audio(audio_file, sample_rate=16000, n_mels=64, max_length=None):
    """
    Preprocess the audio file for the model
    
    Args:
        audio_file: Path or file-like object with audio
        sample_rate: Target sample rate
        n_mels: Number of mel bands
        max_length: Maximum sequence length (optional)
        
    Returns:
        Preprocessed mel spectrogram tensor ready for model input
    """
    if isinstance(audio_file, (str, Path)):
        y, _ = librosa.load(audio_file, sr=sample_rate, mono=True)
    else:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            tmp_file.write(audio_file.getvalue())
            tmp_file_path = tmp_file.name
        
        y, _ = librosa.load(tmp_file_path, sr=sample_rate, mono=True)
        
        try:
            os.unlink(tmp_file_path)
        except:
            pass
    
    max_samples = 10 * sample_rate
    if len(y) > max_samples:
        y = y[:max_samples]
    
    audio_tensor = torch.FloatTensor(y)
    
    mel_spec_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=1024,
        hop_length=512,
        n_mels=n_mels
    )
    
    mel_spec = mel_spec_transform(audio_tensor)
    
    mel_spec = torch.log(mel_spec + 1e-9)
    
    mel_spec = mel_spec.unsqueeze(0).unsqueeze(0)
    
    if max_length is not None and mel_spec.shape[3] < max_length:
        padding = (0, max_length - mel_spec.shape[3])
        mel_spec = F.pad(mel_spec, padding, "constant", 0)
    
    return mel_spec, y

def predict_emotion(model, mel_spec, device):
    """
    Run the model on preprocessed audio and get emotion predictions
    
    Args:
        model: PyTorch model
        mel_spec: Preprocessed mel spectrogram tensor
        device: Device to run inference on
        
    Returns:
        Softmax probabilities for each class
    """
    mel_spec = mel_spec.to(device)
    
    with torch.no_grad():
        with torch.cuda.amp.autocast() if device.type == 'cuda' else torch.no_grad():
            outputs = model(mel_spec)
            probs = F.softmax(outputs, dim=1)[0].cpu().numpy()
    
    return probs

def plot_waveform(y, sr=16000):
    """
    Plot the waveform of the audio signal
    
    Args:
        y: Audio signal
        sr: Sample rate
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 2))
    plt.box(False)
    librosa.display.waveshow(y, sr=sr, ax=ax, color="#1DB954")
    ax.set_title("Audio Waveform", fontsize=12)
    ax.set_xlabel("Time (s)", fontsize=10)
    ax.set_yticks([])
    plt.tight_layout()
    return fig

def plot_melspectrogram(y, sr=16000, n_mels=64):
    """
    Plot the mel spectrogram of the audio signal
    
    Args:
        y: Audio signal
        sr: Sample rate
        n_mels: Number of mel bands
        
    Returns:
        Matplotlib figure
    """
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    S_db = librosa.power_to_db(S, ref=np.max)
    
    colors = [(0, 0, 0.3), (0, 0.3, 0.8), (0, 0.5, 0.5), (0.5, 0.5, 0), (1, 0.5, 0), (1, 0, 0)]
    cmap = LinearSegmentedColormap.from_list('emotion_cmap', colors, N=100)
    
    fig, ax = plt.subplots(figsize=(10, 4))
    img = librosa.display.specshow(S_db, x_axis='time', y_axis='mel', sr=sr, ax=ax, cmap=cmap)
    ax.set_title("Mel Spectrogram", fontsize=12)
    ax.set_xlabel("Time (s)", fontsize=10)
    ax.set_ylabel("Mel Frequency", fontsize=10)
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    plt.tight_layout()
    return fig

def create_emotion_chart(probs):
    """
    Create a horizontal bar chart of emotion probabilities
    
    Args:
        probs: Probability array for each emotion class
        
    Returns:
        Plotly figure
    """
    labels = [f"{EMOTION_CLASSES[i]['emoji']} {EMOTION_CLASSES[i]['name'].capitalize()}" for i in range(len(EMOTION_CLASSES))]
    colors = [EMOTION_CLASSES[i]['color'] for i in range(len(EMOTION_CLASSES))]
    
    sorted_indices = np.argsort(probs)[::-1]
    sorted_probs = probs[sorted_indices]
    sorted_labels = [labels[i] for i in sorted_indices]
    sorted_colors = [colors[i] for i in sorted_indices]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=sorted_labels,
        x=sorted_probs * 100,  
        orientation='h',
        marker=dict(
            color=sorted_colors,
            line=dict(color='rgba(0, 0, 0, 0.5)', width=1)
        ),
        text=[f"{p:.1f}%" for p in sorted_probs * 100],
        textposition='outside'
    ))
    
    fig.update_layout(
        title=dict(
            text="Emotion Confidence Scores",
            font=dict(size=18),
            x=0.5
        ),
        xaxis=dict(
            title="Confidence (%)",
            range=[0, 110]  
        ),
        yaxis=dict(
            title=None,
            tickfont=dict(size=14)
        ),
        margin=dict(l=20, r=20, t=50, b=20),
        height=400,
        plot_bgcolor='rgba(0,0,0,0.03)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

def load_animation():
    """
    Load custom wave animation HTML
    
    Returns:
        HTML for wave animation
    """
    # Wave animation (HTML/CSS)
    wave_html = """
    <div class="wave-container" style="display: flex; justify-content: center; margin: 20px 0;">
        <style>
            .wave {
                width: 5px;
                height: 60px;
                background: linear-gradient(45deg, #1DB954, #4CAF50);
                margin: 10px;
                animation: wave 1s linear infinite;
                border-radius: 20px;
            }
            .wave:nth-child(2) {
                animation-delay: 0.1s;
            }
            .wave:nth-child(3) {
                animation-delay: 0.2s;
            }
            .wave:nth-child(4) {
                animation-delay: 0.3s;
            }
            .wave:nth-child(5) {
                animation-delay: 0.4s;
            }
            .wave:nth-child(6) {
                animation-delay: 0.5s;
            }
            .wave:nth-child(7) {
                animation-delay: 0.6s;
            }
            .wave:nth-child(8) {
                animation-delay: 0.7s;
            }
            .wave:nth-child(9) {
                animation-delay: 0.8s;
            }
            .wave:nth-child(10) {
                animation-delay: 0.9s;
            }
            
            @keyframes wave {
                0% {
                    transform: scale(1);
                }
                50% {
                    transform: scale(1, 0.3);
                }
                100% {
                    transform: scale(1);
                }
            }
        </style>
        <div class="wave"></div>
        <div class="wave"></div>
        <div class="wave"></div>
        <div class="wave"></div>
        <div class="wave"></div>
        <div class="wave"></div>
        <div class="wave"></div>
        <div class="wave"></div>
        <div class="wave"></div>
        <div class="wave"></div>
    </div>
    """
    return wave_html

def add_logo():
    """Add app logo"""
    st.markdown(
        """
        <div style="text-align: center; margin-bottom: 20px;">
            <h1 style="margin-bottom: 0;">🎙️ Speech Emotion Classifier</h1>
            <p style="font-size: 1.2rem; opacity: 0.8; margin-top: 0;">Detect emotions in speech using AI</p>
        </div>
        """, 
        unsafe_allow_html=True
    )

def main():
    """Main function for Streamlit app"""
    # Apply CSS
    local_css()
    
    # App logo
    add_logo()
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Model selection
        st.subheader("Model Configuration")
        model_path = st.text_input(
            "Model Path", 
            value="best_speech_emotion_model.pth", 
            help="Path to your trained PyTorch model file (.pth)"
        )
        
        # Mode selection
        st.subheader("Processing Mode")
        batch_mode = st.toggle("Batch Processing", help="Enable to process multiple audio files at once")
        
        # Sample rate selection
        st.subheader("Audio Settings")
        sample_rate = st.selectbox(
            "Sample Rate", 
            options=[8000, 16000, 22050, 44100], 
            index=1,
            help="Sample rate for audio processing"
        )
        
        # # Information
        # st.subheader("ℹ️ Information")
        # st.info(
        #     "This app detects emotions in speech using a CNN model trained on RAVDESS and TESS datasets. "
        #     "Upload an audio file (.wav) to get started!"
        # )
        
        # Device info
        if torch.cuda.is_available():
            device_info = f"🚀 CUDA available: {torch.cuda.get_device_name(0)}"
        else:
            device_info = "💻 Running on CPU"
        st.write(device_info)
        
    # Main area
    if not batch_mode:
        # Single file mode
        st.header("🎧 Upload Audio")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Upload a WAV file (5-10 seconds recommended)",
            type=["wav"],
            help="Upload an audio file to analyze its emotional content",
            accept_multiple_files=False
        )
        
        if uploaded_file:
            # Audio playback
            st.subheader("🔊 Audio Playback")
            col1, col2 = st.columns([3, 1])
            with col1:
                st.audio(uploaded_file, format="audio/wav")
            with col2:
                duration = st.empty()
            
            # Process button
            if st.button("🔍 Analyze Emotion", key="analyze_btn"):
                try:
                    # Load model
                    with st.spinner("Loading model..."):
                        model, device = load_model(model_path)
                    
                    # Process audio
                    with st.spinner("Processing audio..."):
                        # Preprocess audio
                        mel_spec, y = preprocess_audio(uploaded_file, sample_rate=sample_rate)
                        
                        # Update duration
                        audio_duration = len(y) / sample_rate
                        duration.info(f"Duration: {audio_duration:.2f}s")
                        
                        # Show waveform and mel spectrogram
                        col1, col2 = st.columns(2)
                        with col1:
                            st.pyplot(plot_waveform(y, sr=sample_rate))
                        with col2:
                            st.pyplot(plot_melspectrogram(y, sr=sample_rate))
                        
                        # Display wave animation during prediction
                        # components.html(load_animation(), height=100)
                        
                        # Get prediction
                        probs = predict_emotion(model, mel_spec, device)
                        
                        # Get top prediction
                        top_idx = np.argmax(probs)
                        top_emotion = EMOTION_CLASSES[top_idx]
                        
                        # Display results
                        st.subheader("🔮 Prediction Results")
                        
                        # Top emotion with emoji and confidence
                        result_html = f"""
                        <div class="prediction-result" style="background-color: {top_emotion['color']}20; color: {top_emotion['color']};">
                            {top_emotion['emoji']} 
                            Detected Emotion: {top_emotion['name'].upper()} 
                            ({probs[top_idx]*100:.1f}%)
                        </div>
                        """
                        st.markdown(result_html, unsafe_allow_html=True)
                        
                        # Chart with all emotions
                        st.plotly_chart(create_emotion_chart(probs), use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Error processing audio: {str(e)}")
                    st.error("Please make sure you've selected a valid model path and uploaded a valid WAV file.")
    else:
        # Batch mode
        st.header("📦 Batch Processing")
        
        # Multiple file uploader
        uploaded_files = st.file_uploader(
            "Upload multiple WAV files",
            type=["wav"],
            help="Upload multiple audio files to analyze",
            accept_multiple_files=True
        )
        
        if uploaded_files:
            st.write(f"Uploaded {len(uploaded_files)} files")
            
            # Process button
            if st.button("🔍 Analyze All Files", key="batch_analyze_btn"):
                try:
                    # Load model once
                    with st.spinner("Loading model..."):
                        model, device = load_model(model_path)
                    
                    # Progress bar
                    progress_bar = st.progress(0)
                    
                    # Results container
                    results = []
                    
                    # Process each file
                    for i, file in enumerate(uploaded_files):
                        # Update progress
                        progress_bar.progress((i + 1) / len(uploaded_files))
                        
                        # Status message
                        status_msg = st.empty()
                        status_msg.info(f"Processing {file.name}...")
                        
                        # Process audio
                        mel_spec, y = preprocess_audio(file, sample_rate=sample_rate)
                        probs = predict_emotion(model, mel_spec, device)
                        
                        # Get top prediction
                        top_idx = np.argmax(probs)
                        top_emotion = EMOTION_CLASSES[top_idx]['name']
                        top_emoji = EMOTION_CLASSES[top_idx]['emoji']
                        confidence = probs[top_idx] * 100
                        
                        # Store results
                        results.append({
                            "filename": file.name,
                            "emotion": top_emotion,
                            "emoji": top_emoji,
                            "confidence": confidence,
                            "duration": len(y) / sample_rate
                        })
                    
                    # Clear status message
                    status_msg.empty()
                    
                    # Show results table
                    st.subheader("📊 Batch Results")
                    
                    # Convert results to DataFrame
                    import pandas as pd
                    df = pd.DataFrame(results)
                    
                    # Format columns
                    df['emotion'] = df.apply(lambda row: f"{row['emoji']} {row['emotion'].capitalize()}", axis=1)
                    df['confidence'] = df['confidence'].apply(lambda x: f"{x:.1f}%")
                    df['duration'] = df['duration'].apply(lambda x: f"{x:.2f}s")
                    
                    # Drop emoji column (already merged with emotion)
                    df = df.drop(columns=['emoji'])
                    
                    # Show table
                    st.dataframe(df, use_container_width=True)
                    
                    # Show distribution of emotions
                    st.subheader("📈 Emotion Distribution")
                    
                    # Count emotions
                    emotion_counts = {}
                    for result in results:
                        emotion = result['emotion']
                        if emotion not in emotion_counts:
                            emotion_counts[emotion] = 0
                        emotion_counts[emotion] += 1
                    
                    # Create bar chart
                    emotion_labels = list(emotion_counts.keys())
                    emotion_values = list(emotion_counts.values())
                    
                    # Get colors for emotions
                    emotion_colors = []
                    for emotion in emotion_labels:
                        # Find index for emotion
                        for idx, data in EMOTION_CLASSES.items():
                            if data['name'] == emotion:
                                emotion_colors.append(data['color'])
                                break
                    
                    # Fallback if colors not found
                    if len(emotion_colors) != len(emotion_labels):
                        emotion_colors = ['#1f77b4'] * len(emotion_labels)
                    
                    # Create distribution chart
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=emotion_labels,
                        y=emotion_values,
                        marker=dict(color=emotion_colors),
                        text=emotion_values,
                        textposition='auto'
                    ))
                    
                    fig.update_layout(
                        title="Emotion Distribution in Batch",
                        xaxis=dict(title="Emotion"),
                        yaxis=dict(title="Count"),
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Error processing batch: {str(e)}")
                    st.error("Please make sure you've selected a valid model path and uploaded valid WAV files.")
    
    # App footer
    st.markdown("""
    <div style="text-align: center; margin-top: 40px; opacity: 0.7;">
        <p>Made By Armaan Mittal</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()