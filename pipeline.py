import os
import glob
import random
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchaudio
import librosa
from tqdm import tqdm

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


RAVDESS_EMOTIONS = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

TESS_EMOTIONS = {
    'angry': 'angry',
    'disgust': 'disgust',
    'fear': 'fearful',
    'happy': 'happy',
    'neutral': 'neutral',
    'ps': 'surprised', 
    'sad': 'sad'
}

EMOTION_CLASSES = {
    'neutral': 0,
    'calm': 1,
    'happy': 2,
    'sad': 3,
    'angry': 4,
    'fearful': 5,
    'disgust': 6,
    'surprised': 7
}

class SpeechEmotionDataset(Dataset):
    def __init__(self, data_path, sample_rate=16000, n_mels=64, transform=None):
        """
        Args:
            data_path (str): Path to the dataset directory
            sample_rate (int): Target sample rate for audio
            n_mels (int): Number of mel bands to generate
            transform (callable, optional): Optional transform to be applied on a sample
        """
        self.data_path = data_path
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.transform = transform
        self.audio_files = []
        self.labels = []
        
        self._load_dataset_processed()
        
        if len(self.audio_files) == 0:
            print("ERROR: No audio files were found in the specified directory.")
            print(f"Searched in: {self.data_path}")
            print("Please check that this directory exists and contains .wav files.")
            return
            
        self.mel_spec_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=1024,
            hop_length=512,
            n_mels=n_mels
        )
        
        print(f"Dataset loaded with {len(self.audio_files)} samples")
        
    def _load_dataset_processed(self):
        """Load audio files from processed directory"""
        print(f"Scanning for audio files in: {self.data_path}")
        
        emotion_mapping = {
            'angry': 'angry',
            'disgust': 'disgust',
            'fear': 'fearful',
            'happy': 'happy',
            'neutral': 'neutral',
            'ps': 'surprised',  
            'sad': 'sad',
            'calm': 'calm'
        }
        
        emotion_counts = {}
        
        for root, _, files in os.walk(self.data_path):
            for file in files:
                if file.endswith('.wav'):
                    file_path = os.path.join(root, file)
                    
                    emotion = None
                    
                    folder_name = os.path.basename(root).lower()
                    
                    for emotion_key in emotion_mapping:
                        if emotion_key in folder_name:
                            emotion = emotion_mapping[emotion_key]
                            break
                    
                    if not emotion:
                        file_lower = file.lower()
                        for emotion_key in emotion_mapping:
                            if emotion_key in file_lower:
                                emotion = emotion_mapping[emotion_key]
                                break
                                
                    if emotion and emotion in EMOTION_CLASSES:
                        self.audio_files.append(file_path)
                        self.labels.append(EMOTION_CLASSES[emotion])
                        
                        if emotion not in emotion_counts:
                            emotion_counts[emotion] = 0
                        emotion_counts[emotion] += 1
        
        print("\nEmotion distribution in dataset:")
        for emotion, count in emotion_counts.items():
            print(f"  {emotion}: {count} files")
    
    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, idx):
        try:
            audio_file = self.audio_files[idx]
            label = self.labels[idx]
            
            # Load audio using librosa
            audio, _ = librosa.load(audio_file, sr=self.sample_rate, mono=True)
            
            # Convert to torch tensor
            audio_tensor = torch.FloatTensor(audio)
            
            # Compute mel spectrogram
            mel_spec = self.mel_spec_transform(audio_tensor)
            
            # Apply log transformation to mel spectrogram (adding small constant to avoid log(0))
            mel_spec = torch.log(mel_spec + 1e-9)
            
            # Add channel dimension [1, n_mels, time]
            mel_spec = mel_spec.unsqueeze(0)
            
            if self.transform:
                mel_spec = self.transform(mel_spec)
            
            return {
                'mel_spec': mel_spec,
                'label': label,
                'path': audio_file
            }
            
        except Exception as e:
            print(f"Error loading file {self.audio_files[idx]}: {str(e)}")
            # Return a default tensor with proper shape in case of error
            # This avoids crashing the dataloader
            default_mel_spec = torch.zeros((1, self.n_mels, 100))  # Default time dimension of 100
            return {
                'mel_spec': default_mel_spec,
                'label': 0,  # Default to neutral class
                'path': self.audio_files[idx] if idx < len(self.audio_files) else "error"
            }

# Custom collate function for variable length sequences
def collate_fn(batch):
    """
    Custom collate function to handle variable length sequences by padding to max length in batch
    """
    # Filter out any None values or corrupted items
    batch = [item for item in batch if item is not None and 'mel_spec' in item and item['mel_spec'] is not None]
    
    # Check if batch is empty after filtering
    if len(batch) == 0:
        # Return empty tensors with appropriate dimensions to avoid errors
        return {
            'mel_spec': torch.zeros((0, 1, 64, 100)), # [batch, channel, mel_bands, time]
            'label': torch.tensor([], dtype=torch.long),
            'path': []
        }
    
    # Get max sequence length in batch
    max_len = max([item['mel_spec'].shape[2] for item in batch])
    
    # Pad mel spectrograms to max length
    mel_specs = []
    labels = []
    paths = []
    
    for item in batch:
        mel_spec = item['mel_spec']
        padded_mel_spec = F.pad(mel_spec, (0, max_len - mel_spec.shape[2]), mode='constant', value=0)
        mel_specs.append(padded_mel_spec)
        labels.append(item['label'])
        paths.append(item['path'])
    
    # Stack all tensors
    mel_specs = torch.stack(mel_specs)
    labels = torch.tensor(labels, dtype=torch.long)
    
    return {
        'mel_spec': mel_specs,
        'label': labels,
        'path': paths
    }

class SpeechEmotionCNN(nn.Module):
    def __init__(self, n_mels=64, num_classes=8):
        super(SpeechEmotionCNN, self).__init__()
        
        # Define CNN layers
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
        
        # Adaptive pooling to handle variable sequence length
        self.adaptive_pool = nn.AdaptiveAvgPool2d((2, 2))
        
        # Calculate features after adaptive pooling
        self.fc_input_size = 512 * 2 * 2
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.fc_input_size, 512)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, num_classes)
        
    def forward(self, x):
        # CNN layers
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        
        # Adaptive pooling to handle variable sequence length
        x = self.adaptive_pool(x)
        
        # Flatten
        x = x.view(-1, self.fc_input_size)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        
        return x

def train_epoch(model, dataloader, optimizer, criterion, device, scaler, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Check if dataloader is empty
    if len(dataloader) == 0:
        print(f"Warning: Training dataloader is empty for epoch {epoch+1}")
        return 0.0, 0.0
    
    progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1} [Train]')
    
    for batch in progress_bar:
        mel_specs = batch['mel_spec'].to(device)
        labels = batch['label'].to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass with autocast for mixed precision
        with autocast():
            outputs = model(mel_specs)
            loss = criterion(outputs, labels)
        
        # Backward and optimize with gradient scaling
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': f'{running_loss / (progress_bar.n + 1):.4f}',
            'acc': f'{100 * correct / total:.2f}%' if total > 0 else '0.00%'
        })
    
    epoch_loss = running_loss / len(dataloader) if len(dataloader) > 0 else 0.0
    epoch_acc = 100 * correct / total if total > 0 else 0.0
    
    return epoch_loss, epoch_acc

def validate_epoch(model, dataloader, criterion, device, epoch):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Check if dataloader is empty
    if len(dataloader) == 0:
        print(f"Warning: Validation dataloader is empty for epoch {epoch+1}")
        return 0.0, 0.0
    
    progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1} [Valid]')
    
    with torch.no_grad():
        for batch in progress_bar:
            mel_specs = batch['mel_spec'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass
            outputs = model(mel_specs)
            loss = criterion(outputs, labels)
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{running_loss / (progress_bar.n + 1):.4f}',
                'acc': f'{100 * correct / total:.2f}%' if total > 0 else '0.00%'
            })
    
    epoch_loss = running_loss / len(dataloader) if len(dataloader) > 0 else 0.0
    epoch_acc = 100 * correct / total if total > 0 else 0.0
    
    return epoch_loss, epoch_acc

def main():
    # Config
    data_path = r"E:\emotion classificaton\Speech Project\datasets\processed"
    batch_size = 64
    num_workers = 4
    learning_rate = 1e-3
    weight_decay = 1e-5
    num_epochs = 30
    valid_split = 0.2
    n_mels = 64
    sample_rate = 16000
    num_classes = len(EMOTION_CLASSES)
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create dataset
    dataset = SpeechEmotionDataset(
        data_path=data_path,
        sample_rate=sample_rate,
        n_mels=n_mels
    )
    
    # Check if dataset is empty
    if len(dataset) == 0:
        print("Error: Dataset is empty. Please check the dataset path and structure.")
        return
    
    # Split dataset into train and validation sets
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    np.random.shuffle(indices)
    
    split_idx = int(np.floor(valid_split * dataset_size))
    train_indices, valid_indices = indices[split_idx:], indices[:split_idx]
    
    # Check if train and validation indices are not empty
    if len(train_indices) == 0:
        print("Error: No training samples available. Please check your dataset.")
        return
    
    if len(valid_indices) == 0:
        print("Warning: No validation samples available. Using a small portion of training data for validation.")
        # Use a small portion of training data for validation if none available
        split_idx = int(np.floor(0.1 * len(train_indices)))
        valid_indices = train_indices[:split_idx]
        train_indices = train_indices[split_idx:]
    
    # Create data loaders
    train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
    valid_sampler = torch.utils.data.SubsetRandomSampler(valid_indices)
    
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    valid_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=valid_sampler,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    print(f"Training samples: {len(train_indices)}")
    print(f"Validation samples: {len(valid_indices)}")
    
    # Create model
    model = SpeechEmotionCNN(n_mels=n_mels, num_classes=num_classes)
    model = model.to(device)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Learning rate scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # Gradient scaler for mixed precision
    scaler = GradScaler()
    
    # Training loop
    best_valid_acc = 0.0
    
    for epoch in range(num_epochs):
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device, scaler, epoch
        )
        
        # Validate
        valid_loss, valid_acc = validate_epoch(
            model, valid_loader, criterion, device, epoch
        )
        
        # Learning rate scheduling
        scheduler.step()
        
        # Print stats
        print(f"Epoch: {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Valid Loss: {valid_loss:.4f}, Valid Acc: {valid_acc:.2f}%")
        print("-" * 50)
        
        # Save best model
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'train_loss': train_loss,
                'train_acc': train_acc,
                'valid_loss': valid_loss,
                'valid_acc': valid_acc,
            }, 'best_speech_emotion_model.pth')
            print(f"Model saved with validation accuracy: {valid_acc:.2f}%")
    
    print(f"Training completed. Best validation accuracy: {best_valid_acc:.2f}%")

if __name__ == "__main__":
    main()