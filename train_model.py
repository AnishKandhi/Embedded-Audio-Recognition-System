import os 
import numpy as np 
import librosa 
import torch 
import torch.nn as nn 
from torch.utils.data import Dataset, DataLoader 
from model import ColorAudioCNN 

DATASET_DIR = "dataset" 
COLORS = ["green", "red", "yellow"]  
SAMPLE_RATE = 16000 
BATCH_SIZE = 8 
EPOCHS = 50
LEARNING_RATE = 0.0005

def audio_to_spectrogram(file_path): 
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE) 
    if len(y) < SAMPLE_RATE: 
        y = np.pad(y, (0, SAMPLE_RATE - len(y))) 
    else: 
        y = y[:SAMPLE_RATE] 
    mel_spec = librosa.feature.melspectrogram(y=y, sr=SAMPLE_RATE, n_fft=512, hop_length=256, 
                                              n_mels=64, fmax = SAMPLE_RATE // 2) 
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max) 
    return mel_spec_db 

class AudioDataset(Dataset): 
    def __init__(self, base_dir, colors): 
        self.cached_spectrograms = [] 
        self.labels = [] 
        self.color_to_idx = {color: idx for idx, color in enumerate(colors)} 
        
        print("Pre-computing Mel Spectrograms into RAM...")
        for color in colors: 
            color_folder = os.path.join(base_dir, color) 
            if not os.path.isdir(color_folder): 
                continue 
            
            files = [f for f in os.listdir(color_folder) if f.endswith('.wav')]
            print(f" -> Processing {len(files)} files for color: '{color}'")
            
            for fname in files: 
                file_path = os.path.join(color_folder, fname)
                spec = audio_to_spectrogram(file_path)
                
                self.cached_spectrograms.append(spec)
                self.labels.append(self.color_to_idx[color]) 

    def __len__(self): 
        return len(self.labels) 

    def __getitem__(self, idx): 
        spec = self.cached_spectrograms[idx]
        tensor_spec = torch.tensor(spec, dtype=torch.float32).unsqueeze(0) 
        label = torch.tensor(self.labels[idx], dtype=torch.long) 
        return tensor_spec, label 

def main(): 
    dataset = AudioDataset(DATASET_DIR, COLORS) 
    if len(dataset) == 0: 
        print("Error: No training .wav files found! Run your collection script first.") 
        return 
        
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True) 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    model = ColorAudioCNN(num_classes=len(COLORS)).to(device) 
    criterion = nn.CrossEntropyLoss() 
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE) 
    
    print(f"\nTraining network on {device} for {EPOCHS} epochs...") 
    
    for epoch in range(EPOCHS): 
        model.train()
        running_loss = 0.0 
        correct = 0 
        total = 0 
        
        for inputs, labels in dataloader: 
            inputs, labels = inputs.to(device), labels.to(device) 
            optimizer.zero_grad() 
            outputs = model(inputs) 
            loss = criterion(outputs, labels) 
            loss.backward() 
            optimizer.step() 
            
            running_loss += loss.item() * inputs.size(0) 
            _, predicted = torch.max(outputs, 1) 
            total += labels.size(0) 
            correct += (predicted == labels).sum().item() 
            
        epoch_loss = running_loss / len(dataset) 
        epoch_acc = (correct / total) * 100 
        print(f"Epoch {epoch+1:02d}/{EPOCHS} -> Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.2f}%") 
        
    torch.save(model.state_dict(), "color_model.pth") 
    print("\nSaved trained weights to 'color_model.pth' successfully!") 

if __name__ == "__main__": 
    main()