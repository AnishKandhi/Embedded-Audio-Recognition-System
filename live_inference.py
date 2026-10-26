import serial
import numpy as np
import librosa
import torch
import os
from model import ColorAudioCNN
import time
# Config
SERIAL_PORT = 'COM3'
BAUD_RATE = 500000
SAMPLE_RATE = 16000
EXPECTED_BYTES = 16000 * 4 
COLORS = ["Green", "Red", "Yellow"]

def process_live_audio(audio_buffer):
    """Performs the exact same feature mapping transformation applied during training"""
    mel_spec = librosa.feature.melspectrogram(y=audio_buffer, sr=SAMPLE_RATE, n_fft=512, hop_length=256, n_mels=64)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    tensor = torch.tensor(mel_spec_db, dtype=torch.float32)
    return tensor.unsqueeze(0).unsqueeze(0)

def main():
    if not os.path.exists("color_model.pth"):
        print("Error: 'color_model.pth' not found. Run train_model.py first!")
        return

    model = ColorAudioCNN(num_classes=len(COLORS))
    model.load_state_dict(torch.load("color_model.pth", map_location=torch.device('cpu')))
    model.eval()
    
    print(f"Connecting to ESP32 on port {SERIAL_PORT}...")
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=3)
    ser.set_buffer_size(rx_size=128000, tx_size=128000)

    ser.reset_input_buffer()
    print("\n Network loaded! Say Green, Red, or Yellow into the mic...")

    while True:
        try:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            
            if line == "START":
                raw_data = ser.read(EXPECTED_BYTES)
                
                if len(raw_data) == EXPECTED_BYTES:
                    audio_buffer = np.frombuffer(raw_data, dtype=np.float32)
                    
                    input_tensor = process_live_audio(audio_buffer)
                    with torch.no_grad():
                        outputs = model(input_tensor)
                        probabilities = torch.softmax(outputs, dim=1)
                        prediction = torch.argmax(probabilities, dim=1).item()
                        confidence = probabilities[0][prediction].item() * 100
                        color = COLORS[prediction].lower()
                        ser.write(f"{color}\n".encode('utf-8'))
                        ser.flush()
                        ser.reset_input_buffer()
                    print(f" Detected Color: ** {COLORS[prediction].upper()} ** ({confidence:.1f}% confidence)")

                else:
                    print("Dropped buffer frame (incomplete byte packet received).")
                    
        except KeyboardInterrupt:
            print("\nExiting live viewer...")
            break
        except Exception as e:
            print(f"Runtime connection glitch: {e}")
            
    ser.close()

if __name__ == "__main__":
    main()
