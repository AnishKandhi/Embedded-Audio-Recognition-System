import numpy as np
import json
import os

def main():
    print("Generating representative feature set for Edge Impulse profiling...")
    
    num_samples = 20
    mel_bands = 64
    time_steps = 63
    
    
    features_list = []
    
    for _ in range(num_samples):
        mock_spectrogram = np.random.uniform(low=-80.0, high=0.0, size=(mel_bands, time_steps))
        
        flattened_features = mock_spectrogram.flatten().tolist()
        features_list.append(flattened_features)

    output_data = {
        "features": features_list
    }
    
    output_filename = "representative_features.json"
    with open(output_filename, 'w') as f:
        json.dump(output_data, f)
        
    print(f"\nSUCCESS! File created: {output_filename}")
    print(f"Contains {num_samples} samples. Each sample contains {mel_bands}x{time_steps} ({mel_bands*time_steps}) features.")
    print("Upload this file directly into the Edge Impulse validation prompt.")

if __name__ == "__main__":
    main()
