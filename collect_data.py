import os
import serial
import numpy as np
import time
import scipy.io.wavfile as wav

SERIAL_PORT = 'COM3'
BAUD_RATE = 500000
SAMPLE_RATE = 16000
EXPECTED_BYTES = 16000 * 4  

COLORS = ["green", "red", "yellow"]  
SAMPLES_PER_COLOR = 100  

def setup_folders():
    """Creates the dataset/ directory and color subfolders automatically"""
    base_dir = "dataset"
    for color in COLORS:
        folder_path = os.path.join(base_dir, color)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            print(f"Created folder: {folder_path}")
    return base_dir

def main():
    base_dir = setup_folders()
    
    print(f"Connecting to ESP32 on {SERIAL_PORT}...")
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=3)
        ser.set_buffer_size(rx_size=128000, tx_size=128000)
    except Exception as e:
        print(f"Error opening port {SERIAL_PORT}. Is the ESP32 plugged in and the Arduino Serial Monitor closed? \nDetails: {e}")
        return
        
    time.sleep(2)  
    
    print("\n=== ESP32 Audio Dataset Collector ===")
    print("Instructions: When prompted, say the color loudly and clearly into the mic.")
    print("The ESP32 will auto-detect your voice and send the 1-second sample.")
    input("\nPress Enter to start collecting data...")

    for color in COLORS:
        print(f"\n--- Ready to collect [{color.upper()}] samples ---")
        x = input("N to Skip")
        if x.lower() == 'n':
            print(f"Skipping color: {color}")
            continue
        sample_count = 0
        while sample_count < SAMPLES_PER_COLOR:
            print(f"\n[{color.upper()}] Sample {sample_count + 1}/{SAMPLES_PER_COLOR}: Say '{color}' now...")
            
         
            ser.reset_input_buffer()
            
            
            recorded = False
            while not recorded:
                try:
                    line = ser.readline().decode('utf-8', errors='ignore').strip()
                    
                    if line == "START":
                        raw_data = ser.read(EXPECTED_BYTES)
                        
                        if len(raw_data) == EXPECTED_BYTES:
                            audio_buffer = np.frombuffer(raw_data, dtype=np.float32)
                            
                            timestamp = int(time.time() * 1000)
                            filename = f"{color}_{timestamp}.wav"
                            filepath = os.path.join(base_dir, color, filename)
                            
                            wav.write(filepath, SAMPLE_RATE, audio_buffer)
                            print(f" -> Saved: {filepath}")
                            
                            sample_count += 1
                            recorded = True
                        else:
                            print(" Warning: Got incomplete chunk from serial. Keep speaking clearly...")
                            
                except Exception as e:
                    print(f"Error during audio packet capture: {e}")
                    time.sleep(1)

    print("\n Data Collection Complete! Your folders are full and ready for training.")
    ser.close()

if __name__ == "__main__":
    main()
