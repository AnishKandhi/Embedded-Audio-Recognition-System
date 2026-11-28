# Embedded-AI-Audio-Recognition-System (Voice-Controlled Smart Deadbolt)

DEMO: https://drive.google.com/file/d/1e3vEYJ5D1QH07Ml6g5Neii-dZMY4p2-y/view?usp=sharing

OFFLINE INFERENCE VIDEO: https://drive.google.com/file/d/1vzCh-jGWAWgMOsjL-li9YbEJLhG22hUP/view?usp=drive_link


My project built a **voice-controlled smart lock** based on an **embedded audio recognition system**. I trained an AI model (**2D CNN**) on manually collected audio samples to detect colors spoken into an **I2S Microphone**. I collected around 250 samples for each color (**red, yellow, green**) by manually speaking each color into the mic in various pitches and tones to enhance the model's pattern recognition. I programmed an **ESP32** to read the specific sound samples from the I2S Mic in one-second windows and sent the second full audio buffer over a **USB Serial Cable** to my computer's **USB serial port**. On my computer, I wrote a **Python** script to process this audio buffer into a **.wav** file when collecting data, separating each buffer by class (color). For training, I had it process the audio sample into a 2D **mel-spectogram** by running **Fourier Transform** on it so that it was in the frequency domain. 

**Data Collection Pipeline**
1. The I2S Microphone captures a raw voltage waveform when color is spoken.
2. ESP32 reads samples from the I2S mic's Analog-to-Digital Converter in 1-second windows (16000-sample buffer).
3. Each sample in the buffer is filtered out using a low-pass filter and normalized according to the max peak in the buffer.
4. The max-peak sample is checked to be high enough (>2600), meaning the window is loud enough, and a word was likely spoken.
5. If the peak is high enough, the buffer of samples is transmitted over the USB cable to my computer's USB Serial Port.
6. The raw buffer's bits are processed via a Python script listening to the serial port and saved in a .wav file for the specific color the audio buffer represents.

**Training Pipeline**
1. The .wav files are processed back into raw audio samples
2. The discrete-time signal is converted into a frequency-domain mel-spectrogram through a Fourier Transform
3. The mel-spectrogram is converted into decibels to normalize power values
4. The 2D CNN is trained on the 2D mel-spectrograms

**Deployment Pipeline**
1. Convert the CNN model file (architecture only) to an ONNX file so it can be deployed onto an ESP32 via TensorFlow Lite & Edge Impulse.
2. Set the model's parameters for deployment, such as the input shape for mel-spectrograms.
3. Quantize the model's parameters to 8 bit (int8) via the EON compiler so it can fit on the small ESP32, and benchmark accuracy, SRAM usage, and inference speed.
4. Link the C++ TFLite library (with all the weights) to the firmware code so the model can run inference offline.

After training and retraining the model with different samples (until it achieved **95%** accuracy in validation and performed well in live inference), I exported the file to a **.onnx** and used Edge Impulse to quantize the model's weights to **8-bit** (so it can fit on the small ESP32) and deploy it onto the ESP32 to run inference on the controller. Accuracy degraded slightly by **4** percentage points to **91%** (as expected, the weights aren't as precise), but SRAM use and inference speed improved by **25%** from the 32-bit model. 

Since the model now lived on the ESP32, I wrote the Fourier Transform C++ code directly in the firmware to process the audio samples picked up by the I2S Mic, because my model did not have access to the Python Libraries on my computer anymore (since it runs inference on the ESP32). 

Finally, after the model was working reasonably on the ESP32 and lighting LEDs correctly based on the colors said, I coded the ESP32 to rotate a servo depending on which color was said. "Red" was meant to close the lock, so I had it rotate all the way to its left (0 degrees). Green was meant to open it, so I coded a 180-degree rotation. Yellow was a partial lock, so a 90-degree rotation was ideal. The servo exerted torque on a paperclip, and the paperclip was attached to a sliding cardboard deadbolt I made for linear, low-friction movement.

**Read the .md files in this REPO for code explanations and learn about obstacles I faced.**

_-Anish Kandhi_
