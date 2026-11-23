**Data Collection Pipeline**
1. The I2S Microphone captures a raw voltage waveform when color is spoken.
2. ESP32 reads samples from the I2S mic's Analog-to-Digital Converter in 1-second windows (16000-sample buffer).
3. Each sample in the buffer is filtered out using a low-pass filter and normalized according to the max peak in the buffer.
4. The max-peak sample is checked to be high enough (>2600), meaning the window is loud enough, and a word was likely spoken.
5. If the peak is high enough, the buffer of samples is transmitted over the USB cable to my computer's USB Serial Port.
6. The raw buffer's bits are processed via a Python script listening to the serial port and saved in a .wav file for the specific color the audio buffer represents.



<img width="660" height="226" alt="image" src="https://github.com/user-attachments/assets/9f558d49-9cb9-4d85-9874-aab40d8db40e" />

I used the C++ code above to program the ESP32 to read a sample from the I2S Mic and apply a low-pass filter to it to clean out noise and low-frequency room hum, like my
air conditioning and/or fan.

<img width="511" height="161" alt="image" src="https://github.com/user-attachments/assets/b709c539-d529-4fee-b713-2f02c266ce77" />

I then added the sample to the 1-second audio buffer, and when the buffer was full (meaning a 1-second window of samples was recorded), that is when I processed the buffer (normalizing the samples, checking if the peak was high enough,
and sending it over to my computer's USB serial port).

<img width="725" height="607" alt="image" src="https://github.com/user-attachments/assets/9abcd495-b679-4607-a3ea-d8bcd8451f52" />
The process_and_inference() function finds the magnitude of the maximum peak in the buffer window, checks if it is loud enough so we can conclude a word was captured, and normalizes
all the samples to values from -1 to 1, reflecting their proportion to the max peak. If the window is loud enough, the buffer is written over the USB Serial Cable to my computer port 'COM3'
in an 8-bit integer form.

<img width="867" height="319" alt="image" src="https://github.com/user-attachments/assets/814a7e63-b0d3-4c7a-8445-083060f18391" />

I used the code above to create 3 new subfolders for each color (red, yellow, green) in the dataset folder. In each color folder, .wav files for that color containing
the audio samples will be stored.

<img width="937" height="98" alt="image" src="https://github.com/user-attachments/assets/c0beb278-eac0-4345-941f-fa61d04b8b4f" />

I initialized the serial port for my computer's specific serial port 'COM3' and set the BAUD rate to match the BAUD rate on the ESP32 IDE to ensure the script listens to the port
at the same speed that data arrives.

<img width="1234" height="281" alt="image" src="https://github.com/user-attachments/assets/b2be4562-84c1-469d-a0fe-e393d6a6d2ef" />
<img width="979" height="607" alt="image" src="https://github.com/user-attachments/assets/223014d4-dc83-4be2-a1ab-1ab7daf97986" />


The snippet above loops through each color, and collects samples one by one. The _readline()_ method is used to extract the string at the serial port. If "START" is fetched at the port,
that indicates the buffer is ready. The _read()_ method allows the serial to fetch the data in raw bits, and the data is converted into a NumPy array of 32 bit floats using 
the _frombuffer()_ method. After building the file path (by combining the "dataset", the color, and the timestamp, _wav.write()_ is used to store the audio clip as a playable file
to be used for training the CNN.
