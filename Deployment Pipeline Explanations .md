**Deployment Pipeline**
1. Convert the CNN model file (architecture only) to an ONNX file so it can be deployed onto an ESP32 via TensorFlow Lite & Edge Impulse.
2. Set the model's parameters for deployment, such as the input shape for mel-spectrograms.
3. Quantize the model's parameters to 8 bit (int8) via the EON compiler so it can fit on the small ESP32, and benchmark accuracy, SRAM usage, and inference speed.
4. Link the C++ TFLite library (with all the weights) to the firmware code so the model can run inference offline.

<img width="707" height="522" alt="image" src="https://github.com/user-attachments/assets/779413ee-a56a-4989-9e09-4cfda42f0123" />

I created a Python script that uploads and saves the trained model as an .onnx file. 

<img width="712" height="87" alt="image" src="https://github.com/user-attachments/assets/4e693151-3562-48b4-8665-f6e3779cbbf9" />

The snippet above creates the CNN and loads its trained weights saved in the color_model.pth file into the model.

<img width="496" height="344" alt="image" src="https://github.com/user-attachments/assets/dce98393-ed25-448e-a38c-aca13e44502e" />

I created a dummy input with the same shape as my input mel-spectrograms (1, 1, 64, 63) to calibrate my model to the input's shape before making predictions. The ONNX file I am storing the
model in is color_voice_lock.onnx. torch.onnx.export loads my model into that ONNX file alongside the calibration input data. 
<img width="185" height="19" alt="image" src="https://github.com/user-attachments/assets/61caabcb-f9e4-4b20-a3ec-e21392a6e486" />


After uploading the CNN to Edge Impulse via the Bring Your Own Model (BYOM) pipeline, I benchmarked the quantized model's metrics.

**Quantized (Int8) Model:**

<img width="637" height="669" alt="image" src="https://github.com/user-attachments/assets/4f1873a8-43a1-4768-88f0-5cd5ead10d20" />




**Baseline Int32 Model:**

<img width="806" height="352" alt="image" src="https://github.com/user-attachments/assets/89c19a18-b42c-4d98-85d7-4b76c2ae1afa" />

Inference speed increased by **17%**, Peak RAM usage reduced by **25%**, and flash memory usage improved by **25%**. Though memory usage and efficiency improved significantly, accuracy dropped from **95%** to **91%**.
since 8 bit integers lose precision compared to 32 bit, which is not that bad.

<img width="540" height="31" alt="image" src="https://github.com/user-attachments/assets/79b18685-532c-4aab-9365-6700cd3707de" />

The header line above links my TFLite Edge Impulse C++ library that contains my weights to the firmware so the model can use the same learned weights to run inference on the controller.
