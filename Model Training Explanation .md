**Training Pipeline**

1. The .wav files are processed back into raw audio samples
2. The discrete-time signal is converted into a frequency-domain mel-spectrogram through a Fourier Transform
3. The mel-spectrogram is converted into decibels to normalize power values
4. The 2D CNN is trained on the 2D mel-spectograms


<img width="1056" height="521" alt="image" src="https://github.com/user-attachments/assets/1beb5c59-3257-4f95-9c5d-fa6c959530b6" />

The Python Librosa library was used to process the .wav files and extract the array of audio samples from them to recreate the buffer. The Librosa 
library was also used to create the mel-spectrogram from the audio buffer. For the conversion, I passed in parameters like the sampling rate of 16000 Hz 
64 mels (groups of samples), 512 samples per mel, and a hop length of 256 (we move 256 samples over after the Fourier Transform on each mel, so the mels will overlap).
In summary, the Short-Time Fourier Transform uses a divide-and-conquer approach. 

<img width="756" height="746" alt="image" src="https://github.com/user-attachments/assets/200ebed3-4187-43b8-8b34-a70dcbb3c253" />

The _AudioDataset_ class creates the dataset for training by loading all of the mel-spectrograms into a single PyTorch-ready tensor, and their corresponding labels (0 = green, 1 
= red, 2 = yellow). I did this by first looping through each color, then looping through the color's subfolder in the dataset folder because
this is where the .wav files are stored. I captured all the .wav files in the _files_ array. I then iterated through each file, converted the raw audio waveform 
into a frequency-spectrum mel-spectrogram showing frequency over time, and added each spectrogram to the _cached_spectrograms_ list and each color's numerical label (0-2)
to the _labels_ list. So, all of the mel-spectrograms for all 3 colors were dumped into a final dataset tensor, distinguished by their labels.

The ___getitem()___ function tells PyTorch how to extract a single training example during the training loop when it's in the DataLoader. I simply fetched the mel-spectrogram 
for the specific index it wants, and the corresponding label, and converted both of them to torch-ready tensors so that the PyTorch CNN can make predictions from them
inputs.

**CNN Architecture**

<img width="633" height="670" alt="image" src="https://github.com/user-attachments/assets/3392aaa1-0fac-4c14-b9b3-37d591f1198e" />

I split the CNN into 2 branches: the convolution layers that detect patterns/features in the mel-spectrogram, and the classifier layers (fc_layers) that output the scores for each class and predict the color.

The convolution branch has 2 2D convolution layers that use 3x3 filters to slide over the 2D mel-spectrogram and produce feature maps. These feature maps are put through a series of BatchNorm, ReLU, and
MaxPool layers to normalize, condense, and add non-linearity to the model's feature detection. 

The prediction branch acts as a classifier head. It flattens the final feature map produced by the convolution layers into a 1D tensor, so that it can be passed through a linear regression layer with 64 neurons (outputs),
and those 64 neurons are passed into a final dense layer of 3 neurons (number of classes), which each produce a logit (score) for each class (green, red, yellow).

**Training Loop**

<img width="734" height="687" alt="image" src="https://github.com/user-attachments/assets/a11e902b-bd72-4dac-af14-dc8a91ad0860" />

Here is my training loop. I first created an _AudioDataset_ object to instantiate the dataset and the CNN model. I used _CrossEntropyLoss()_ to calculate error based
on predicted probabilities and the Adam optimizer. I stored the dataset instance in a DataLoader, which holds the dataset while the training loop traverses through the examples and feeds them into the
model. Within the loop, I fetched the model's output on each mel-spectrogram, calculated the error/loss from the correct label, and updated the weights using gradient descent (_loss.backward()_
uses backpropagation to compute how much the weights should change to minimize error, and _optimizer.step()_ updates them). 
