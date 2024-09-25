<!-- ABOUT THE PROJECT -->
# EAV: EEG-Audio-Video Dataset for Emotion Recognition in Conversational Contexts 

We introduce a multimodal emotion dataset comprising data from 30-channel electroencephalography
(EEG), audio, and video recordings from 42 participants. Each participant engaged in a cue-based conversation scenario,
eliciting five distinct emotions: neutral(N), anger(A), happiness(H), sadness(S), and calmness(C). 

Participants engage in paired listen/speak sets with recordings of an experienced actor, seated in front of a 27-inch monitor displaying visual stimuli. 
The experiment is designed as a pseudo-random class-iteration sequence: [A, A, C, C, S, S, H, A, C, H, H, S, S, A, A, C, C, H, H, S]. 
Throughout the experiment, each participant contributed 200 interactions. This resulted in a cumulative total of 8,400 interactions across all participants.
Please refer to the paper [TODO: add link] for more details.
## Domains

### Video
Each 'Video' subfolder contains 200 video clips, each is 20 sec in lengths, 30 fps and performs either ’listening’ or ’speaking’ tasks. 
The Video data adopts the structure - [5 emotion classes × 2 tasks × 20 iterations]

File format: .mp4

Baseline performance of DeepFace: Mean ACC = 52.8 %, Mean F1-score = 51.5 %

### Audio
Each 'Audio' subfolder contains 100 audio files, each is 20 sec in lengths and performs only ’speaking’ task. 
The audio data adopts the structure - [5 classes × 1 task ('speaking') × 20 conversations]

File format: .wav

Baseline performance of SCNN: Mean ACC = 36.7 %, Mean F1-score = 34.1 %
### EEG
Each 'EEG' subfolder contains 2 EEG data files. Each instance is 20 sec in lengths and an initial sampling rate of 500 Hz. Due to continuous recording, 
the processed EEG data adopts the structure  - [200 instances × 10,000 time points(20s × 500 Hz) × 30 channels].
The labels for this data use a one-hot encoding format, structured as 200 trials by 10 classes (5 emotions multiplied by 2
tasks).

File format: .mat

Baseline performance of EEGnet: Mean ACC = 36.7 %, Mean F1-score = 34.1 %

_Note that the label information can be applied across all modalities since all recordings, regardless of the modality, were
conducted synchronously. This ensures uniform annotations throughout the dataset._

<!-- GETTING STARTED -->
## Getting Started


* conda environment
  ```sh
  conda create --name eav python=3.10
  conda activate eav
  ```

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/nubcico/EAV.git
   cd EAV
   ```
2. Install requirements
   ```sh
   pip install -r requirements.txt
   ```

<!-- USAGE EXAMPLES -->
## Usage

### Dataset

The raw dataset, along with the pre-extracted features, can be accessed and downloaded from [Zenodo](https://doi.org/10.5281/zenodo.10205702).

### Program Execution Process

#### Selecting the Dataset Type

After downloading the dataset, you must choose between utilizing the raw dataset or the pre-extracted features, as this decision will determine your subsequent steps.

If you opt for the raw dataset, only a minor modification is required in the `Dataload_Audio.py` file: adjust the `parent_directory` parameter in the `DataLoadAudio` class to the directory location of the "EAV" folder on your system. 

Using the raw dataset enables customization of your training and testing data split ratio through the EAVDataSplit class. In our case, we employed a 70/30 split, calculated as `h_idx = 56`. If `x` is your desired training dataset percentage (e.g., x = 70), `h_idx` can be calculated using the formula `h_idx = (x * 80) / 100`.

If you decide to work with the pre-extracted features, you need to modify the code as follows: comment out the lines currently used for the raw dataset before `aud_list.get_split`, then uncomment the section for the pre-extracted features. Additionally, set the `direct` variable to point to the path containing the "Audio" directory on your system.

```python
aud_loader = DataLoadAudio(subject=sub, parent_directory=r'D:\EAV')
[data_aud , data_aud_y] = aud_loader.process()
aud_list = EAVDataSplit(data_aud, data_aud_y)
[tr_x_aud, tr_y_aud, te_x_aud , te_y_aud] = aud_list.get_split(h_idx=56)
        
# direct=r"D:\EAV\Inputs\Audio"
# file_name = f"subject_{sub:02d}_aud.pkl"
# file_ = os.path.join(direct, file_name)

# with open(file_, 'rb') as f:
#     aud_list = pickle.load(f) 
# tr_x_aud, tr_y_aud, te_x_aud , te_y_aud = aud_list    
            
data = [tr_x_aud, tr_y_aud, te_x_aud , te_y_aud]
```

The same adjustments should be applied for each modality.

#### Selecting the classification model

For the classification of the audio modality, we employ the Audio Spectrogram Transformer (AST) model pretrained on the AudioSet dataset, which we will subsequently fine-tune on our specific dataset, as implemented in the 'Dataload_audio.py' and 'Transformer_torch/Transformer_Audio.py' files.

```python
from Transformer_torch import Transformer_Audio
...
mod_path = os.path.join(os.getcwd(), 'ast-finetuned-audioset')
Trainer = Transformer_Audio.AudioModelTrainer(data, model_path=mod_path, sub =f"subject_{sub:02d}",
                                                      num_classes=5, weight_decay=1e-5, lr=0.005, batch_size = 8)
Trainer.train(epochs=10, lr=5e-4, freeze=True)
Trainer.train(epochs=15, lr=5e-6, freeze=False)
test_acc.append(Trainer.outputs_test)
```
The 'AudioModelTrainer' class is designed to train and fine-tune this model effectively. It leverages PyTorch and the Hugging Face Transformers library to adapt the AST model for the emotion classification task. 

```python
from transformers import AutoModelForAudioClassification
...
class AudioModelTrainer:
    def __init__(self, DATA, model_path, sub='', num_classes=5, weight_decay=1e-5, lr=0.001, batch_size=128):
        ...
        self.model = AutoModelForAudioClassification.from_pretrained(model_path)
```

For the video and EEG modalities, the framework allows the choice between Transformer-based and CNN-based models. Specifically, for video, we utilize the Vision Transformer model, which is pretrained on the facial_emotions_image_detection dataset. The following example from the 'Dataload_vision.py' file illustrates both options: 

```python
# Transformer for Vision
from Transformer_torch import Transformer_Vision

mod_path = os.path.join('C:\\Users\\minho.lee\\Dropbox\\Projects\\EAV', 'facial_emotions_image_detection')
trainer = Transformer_Vision.ImageClassifierTrainer(data,
                                                    model_path=mod_path, sub=f"subject_{sub:02d}",
                                                    num_labels=5, lr=5e-5, batch_size=128)
trainer.train(epochs=10, lr=5e-4, freeze=True)
trainer.train(epochs=5, lr=5e-6, freeze=False)
trainer.outputs_test
```

Alternatively, the CNN-based model can be utilized as follows:
```python
# CNN for Vision
from CNN_torch.CNN_Vision import ImageClassifierTrainer
trainer = ImageClassifierTrainer(data, num_labels=5, lr=5e-5, batch_size=32)
trainer.train(epochs=3, lr=5e-4, freeze=True)
trainer.train(epochs=3, lr=5e-6, freeze=False)
trainer._delete_dataloader()
trainer.outputs_test
```
The same approach can be applied for the EEG modality, providing flexibility in choosing between Transformer and CNN architectures based on the requirements of the task.

To execute the program via the command line, run the following commands for each modality respectively:

   ```sh
   python Dataload_audio.py
   ```

Of course! Here’s the revised version without bold text and using numbers:

### Data Preprocessing

The data preprocessing for audio, EEG, and video modalities is designed to prepare the raw data for emotion classification. Each modality follows its own distinct workflow:

1. Audio Modality:  
   The `DataLoadAudio` class handles audio data processing:
   - Data File Loading: The `data_files()` method retrieves audio file paths and their corresponding emotion labels.
   - Feature Extraction: The `feature_extraction()` method loads the audio files, resamples them to a target rate, and segments the audio into 5-second clips. Each segment is labeled according to the associated emotion.
   - Label Encoding: Emotion labels are converted to numerical indices for model compatibility.
   - Processing Coordination: The `process()` method orchestrates these steps, returning the extracted features and labels.

   ```python
   class DataLoadAudio:
       def process(self):
           self.data_files()  # Load audio file paths and labels
           self.feature_extraction()  # Extract audio features
           return self.feature, self.label_indexes  # Return features and labels
   ```

2. EEG Modality:  
   The `DataLoadEEG` class manages EEG data processing:
   - Data File Loading: The `data_mat()` method loads EEG data and labels from MAT files.
   - Downsampling: The `downsampling()` method reduces the sampling frequency to a target rate.
   - Bandpass Filtering: The `bandpass()` method applies a bandpass filter to retain frequencies of interest.
   - Segmentation: The `data_div()` method divides the data into smaller segments for analysis.
   - Data Preparation: The `data_prepare()` method coordinates the above steps, returning the processed EEG segments and labels.

   ```python
   class DataLoadEEG:
       def data_prepare(self):
           self.data_mat()  # Load EEG data and labels
           self.downsampling()  # Downsample the data
           self.bandpass()  # Apply bandpass filtering
           self.data_div()  # Divide the data into segments
           return self.seg_f_div, self.label_div  # Return filtered segments and labels
   ```

3. Video Modality:  
   The `DataLoadVision` class processes video data:
   - Data File Loading: The `data_files()` method gathers video file paths and emotion labels.
   - Face Detection and Frame Extraction: The `data_load()` method captures frames from each video, using face detection to align faces if enabled. It collects segments of 25 frames, each representing a 5-second interval.
   - Label Encoding: Emotion labels are converted to numerical indices for consistency.
   - Data Preparation: The `process()` method manages these tasks, returning the segmented images and their corresponding labels.

   ```python
   class DataLoadVision:
       def process(self):
           self.data_files()  # Load video file paths and labels
           self.data_load()  # Extract and process frames from videos
           return self.images, self.image_label_idx  # Return processed image segments and labels
   ```

This structured approach ensures that each modality's data is appropriately preprocessed, facilitating effective training and evaluation of the emotion classification model.

<!-- ROADMAP -->
## Roadmap

- [x] CNN based Emotion recognition on Video, Audio and EEG domains using Tensorflow
- [x] CNN based Emotion recognition on Video and EEG domains using PyTorch
- [x] Transformer based Emotion recognition on Video, Audio and EEG domains using PyTorch
- [ ] Create demo file 
- [ ] Add .pkl files of preprocessed video data (Feature_vision folder)
- [ ] Add inference files


<!-- CONTACT -->
## Contact

Minho Lee - minho.lee@nu.edu.kz

Adai Shomanov - adai.shomanov@nu.edu.kz

Zhuldyz Kabidenova - zhuldyz.kabidenova@nu.edu.kz

Adnan Yazici - adnan.yazici@nu.edu.kz


<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.




<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->

[product-screenshot]: images/EAVlogo.png
