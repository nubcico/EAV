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

Run demo:
   ```sh
   python demo.py
   ```

## Training

To train the Transfromer for Speech emotion recognition run:
   ```sh
   python Dataload_audio.py
   ```

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