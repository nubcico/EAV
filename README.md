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

The raw dataset, along with the pre-extracted features, can be accessed and downloaded from Zenodo.

### Program Execution Process

After downloading the dataset, you must choose between utilizing the raw dataset or the pre-extracted features, as this decision will determine your subsequent steps.
If you opt for the raw dataset, only a minor modification is required in the `Dataload_Audio.py` file: adjust the `parent_directory` parameter in the `DataLoadAudio` class to the directory location of the "EAV" folder on your system. Using the raw dataset enables customization of your training and testing data split ratio through the EAVDataSplit class. In our case, we employed a 70/30 split, calculated as `h_idx = 56`. If `x` is your desired training dataset percentage (e.g., x = 70), `h_idx` can be calculated using the formula `h_idx = (x * 80) / 100`.
If you decide to work with the pre-extracted features, you need to modify the code as follows: comment out the lines currently used for the raw dataset before `aud_list.get_split`, then uncomment the section for the pre-extracted features. Additionally, set the `direct` variable to point to the path containing the "Audio" directory on your system.

```sh
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
