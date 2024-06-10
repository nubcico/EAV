[//]: # (<!-- PROJECT LOGO -->)

[//]: # (<br />)

[//]: # (<div align="center">)

[//]: # (  <a href="https://github.com/othneildrew/Best-README-Template">)

[//]: # (    <img src="images/logo.png" alt="Logo" width="80" height="80">)

[//]: # (  </a>)

[//]: # ()
[//]: # (  <h3 align="center">Best-README-Template</h3>)

[//]: # ()
[//]: # (  <p align="center">)

[//]: # (    An awesome README template to jumpstart your projects!)

[//]: # (    <br />)

[//]: # (    <a href="https://github.com/othneildrew/Best-README-Template"><strong>Explore the docs »</strong></a>)

[//]: # (    <br />)

[//]: # (    <br />)

[//]: # (    <a href="https://github.com/othneildrew/Best-README-Template">View Demo</a>)

[//]: # (    ·)

[//]: # (    <a href="https://github.com/othneildrew/Best-README-Template/issues/new?labels=bug&template=bug-report---.md">Report Bug</a>)

[//]: # (    ·)

[//]: # (    <a href="https://github.com/othneildrew/Best-README-Template/issues/new?labels=enhancement&template=feature-request---.md">Request Feature</a>)

[//]: # (  </p>)

[//]: # (</div>)



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## EAV: EEG-Audio-Video Dataset for Emotion Recognition in Conversational Contexts 

[![EAV logo][product-screenshot]](https://example.com)

We introduce a multimodal emotion dataset comprising data from 30-channel electroencephalography
(EEG), audio, and video recordings from 40 participants. Each participant engaged in a cue-based conversation scenario,
eliciting five distinct emotions: neutral, anger, happiness, sadness, and calmness. Throughout the experiment, each participant
contributed 200 interactions, which encompassed both listening and speaking. This resulted in a cumulative total of 8,000
interactions across all participants.


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

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->
## Usage

Run demo:
   ```sh
   python demo.py
   ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Training

To train the Transfromer for Speech emotion recognition run:
   ```sh
   python Dataload_audio.py
   ```
<p align="right">(<a href="#readme-top">back to top</a>)</p>    

<!-- ROADMAP -->
## Roadmap

- [x] CNN based Emotion recognition on Video, Audio and EEG domains using Tensorflow
- [x] CNN based Emotion recognition on Video and EEG domains using PyTorch
- [x] Transformer based Emotion recognition on Video, Audio and EEG domains using PyTorch
- [ ] Create demo file 
- [ ] Add .pkl files of preprocessed video data (Feature_vision folder)
- [ ] Add inference files


<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTACT -->
## Contact

Minho Lee - minho.lee@nu.edu.kz

Zhuldyz Kabidenova - zhuldyz.kabidenova@nu.edu.kz

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->

[product-screenshot]: images/EAVlogo.png