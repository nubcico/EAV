---
license: bsd-3-clause
tags:
- audio-classification
---

# Audio Spectrogram Transformer (fine-tuned on AudioSet) 

Audio Spectrogram Transformer (AST) model fine-tuned on AudioSet. It was introduced in the paper [AST: Audio Spectrogram Transformer](https://arxiv.org/abs/2104.01778) by Gong et al. and first released in [this repository](https://github.com/YuanGongND/ast). 

Disclaimer: The team releasing Audio Spectrogram Transformer did not write a model card for this model so this model card has been written by the Hugging Face team.

## Model description

The Audio Spectrogram Transformer is equivalent to [ViT](https://huggingface.co/docs/transformers/model_doc/vit), but applied on audio. Audio is first turned into an image (as a spectrogram), after which a Vision Transformer is applied. The model gets state-of-the-art results on several audio classification benchmarks.

## Usage

You can use the raw model for classifying audio into one of the AudioSet classes. See the [documentation](https://huggingface.co/docs/transformers/main/en/model_doc/audio-spectrogram-transformer#transformers.ASTForAudioClassification.forward.example) for more info.