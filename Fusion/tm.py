from transformers import AutoModelForAudioClassification
from Fusion.Transformer import ViT_Encoder, ast_feature_extract, Trainer_uni
import torch
from Dataload_audio import DataLoadAudio
from EAV_datasplit import EAVDataSplit

mod_path = r"C:\Users\minho.lee\Dropbox\Projects\EAV\ast-finetuned-audioset"
model_pre = AutoModelForAudioClassification.from_pretrained(mod_path)

model_weights = []
model_bias = []
for layer_name, param in model_pre.state_dict().items():
    if 'weight' in layer_name:  # Check if the parameter is a weight parameter
        model_weights.append((layer_name, param))

for layer_name, param in model_pre.state_dict().items():
    if 'bias' in layer_name:  # Check if the parameter is a bias parameter
        model_bias.append((layer_name, param))
        print(layer_name)

model = ViT_Encoder(classifier = True, img_size=[1024, 128], in_chans=1, patch_size = (16, 16), stride = 10, embed_pos = True)

weight_layer_name = "audio_spectrogram_transformer.embeddings.patch_embeddings.projection.weight"
weight_tensor = next((tensor for name, tensor in model_weights if name == weight_layer_name), None)
model.patch_embed.proj.weight.data = weight_tensor

bias_layer_name = "audio_spectrogram_transformer.embeddings.patch_embeddings.projection.bias"
bias_tensor = next((tensor for name, tensor in model_bias if name == bias_layer_name), None)
model.patch_embed.proj.bias.data = bias_tensor

for idx in range(12):
    # Layer norm before weights and biases
    weight_tensor = next((tensor for layer_name, tensor in model_weights if layer_name == f"audio_spectrogram_transformer.encoder.layer.{idx}.layernorm_before.weight"), None)
    bias_tensor = next((tensor for layer_name, tensor in model_bias if layer_name == f"audio_spectrogram_transformer.encoder.layer.{idx}.layernorm_before.bias"), None)

    model.blocks[idx].norm1.weight.data = weight_tensor
    model.blocks[idx].norm1.bias.data = bias_tensor

    # Query, Key, Value weights and biases for attention
    q_weight = next((tensor for layer_name, tensor in model_weights if layer_name == f"audio_spectrogram_transformer.encoder.layer.{idx}.attention.attention.query.weight"), None)
    k_weight = next((tensor for layer_name, tensor in model_weights if layer_name == f"audio_spectrogram_transformer.encoder.layer.{idx}.attention.attention.key.weight"), None)
    v_weight = next((tensor for layer_name, tensor in model_weights if layer_name == f"audio_spectrogram_transformer.encoder.layer.{idx}.attention.attention.value.weight"), None)
    q_bias = next((tensor for layer_name, tensor in model_bias if layer_name == f"audio_spectrogram_transformer.encoder.layer.{idx}.attention.attention.query.bias"), None)
    k_bias = next((tensor for layer_name, tensor in model_bias if layer_name == f"audio_spectrogram_transformer.encoder.layer.{idx}.attention.attention.key.bias"), None)
    v_bias = next((tensor for layer_name, tensor in model_bias if layer_name == f"audio_spectrogram_transformer.encoder.layer.{idx}.attention.attention.value.bias"), None)

    weight_tensor = torch.cat((q_weight, k_weight, v_weight), dim=0)
    bias_tensor = torch.cat((q_bias, k_bias, v_bias), dim=0)
    model.blocks[idx].attn.qkv.weight.data = weight_tensor
    model.blocks[idx].attn.qkv.bias.data = bias_tensor

    # Attention output dense weights and biases
    weight_tensor = next((tensor for layer_name, tensor in model_weights if layer_name == f"audio_spectrogram_transformer.encoder.layer.{idx}.attention.output.dense.weight"), None)
    bias_tensor = next((tensor for layer_name, tensor in model_bias if layer_name == f"audio_spectrogram_transformer.encoder.layer.{idx}.attention.output.dense.bias"), None)

    model.blocks[idx].attn.proj.weight.data = weight_tensor
    model.blocks[idx].attn.proj.bias.data = bias_tensor

    # Layer norm after weights and biases
    weight_tensor = next((tensor for layer_name, tensor in model_weights if layer_name == f"audio_spectrogram_transformer.encoder.layer.{idx}.layernorm_after.weight"), None)
    bias_tensor = next((tensor for layer_name, tensor in model_bias if layer_name == f"audio_spectrogram_transformer.encoder.layer.{idx}.layernorm_after.bias"), None)

    model.blocks[idx].norm2.weight.data = weight_tensor
    model.blocks[idx].norm2.bias.data = bias_tensor

    # MLP intermediate and output dense weights and biases
    weight_tensor = next((tensor for layer_name, tensor in model_weights if layer_name == f"audio_spectrogram_transformer.encoder.layer.{idx}.intermediate.dense.weight"), None)
    bias_tensor = next((tensor for layer_name, tensor in model_bias if layer_name == f"audio_spectrogram_transformer.encoder.layer.{idx}.intermediate.dense.bias"), None)

    model.blocks[idx].mlp.fc1.weight.data = weight_tensor
    model.blocks[idx].mlp.fc1.bias.data = bias_tensor

    weight_tensor = next((tensor for layer_name, tensor in model_weights if layer_name == f"audio_spectrogram_transformer.encoder.layer.{idx}.output.dense.weight"), None)
    bias_tensor = next((tensor for layer_name, tensor in model_bias if layer_name == f"audio_spectrogram_transformer.encoder.layer.{idx}.output.dense.bias"), None)

    model.blocks[idx].mlp.fc2.weight.data = weight_tensor
    model.blocks[idx].mlp.fc2.bias.data = bias_tensor

for name, param in model.named_parameters():
    if 'norm.weight' in name or 'norm.bias' in name:
        param.requires_grad = True
for name, param in model.named_parameters():
    print(f"{name} trainable: {param.requires_grad}")

aud_loader = DataLoadAudio(subject=3, parent_directory=r'C:\\Users\\minho.lee\\Dropbox\\Datasets\\EAV')
[data_aud, data_aud_y] = aud_loader.process()
division_aud = EAVDataSplit(data_aud, data_aud_y)
[tr_x_aud, tr_y_aud, te_x_aud, te_y_aud] = division_aud.get_split()
tr_x_aud_ft = ast_feature_extract(tr_x_aud)
te_x_aud_ft = ast_feature_extract(te_x_aud)
data = [tr_x_aud_ft.unsqueeze(1), tr_y_aud, te_x_aud_ft.unsqueeze(1), te_y_aud]

trainer = Trainer_uni(model=model, data=data, lr=1e-5, batch_size=8, num_epochs=20)
trainer.train()

for name, param in model.named_parameters():
    param.requires_grad = True

trainer = Trainer_uni(model=model, data=data, lr=1e-6, batch_size=8, num_epochs=20)
trainer.train()


