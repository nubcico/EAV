from transformers import AutoImageProcessor, AutoModelForImageClassification
from Fusion.VIT_vision.Transformer_video import ViT_Encoder_Video, Trainer_uni
import torch
from Dataload_audio import DataLoadAudio
from EAV_datasplit import EAVDataSplit
import os
mod_path = os.path.join(r'C:\Users\minho.lee\Dropbox\Projects\EAV', 'facial_emotions_image_detection')
model_pre = AutoModelForImageClassification.from_pretrained(mod_path)

model_weights = []
model_bias = []
for layer_name, param in model_pre.state_dict().items():
    if 'weight' in layer_name:  # Check if the parameter is a weight parameter
        model_weights.append((layer_name, param))

for layer_name, param in model_pre.state_dict().items():
    if 'bias' in layer_name:  # Check if the parameter is a bias parameter
        model_bias.append((layer_name, param))
        print(layer_name)

model = ViT_Encoder_Video(classifier=True, img_size=(224, 224), in_chans=3, patch_size=(16, 16), stride=16, embed_pos=True)

weight_layer_name = "vit.embeddings.patch_embeddings.projection.weight"
weight_tensor = next((tensor for name, tensor in model_weights if name == weight_layer_name), None)
model.patch_embed.proj.weight.data = weight_tensor

bias_layer_name = "vit.embeddings.patch_embeddings.projection.bias"
bias_tensor = next((tensor for name, tensor in model_bias if name == bias_layer_name), None)
model.patch_embed.proj.bias.data = bias_tensor

for idx in range(12):
    # Layer norm before weights and biases
    weight_tensor = next((tensor for layer_name, tensor in model_weights if layer_name == f"vit.encoder.layer.{idx}.layernorm_before.weight"), None)
    bias_tensor = next((tensor for layer_name, tensor in model_bias if layer_name == f"vit.encoder.layer.{idx}.layernorm_before.bias"), None)

    model.blocks[idx].norm1.weight.data = weight_tensor
    model.blocks[idx].norm1.bias.data = bias_tensor

    # Query, Key, Value weights and biases for attention
    q_weight = next((tensor for layer_name, tensor in model_weights if layer_name == f"vit.encoder.layer.{idx}.attention.attention.query.weight"), None)
    k_weight = next((tensor for layer_name, tensor in model_weights if layer_name == f"vit.encoder.layer.{idx}.attention.attention.key.weight"), None)
    v_weight = next((tensor for layer_name, tensor in model_weights if layer_name == f"vit.encoder.layer.{idx}.attention.attention.value.weight"), None)
    q_bias = next((tensor for layer_name, tensor in model_bias if layer_name == f"vit.encoder.layer.{idx}.attention.attention.query.bias"), None)
    k_bias = next((tensor for layer_name, tensor in model_bias if layer_name == f"vit.encoder.layer.{idx}.attention.attention.key.bias"), None)
    v_bias = next((tensor for layer_name, tensor in model_bias if layer_name == f"vit.encoder.layer.{idx}.attention.attention.value.bias"), None)

    weight_tensor = torch.cat((q_weight, k_weight, v_weight), dim=0)
    bias_tensor = torch.cat((q_bias, k_bias, v_bias), dim=0)
    
    model.blocks[idx].attn.qkv.weight.data = weight_tensor
    model.blocks[idx].attn.qkv.bias.data = bias_tensor

    # Attention output dense weights and biases
    weight_tensor = next((tensor for layer_name, tensor in model_weights if layer_name == f"vit.encoder.layer.{idx}.attention.output.dense.weight"), None)
    bias_tensor = next((tensor for layer_name, tensor in model_bias if layer_name == f"vit.encoder.layer.{idx}.attention.output.dense.bias"), None)

    model.blocks[idx].attn.proj.weight.data = weight_tensor
    model.blocks[idx].attn.proj.bias.data = bias_tensor

    # Layer norm after weights and biases
    weight_tensor = next((tensor for layer_name, tensor in model_weights if layer_name == f"vit.encoder.layer.{idx}.layernorm_after.weight"), None)
    bias_tensor = next((tensor for layer_name, tensor in model_bias if layer_name == f"vit.encoder.layer.{idx}.layernorm_after.bias"), None)

    model.blocks[idx].norm2.weight.data = weight_tensor
    model.blocks[idx].norm2.bias.data = bias_tensor

    # MLP intermediate and output dense weights and biases
    weight_tensor = next((tensor for layer_name, tensor in model_weights if layer_name == f"vit.encoder.layer.{idx}.intermediate.dense.weight"), None)
    bias_tensor = next((tensor for layer_name, tensor in model_bias if layer_name == f"vit.encoder.layer.{idx}.intermediate.dense.bias"), None)

    model.blocks[idx].mlp.fc1.weight.data = weight_tensor
    model.blocks[idx].mlp.fc1.bias.data = bias_tensor

    weight_tensor = next((tensor for layer_name, tensor in model_weights if layer_name == f"vit.encoder.layer.{idx}.output.dense.weight"), None)
    bias_tensor = next((tensor for layer_name, tensor in model_bias if layer_name == f"vit.encoder.layer.{idx}.output.dense.bias"), None)

    model.blocks[idx].mlp.fc2.weight.data = weight_tensor
    model.blocks[idx].mlp.fc2.bias.data = bias_tensor
    
    torch.save(model.state_dict(), 'model_with_weights_video.pth') #we can just use the state dict after the first assignment

# for name, param in model.named_parameters():
#     if 'norm.weight' in name or 'norm.bias' in name:
#         param.requires_grad = True
# for name, param in model.named_parameters():
#     print(f"{name} trainable: {param.requires_grad}")

# aud_loader = DataLoadAudio(subject=3, parent_directory=r'D:\EAV')
# [data_aud, data_aud_y] = aud_loader.process()
# division_aud = EAVDataSplit(data_aud, data_aud_y)
# [tr_x_aud, tr_y_aud, te_x_aud, te_y_aud] = division_aud.get_split()
# tr_x_aud_ft = ast_feature_extract(tr_x_aud)
# te_x_aud_ft = ast_feature_extract(te_x_aud)
# data = [tr_x_aud_ft.unsqueeze(1), tr_y_aud, te_x_aud_ft.unsqueeze(1), te_y_aud]

# trainer = Trainer_uni(model=model, data=data, lr=1e-5, batch_size=8, num_epochs=20)
# trainer.train()

# for name, param in model.named_parameters():
#     param.requires_grad = True

# trainer = Trainer_uni(model=model, data=data, lr=1e-6, batch_size=8, num_epochs=20)
# trainer.train()


if __name__ == '__main__':
    import numpy as np
    import pickle
    import os
    from sklearn.metrics import f1_score
    test_acc = []
    for idx in range(4, 5):
        torch.cuda.reset_max_memory_allocated()
        torch.cuda.reset_max_memory_cached()
        file_name = f"subject_{idx:02d}_vis.pkl"
        file_ = os.path.join(r"C:\Users\minho.lee\Dropbox\Projects\EAV\Feature_vision", file_name)

        with open(file_, 'rb') as f:
            vis_list2 = pickle.load(f)
        tr_x_vis, tr_y_vis, te_x_vis, te_y_vis = vis_list2

        data = [tr_x_vis, tr_y_vis, te_x_vis, te_y_vis]
        
        trainer = Trainer_uni(model=model, data=data, lr=1e-5, batch_size=128, num_epochs=30)

        trainer.train(epochs=10, lr=5e-4, freeze=True)
        trainer.train(epochs=5, lr=5e-6, freeze=False)
        
        feature_map=trainer.feature_map
        
        test_acc_all = list()
        test_f1_all = list()
        
        aa = trainer.outputs_test
        #print(f"Shape of aa before reshaping: {aa.shape}")
        aa2 = np.reshape(aa, (200, 25, 5), 'C')
        #print(f"Shape of aa2 after reshaping: {aa2.shape}")
        aa3 = np.mean(aa2, 1)
        out1 = np.argmax(aa3, axis = 1)
        accuracy = np.mean(out1 == te_y_vis)
        test_acc_all.append(accuracy)

        f1 = f1_score(te_y_vis, out1, average='weighted')
        test_f1_all.append(f1)
        print(f"Accuracy is {accuracy} and f1 score is {f1}")
        
        f = open("accuracy_vid.txt", "a")
        f.write("\n Subject ")
        f.write(str(idx))
        f.write("\n")
        f.write(f"The accuracy of the {idx}-subject is ")
        f.write(str(accuracy))
        f.write(f"The f1-score of the {idx}-subject is ")
        f.write(str(f1))
        f.close()   

        import numpy as np
        from sklearn.manifold import TSNE
        import matplotlib.pyplot as plt
        
        # Flatten the feature map if necessary, e.g., from (B, N, D) to (B*N, D)
        flattened_feature_map = feature_map.view(-1, feature_map.size(-1)).cpu().detach().numpy()
        
        # Apply t-SNE
        tsne = TSNE(n_components=2, random_state=42)
        tsne_results = tsne.fit_transform(flattened_feature_map)

# Plot the t-SNE results
plt.figure(figsize=(8, 6))
plt.scatter(tsne_results[:, 0], tsne_results[:, 1], s=1, alpha=0.7)
plt.title('t-SNE of Feature Map')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.show()
    
# import os
# import pickle   

# test_acc_all = list()
# for idx in range(4, 5):
#     test_acc = []
#     torch.cuda.empty_cache()
#     direct=r"C:\Users\user.DESKTOP-HI4HHBR\Downloads\Feature_vision"
#     file_name = f"subject_{idx:02d}_aud.pkl"
#     file_ = os.path.join(direct, file_name)

#     with open(file_, 'rb') as f:
#         vis_list2 = pickle.load(f)
        
#     tr_x_vis, tr_y_vis, te_x_vis, te_y_vis = vis_list2

#     tr_x_aud_ft = ast_feature_extract(tr_x_vis)
#     te_x_aud_ft = ast_feature_extract(te_x_vis)
#     tr_y_aud=tr_y_vis
#     te_y_aud=te_y_vis

#     data = [tr_x_aud_ft.unsqueeze(1), tr_y_aud, te_x_aud_ft.unsqueeze(1), te_y_aud]
    
#     for name, param in model.named_parameters():
#         param.requires_grad = False
        
#     for param in model.head.parameters():
#         param.requires_grad = True

#     for name, param in model.named_parameters():
#         print(f"{name} trainable: {param.requires_grad}")
        
#     trainer = Trainer_uni(model=model, data=data, lr=5e-4, batch_size=8, num_epochs=20)
#     trainer.train()

#     for name, param in model.named_parameters():
#         param.requires_grad = True

#     trainer = Trainer_uni(model=model, data=data, lr=5e-6, batch_size=8, num_epochs=20)
#     trainer.train()

#     accuracy=trainer.outputs_test
#     f = open("accuracy_aud_tm.txt", "a")
#     f.write("\n Subject ")
#     f.write(str(idx))
#     f.write("\n")
#     f.write(f"The accuracy of the {idx}-subject is ")
#     f.write(str(accuracy))
#     print(f"The accuracy of the {idx}-subject is ")
#     print(accuracy)
#     f.close()





