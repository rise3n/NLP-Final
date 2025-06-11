import os
import torch
import torchaudio
import pandas as pd
from tqdm import tqdm
from transformers import Wav2Vec2Processor, Wav2Vec2Model, Wav2Vec2Config
from sklearn.preprocessing import LabelEncoder
import torch.nn as nn
import torch.nn.functional as F

from decoder import EmergencyDecoder
from adapter import Adapter

'''
next step: 
1. split all recordings into training set and validation set.
2. train using batch 
3. observe performance and determine if we need to modify any thing (like if we should use z in decoder)
need to be done before friday
'''


# load processor and wav2vec2 model
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
encoder = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h").cuda()
CHUNK_LEN = 160000 
hidden_size = encoder.config.hidden_size

#add in a adapter for each layer of encoder
for layer in encoder.encoder.layers:
    #hidden_size = layer.feed_forward.intermediate_dense.out_features  
    layer.adapter = Adapter(hidden_size, bottleneck_size=64).to(layer.feed_forward.intermediate_dense.weight.device)

#this would be invoked when using hooks
def _add_adapter(module, inputs, outputs):
    return module.adapter(outputs.last_hidden_state)

#hooks are auto invoked functions that we defined between layers
#here I use hook to invoke my adapter
hooks = []
for layer in encoder.encoder.layers:
    h = layer.feed_forward.register_forward_hook(
        lambda module, input, output: layer.adapter(output) # don't delete module and input, they are there because it requires three parameters
    )
    hooks.append(h)

for name, param in encoder.named_parameters():
    if "adapter" not in name:
        param.requires_grad = False # only update adapters param

# load csv with labels
df = pd.read_csv("processed_data.csv")
df = df[df["file_name"].notna()].head(100)

# encode type and priority
type_encoder = LabelEncoder()
priority_encoder = LabelEncoder()
df["type_label"] = type_encoder.fit_transform(df["type"])
df["priority_label"] = priority_encoder.fit_transform(df["priority"])

AUDIO_DIR = "./data/waveForm"

decoder = EmergencyDecoder(input_dim_ct=768, input_dim_z=512).cuda()

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(decoder.parameters(), lr=1e-4)

decoder.train()
for idx, row in tqdm(df.iterrows(), total=len(df)):
    path = os.path.join(AUDIO_DIR, row["file_name"].replace(".mp3", "output.wav"))

    try:
        wav, sr = torchaudio.load(path)
    except Exception as e:
        print(f"Error loading {path}: {e}")
        continue

    if sr != 16000:
        wav = torchaudio.transforms.Resample(sr, 16000)(wav)
    wav = wav[0].numpy()

    chunks = [wav[i:i+CHUNK_LEN] for i in range(0, len(wav), CHUNK_LEN)]
    C_t = []
    Z = []

    for chunk in chunks:
        inputs = processor(
        chunk,
        sampling_rate=16000,
        return_tensors="pt",
        padding=True,               
        return_attention_mask=True,
)
        with torch.no_grad():
            input_values = inputs.input_values.cuda()
            attention_mask = inputs.attention_mask.cuda()
            output = encoder(input_values, attention_mask=attention_mask)
            hidden_states = output.last_hidden_state

        c_t = hidden_states.squeeze(0)
        z = output.extract_features
        
        C_t.append(c_t)
        Z.append(z)
        
    Ct = torch.cat(C_t, dim=0) 
    Z = torch.cat(Z, dim=1)
    Ct = Ct.unsqueeze(0) # to make sure shape of Ct align with Z
    
    Ct = Ct.permute(1, 0, 2)
    Z = Z.permute(1, 0, 2)
    
    logits_em, logits_pr = decoder(Ct, Z)

    loss1 = loss_fn(logits_em, torch.tensor([row["type_label"]]).cuda())
    loss2 = loss_fn(logits_pr, torch.tensor([row["priority_label"]]).cuda())
    loss = loss1 + loss2

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    print(f"{idx}: loss = {loss.item():.4f}")

torch.save(decoder.state_dict(), "decoder_trained.pt")
torch.save(type_encoder, "type_encoder.pkl")
torch.save(priority_encoder, "priority_encoder.pkl")
