import os
import torch
import torchaudio
import pandas as pd
from tqdm import tqdm
from transformers import Wav2Vec2Processor, Wav2Vec2Model, Wav2Vec2Config
from sklearn.preprocessing import LabelEncoder
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from decoder import EmergencyDecoder
from adapter import Adapter
import wandb
from torch.optim.lr_scheduler import CosineAnnealingLR

'''
next step: 
1. split all recordings into training set and validation set.
2. train using batch 
3. observe performance and determine if we need to modify any thing (like if we should use z in decoder)
need to be done before friday
'''

use_cuda_if_avail = True
if use_cuda_if_avail and torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

print(device)
config = {
    "bs":10,   # batch size
    "lr":1e-4, # learning rate
    "l2reg":0.0000001, # weight decay
    "max_epoch":20,
    "input_dim_ct":768,
    "input_dim_z":512,
    "n_layers":8
}

CHUNK_LEN = 160000
AUDIO_DIR = "./data/waveForm"

def collate_fn(batch, processor, encoder):
    max_len = 0
    Ct_batch, Z_batch, type_labels, priority_labels = [], [], [], []
    for row in batch:
        path = os.path.join(AUDIO_DIR, row["file_name"].replace(".mp3", "output.wav"))
        try:
            wav, sr = torchaudio.load(path)
            if sr != 16000:
                wav = torchaudio.transforms.Resample(sr, 16000)(wav)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            wav = torch.zeros(1, CHUNK_LEN)
        wav = wav[0]  # mono
        chunks = [wav[i:i+CHUNK_LEN] for i in range(0, len(wav), CHUNK_LEN)]
        C_t, Z = [], []
        for chunk in chunks:
            inputs = processor(chunk, sampling_rate=16000, return_tensors="pt", padding=True, return_attention_mask=True)
            input_values = inputs.input_values.cuda()
            attention_mask = inputs.attention_mask.cuda()
            with torch.no_grad():
                output = encoder(input_values, attention_mask=attention_mask)
                hidden_states = output.last_hidden_state
                c_t = hidden_states.squeeze(0)
                z = output.extract_features
                C_t.append(c_t)
                Z.append(z)
        #Ct = torch.cat(C_t, dim=0).unsqueeze(1)
        Ct = torch.cat(C_t, dim=0)
        #Z = torch.cat(Z, dim=1).permute(1, 0, 2)
        Z = torch.cat(Z, dim=1).squeeze(0)
        max_len = max(max_len, Ct.size(0))
        #Ct = Ct.permute(1, 0, 2)
        Ct_batch.append(Ct)
        Z_batch.append(Z)
        type_labels.append(torch.tensor(row["type_label"]))
        priority_labels.append(torch.tensor(row["priority_label"]))
        
    Ct_batch = pad_sequence(Ct_batch, batch_first=True, padding_value=-1)
    Z_batch = pad_sequence(Z_batch, batch_first=True, padding_value=-1)

    return Ct_batch, Z_batch, torch.stack(type_labels), torch.stack(priority_labels)


def load_model():
    # load processor and wav2vec2 model
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    encoder = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h").cuda()
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
            
    return processor, encoder


def load_dataset(processor, encoder, batchsize):
    # load csv with labels
    df = pd.read_csv("processed_data.csv")
    df = df[df["file_name"].notna()].reset_index(drop=True)

    # encode type and priority
    type_encoder = LabelEncoder()
    priority_encoder = LabelEncoder()
    df["type_label"] = type_encoder.fit_transform(df["type"])
    df["priority_label"] = priority_encoder.fit_transform(df["priority"])
    
    dataloader = DataLoader(
        df.to_dict("records"),
        batch_size=batchsize,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, processor, encoder)
    )
    
    return dataloader, type_encoder, priority_encoder


def main():
    batchsize = 10
    processor, encoder = load_model()
    dataloader, type_encoder, priority_encoder = load_dataset(processor, encoder, batchsize)

    num_em_classes = len(type_encoder.classes_)
    num_pr_classes = len(priority_encoder.classes_)
    decoder = EmergencyDecoder(input_dim_ct=768, 
                            input_dim_z=512,
                            num_emergency_classes=num_em_classes,
                            num_priority_classes=num_pr_classes).cuda()

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(decoder.parameters(), lr=1e-4)
    
    run_name = "1"
    wandb.login(key = "96b1bbbb59977d4eb984e270c9c5aae2077b1f41") # modify this to your key 
    wandb.init(project="[AI539] final project", name=run_name, config=config)

    scheduler = CosineAnnealingLR(optimizer, T_max = config["max_epoch"])

    decoder.train()
    for epoch in range(config["max_epoch"]):
        for batch in tqdm(dataloader):
            Ct, Z, type_label, priority_label = batch
            logits_em, logits_pr = decoder(Ct.permute(1, 0, 2), Z.permute(1, 0, 2))

            loss1 = loss_fn(logits_em, type_label.cuda())
            loss2 = loss_fn(logits_pr, priority_label.cuda())
            loss = loss1 + loss2

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            wandb.log({"LR/lr": scheduler.get_last_lr()[0]}, step=epoch)
            wandb.log({"Loss/train": loss.item()}, step=epoch)

    torch.save(decoder.state_dict(), "decoder_trained.pt")
    torch.save(type_encoder, "type_encoder.pkl")
    torch.save(priority_encoder, "priority_encoder.pkl")


if __name__ == "__main__":
    main()