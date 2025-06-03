import fairseq 
import torch 
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import numpy as np 

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
model     = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")

model.eval().cuda()
wav, sr = torchaudio.load('.\\data\\waveForm\\call_1output.wav')

assert(sr == 16000)

inputs = processor(wav.squeeze(0).numpy(),
                sampling_rate=16000,
                return_tensors="pt",
                padding=True).to("cuda")

segment_length = sr * 10
segments = torch.split(wav, segment_length, dim=1)
C = torch.tensor([]).to('cuda')
Z = torch.tensor([]).to('cuda')

with torch.no_grad():
    for segment in segments:
        segment = segment.to('cuda')
        outputs = model(input_values = segment, return_dict=True)
        # last_hidden_state: (batch, time, hidden_dim)
        c_t = outputs.last_hidden_state.squeeze(0)  # context representation
        z = outputs.extract_features
        C = torch.cat((C, c_t), dim = 0)
        Z = torch.cat((Z, z), dim = 1)
    
    print("c_t", C.shape)
    print("Z", Z.shape)