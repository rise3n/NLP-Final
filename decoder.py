import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2Model

'''
this is a tentative local attention layer since some audio is very long, forcing a small window may speed up the response
but this may sacrifice accuracy as small window size may not be able to capture useful information

'''
class SparseLocalAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, window_size: int):
        super(SparseLocalAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.window_size = window_size

        self.mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=False)

    def _generate_local_mask(self, seq_len: int, device: torch.device):
        mask = torch.full((seq_len, seq_len), float('-inf'), device=device)
        for i in range(seq_len):
            low = max(0, i - self.window_size)
            high = min(seq_len, i + self.window_size + 1)
            mask[i, low:high] = 0.0
        return mask  # [S, S]

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor):
        seq_len, batch_size, _ = key.size()
        device = key.device

        attn_mask = self._generate_local_mask(seq_len, device)  # [S, S]

        out, attn_weights = self.mha(query, key, value, attn_mask=attn_mask)
        return out, attn_weights


'''
In decoder part, I plan to use a unmasked global attention for one time, and use local attention for latter layers
'''
class EmergencyDecoder(nn.Module):
    def __init__(
        self,
        input_dim_ct: int,
        input_dim_z: int,
        embed_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 2,
        window_size: int = 50,
        num_emergency_classes: int = 11,
        num_priority_classes: int = 5,
        dropout: float = 0.1
    ):
        super(EmergencyDecoder, self).__init__()

        self.embed_dim = embed_dim
        self.num_layers = num_layers

        self.linear_ct = nn.Linear(input_dim_ct, embed_dim)
        self.linear_z = nn.Linear(input_dim_z, embed_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            layer = nn.ModuleDict({
                #
                'global_attn': nn.MultiheadAttention(embed_dim, num_heads, batch_first=False),
                'sparse_attn': SparseLocalAttention(embed_dim, num_heads, window_size),
                'norm1': nn.LayerNorm(embed_dim),
                'norm2': nn.LayerNorm(embed_dim),
                'norm3': nn.LayerNorm(embed_dim),
                'ffn': nn.Sequential(
                    nn.Linear(embed_dim, embed_dim * 4),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(embed_dim * 4, embed_dim),
                    nn.Dropout(dropout),
                ),
            })
            self.layers.append(layer)

        self.classifier_emergency = nn.Linear(embed_dim, num_emergency_classes)
        self.classifier_priority = nn.Linear(embed_dim, num_priority_classes)

        self.dropout = nn.Dropout(dropout)

    def forward(self, ct: torch.Tensor, z: torch.Tensor):
        ct_proj = self.linear_ct(ct)   # [S_ct, B, embed_dim]
        z_proj  = self.linear_z(z)     # [S_z,  B, embed_dim]

        key_value = torch.cat([ct_proj, z_proj], dim=0)
        seq_len_kv, batch_size, _ = key_value.size()

        cls_q = self.cls_token.expand(-1, batch_size, -1)  # [1, B, D]

        x = cls_q  
        for layer in self.layers:
            attn_out_global, _ = layer['global_attn'](x, key_value, key_value, attn_mask=None)
            x_global = layer['norm1'](x + self.dropout(attn_out_global))
            
            attn_out, _ = layer['sparse_attn'](x_global, key_value, key_value)  # [1, B, D]
            x = layer['norm2'](x_global + self.dropout(attn_out))  # [1, B, D]

            ffn_out = layer['ffn'](x)  # [1, B, D]
            x = layer['layernorm2'](x + ffn_out)  # [1, B, D]

        cls_repr = x.squeeze(0)  # [B, D]

        logits_emergency = self.classifier_emergency(cls_repr)  # [B, num_emergency_classes]
        logits_priority  = self.classifier_priority(cls_repr)   # [B, num_priority_classes]

        return logits_emergency, logits_priority


def main():
    S_ct = 800          
    S_z = 200           
    C_ct_dim = 256      
    Z_dim = 128

    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
    encoder     = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")

    encoder.eval().cuda()
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
            outputs = encoder(input_values = segment, return_dict=True)
            # last_hidden_state: (batch, time, hidden_dim)
            c_t = outputs.last_hidden_state.squeeze(0)  # context representation
            z = outputs.extract_features
            C = torch.cat((C, c_t), dim = 0)
            Z = torch.cat((Z, z), dim = 1)
        
        print("c_t", C.shape)
        print("Z", Z.shape)

    # Decoder
    decoder = EmergencyDecoder(
        input_dim_ct=C_ct_dim,
        input_dim_z=Z_dim,
        embed_dim=512,
        num_heads=8,
        num_layers=2,
        window_size=50,
        num_emergency_classes=5,
        num_priority_classes=3,
        dropout=0.1
    )

    logits_emergency, logits_priority = decoder(C, Z)
    print("Logits Emergency:", logits_emergency.shape)  # [B, 5]
    print("Logits Priority :", logits_priority.shape)   # [B, 3]


    y_emergency = torch.randint(0, 5, (B,))   # place holder for real tag y
    y_priority  = torch.randint(0, 3, (B,))   # same as above

    emergency_weights = torch.tensor([1.0, 2.0, 1.0, 1.5, 1.0], device=logits_emergency.device) # this weight is here for imbalanced emergency type and priority
    priority_weights  = torch.tensor([1.0, 3.0, 1.0], device=logits_priority.device)

    loss_fn_emergency = nn.CrossEntropyLoss(weight=emergency_weights)
    loss_fn_priority  = nn.CrossEntropyLoss(weight=priority_weights)

    loss_em = loss_fn_emergency(logits_emergency, y_emergency)
    loss_pr = loss_fn_priority(logits_priority, y_priority)
    total_loss = loss_em + loss_pr
    print("Total Loss:", total_loss.item())


if __name__ == "__main__":
    main()