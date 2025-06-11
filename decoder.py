import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2Processor, Wav2Vec2Model, LongformerModel
from transformers.models.longformer.configuration_longformer import LongformerConfig

'''
modified: remove local attention because it need way too much memory when audio is too long.
instead, I use longformer which use pooling and few layers of mlp to generate short enough summaries of chunks (length of chunk is
tunable)
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
        dropout: float = 0.1,
        chunk_size: int = 800,
        summary_dim: int = 256
    ):
        super(EmergencyDecoder, self).__init__()
        self.embed_dim   = embed_dim
        self.chunk_size  = chunk_size
        self.window_size = window_size
        self.num_layers  = num_layers

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        hidden_size = input_dim_ct
        self.summary_mlp = nn.Sequential(
            nn.Conv1d(hidden_size, summary_dim, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(summary_dim, embed_dim, kernel_size=1)
        )

        attention_window = [window_size] * num_layers
        #attention_dilation is how many tokens are skipped within window
        #when dilation = 1, window might be [Token1, Token2, Token3, Token4, Token5]
        #when dilation = 3, window might be [Token1, Token3, Token5]
        attention_dilation = [1] * num_layers 
        lf_config = LongformerConfig(
            hidden_size=embed_dim,
            num_attention_heads=num_heads,
            attention_window=attention_window,
            attention_dilation=attention_dilation,
            hidden_dropout_prob=dropout,
            attention_probs_dropout_prob=dropout,
            layer_norm_eps=1e-5,
            num_hidden_layers=num_layers,
            intermediate_size=embed_dim * 4
        )
        self.longformer = LongformerModel(lf_config)

        self.norm_final    = nn.LayerNorm(embed_dim)
        self.classifier_em = nn.Linear(embed_dim, num_emergency_classes)
        self.classifier_pr = nn.Linear(embed_dim, num_priority_classes)
        self.dropout       = nn.Dropout(dropout)

    # z is unused because z has no contextual information so it should not be that useful
    #not sure, check its performance later
    def forward(self, ct: torch.Tensor, z: torch.Tensor):
        T, B, H = ct.size()

        conv_in = ct.permute(1, 2, 0)  # [B, H, T]
        pad_len = (-conv_in.size(-1)) % self.chunk_size
        if pad_len > 0:
            #pad the tensor so it is integer times of chunk length
            conv_in = F.pad(conv_in, (0, pad_len))
        chunks = conv_in.unfold(2, self.chunk_size, self.chunk_size)
        chunk_summ = chunks.mean(-1)  # [B, H, num_chunks]

        summaries = self.summary_mlp(chunk_summ)
        summaries = summaries.permute(0, 2, 1) # [B, num_chunks, embed_dim]

        cls_expand = self.cls_token.expand(B, -1, -1)  # [B, 1, embed_dim]
        seq = torch.cat([summaries, cls_expand], dim=1)  # [B, S, embed_dim]

        # pad to multiple of window_size, now padding seq variable
        seq_len = seq.size(1)
        mod = seq_len % self.window_size
        if mod != 0:
            pad_len = self.window_size - mod
            seq = F.pad(seq, (0, 0, 0, pad_len))
            seq_len = seq.size(1)

        #longformer attention_mask: 1=local, 2=global for CLS
        #1 means attention is local, when the token is within window, it is unmasked, and 2 means that attention is global
        attn_mask = torch.ones(B, seq_len, dtype=torch.long, device=seq.device)
        attn_mask[:, -1] = 2

        outputs = self.longformer(
            inputs_embeds=seq,
            attention_mask=attn_mask,
            output_hidden_states=False,
            output_attentions=False
        )
        lm_out = outputs.last_hidden_state  # [B, S_padded, embed_dim]

        #remove paddings
        lm_out = lm_out[:, :summaries.size(1)+1, :]

        x = lm_out[:, -1, :]  # [B, embed_dim]

        #classification
        x = self.norm_final(x)
        x = self.dropout(x)
        logits_em = self.classifier_em(x)
        logits_pr = self.classifier_pr(x)
        return logits_em, logits_pr
