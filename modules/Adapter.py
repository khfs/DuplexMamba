import torch
import torch.nn as nn

class EncoderProjectorConcat(nn.Module):
    def __init__(self, config, bias=True):
        super().__init__()
        # Downsample factor and dimensions from config
        self.k = config['speech_encoder_ds_rate']
        self.encoder_dim = config['speech_encoder_hidden_size']
        self.llm_dim = config['hidden_size']
        
        # Linear layers with ReLU activation in between
        self.linear1 = nn.Linear(self.encoder_dim * self.k, 1024, bias=bias)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(1024, self.llm_dim, bias=bias)

    def forward(self, x):
        batch_size, seq_len, dim = x.size()

        # Handling the case where sequence length is not divisible by k
        num_frames_to_discard = seq_len % self.k
        if num_frames_to_discard > 0:
            x = x[:, :-num_frames_to_discard, :]
        
        # Reshape by concatenating every k frames along the feature dimension
        seq_len = x.size(1)
        x = x.view(batch_size, seq_len // self.k, dim * self.k)
        
        # Pass through MLP (two linear layers with ReLU in between)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        
        return x





