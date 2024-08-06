import torch
from torch import nn
from torch.nn import functional as F
import math

class SelfAttention(nn.Module):
    
    def __init__(self, n_heads: int, d_embed: int, in_proj_bias=True, out_proj_bias=True):
        super().__init__()
        
        self.in_proj = nn.Linear(d_embed, 3 * d_embed, bias=in_proj_bias) #Here we are writing the 3 matrices, q,k,v as one big matrix
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias) #This is what would correspond to Wo in the Attention paper
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads
        
    def forward(self, x: torch.Tensor, causal_mask=False):
        # x: (Batch_size, seq_len, dim)
        
        input_shape = x.shape
        
        batch_size, sequence_length, d_embed = input_shape
        
        intermim_shape = (batch_size, sequence_length, self.n_heads, self.d_head)
        
        # (batch_size, seq_len, dim) -> (batch_size, seq_len, dim * 3) -> 3 tensors of shape (batch_size, seq_len, dim)
        q, k ,v = self.in_proj(x).chunk(3, dim=-1)
        
        # (Batch_size, seq_len, dim) -> (Batch, seq_len, H, Dim/H) -> (Batch, H, seq_len, Dim/H)) So each head will watch all the sequence but only a part of the embedding
        q = q.view(intermim_shape).transpose(1,2) 
        k = k.view(intermim_shape).transpose(1,2) 
        v = v.view(intermim_shape).transpose(1,2) 
        
        # (Batch_size, H, seq_len, seq_len)  
        weight = q @ k.transpose(-1, -2)
        
        if causal_mask:
            #mask where the upper triangle (above the principle diagonal) is made up of 1
            mask = torch.ones_like(weight, dtype=torch.bool).triu(1)
            weight.masked_fill_(mask, -torch.inf)
            
        weight /= math.sqrt(self.d_head) #just follow the self attention formula
        
        weight = F.softmax(weight, dim=-1)
        
        # (Batch_size, H, seq_len, seq_len) @ (Batch_size, H, seq_len, Dim / H) -> (Batch_size, H, seq_len, Dim / H)
        output = weight @ v

        # (Batch_size, H, seq_len, Dim / H) -> (Batch_size, seq_len, H, Dim / H)
        output = output.transpose(1, 2)
        
        output = output.reshape(input_shape)
        
        output = self.out_proj(output) #Here we're multiplying by the Wo matrix
        
        # (Batch_size, seq_len, dim)
        return output
        
        
        