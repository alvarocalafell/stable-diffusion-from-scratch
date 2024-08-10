import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention

class CLIPEmbedding(nn.Module):
    def __init__(self, n_vocab: int, n_embd: int, n_tokens: int):
        super().__init__()
        
        self.token_embedding = nn.Embedding(n_vocab, n_embd)
        self.position_embedding = nn.Parameter(torch.zeros((n_tokens, n_embd)))
        
    def forward(self, tokens):
        # (Batch_size, seq_len) -> (Batch_size, seq_len, dim)
        x = self.token_embedding(tokens)
        
        x += self.position_embedding
        
        return x
    
class CLIPLayer(nn.Module):
    
    
    def __init__(self, n_head: int, n_embd: int ):
        super().__init__()
        
        self.layernorm_1 = nn.LayerNorm(n_embd)
        self.attention = SelfAttention(n_head, n_embd)
        self.layernorm_2 = nn.LayerNorm(n_embd)
        self.linear_1 = nn.Linear(n_embd, 4 * n_embd)
        self.linear_2 = nn.Linear(4 * n_embd, n_embd)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        # (Batch_size, seq_len, dim)
        
        residue = x
        
        #Self Attention
        
        x = self.layernorm_1(x)
        x= self.attention(x, causal_mask=True) 
        x += residue
        
        #FeedForward Layer
        
        residue = x
        x = self.layernorm_2(x)
        x = self.linear_1(x) 
        x = x * torch.sigmoid(1.702 * x) #QuickGELU activation function, which in practice is demonstrated to work better
        x= self.linear_2(x)
        x += residue
        
        return x 
        
        


class CLIP(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.embedding = CLIPEmbedding(49408, 768, 77) #TO-DO:Use some configuration file
                                                        #In this case parameters already fixed by pretraind model
        self.layers = nn.ModuleList([
            CLIPLayer(12, 768) for i in range(12) # 12 is nº of heads in the MHA and the embedding size = 768
        ])
        
        self.layernorm = nn.LayerNorm(768)
        
    def forward(self, tokens: torch.LongTensor) -> torch.FloatTensor:
        tokens = tokens.type(torch.long)
        
        # (Batch_size, seq_len) -> (Batch_size, seq_len, dim=768)
        state = self.embedding(tokens)
        
        for layer in self.layers:
            state = layer(state)
        
        # (Batch_size, seq_len, dim=768)    
        output = self.layernorm(state)
        
        return output