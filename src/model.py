import torch
import torch.nn as nn
import torch.optim as optim

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout):
        super(TransformerBlock, self).__init__()
        
        self.attention = nn.MultiheadAttention(embed_dim=embed_size, num_heads=heads, dropout=dropout, batch_first=True)
        
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attention_out, _ = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attention_out))
        
        feed_forward_out = self.feed_forward(x)
        x = self.norm2(x + self.dropout(feed_forward_out))

        return x

class SatyaGPT(nn.Module):
    def __init__(self, vocab_size, embed_size, num_layers, heads, forward_expansion, dropout, max_length):
        super(SatyaGPT, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.position_encoding = nn.Parameter(torch.randn(1, max_length, embed_size))  # Learnable Positional Encoding

        self.layers = nn.ModuleList([
            TransformerBlock(embed_size, heads, forward_expansion, dropout) for _ in range(num_layers)
        ])
        
        self.fc_out = nn.Linear(embed_size, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size, seq_length = x.shape
        embeddings = self.embedding(x)

        # Ensure positional encoding matches input size
        position_encoding = self.position_encoding[:, :seq_length, :]
        embeddings = embeddings + position_encoding
        embeddings = self.dropout(embeddings)

        for layer in self.layers:
            embeddings = layer(embeddings)

        logits = self.fc_out(embeddings)
        return logits

if __name__ == "__main__":
    model = SatyaGPT(
        vocab_size=5000, embed_size=256, num_layers=4, heads=8, 
        forward_expansion=4, dropout=0.1, max_length=512
    )
    
    print(model)