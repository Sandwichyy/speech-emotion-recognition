import torch
import torch.nn as nn
import torch.optim as optim
import os


class SERModel(nn.Module):
    def __init__(self, d_model, nhead, hidden_dim, num_layers, num_classes=6, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        
        self.positional = nn.Parameter(torch.randn(1, 512, d_model))

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=hidden_dim, dropout=dropout, batch_first=True) # [batch_size, seq_length, d_model]
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.classifier = nn.Sequential(
            nn.Linear(d_model, hidden_dim), 
            nn.ReLU(), 
            nn.Dropout(dropout), 
            nn.Linear(hidden_dim, num_classes))

    def forward(self, x, padding_mask=None):
        batch_size, seq_length, _ = x.size()

        x = x + self.positional[:, :seq_length, :]

        x = self.transformer(x, src_key_padding_mask=padding_mask)

        if padding_mask is not None:
            mask = ~padding_mask
            mask = mask.unsqueeze(-1)
            x = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            x = x.mean(dim=1)

        output = self.classifier(x) 
        return output

class SERModelOneLL(nn.Module):
    def __init__(self, d_model, nhead, hidden_dim, num_layers, num_classes=6, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        
        self.positional = nn.Parameter(torch.randn(1, 512, d_model))

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=hidden_dim, dropout=dropout, batch_first=True) # [batch_size, seq_length, d_model]
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x, padding_mask=None):
        batch_size, seq_length, _ = x.size()

        x = x + self.positional[:, :seq_length, :]

        x = self.transformer(x, src_key_padding_mask=padding_mask)

        if padding_mask is not None:
            mask = ~padding_mask
            mask = mask.unsqueeze(-1)
            x = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            x = x.mean(dim=1)

        output = self.classifier(x) 
        return output







    
