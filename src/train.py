import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from model import SatyaGPT
from tokenizer import train_tokenizer

# Fix FileNotFoundError - Ensure dataset file exists
DATA_PATH = os.path.join(os.path.dirname(__file__), "../data/cleaned_constitution.txt")
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Error: File '{DATA_PATH}' not found. Please check the path.")

# Ensure the model directory exists
MODEL_DIR = os.path.join(os.path.dirname(__file__), "../models")
os.makedirs(MODEL_DIR, exist_ok=True)

# Dataset Class
class ConstitutionDataset(Dataset):
    def __init__(self, text_file, tokenizer, max_length=512):
        with open(text_file, "r", encoding="utf-8") as f:
            self.data = f.readlines()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tokens = self.tokenizer.encode(self.data[idx]).ids  # Tokenize
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]  # Truncate to max_length
        tokens = torch.tensor(tokens, dtype=torch.long)  
        return tokens[:-1], tokens[1:]  # Input and target for next-token prediction

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device == "cpu":
    print("Warning: CUDA is not available. Training will be slower.")

# Load tokenizer and dataset
tokenizer = train_tokenizer(DATA_PATH)
dataset = ConstitutionDataset(DATA_PATH, tokenizer)

# Custom collate function for padding sequences
def collate_fn(batch):
    inputs, targets = zip(*batch)
    inputs = pad_sequence(inputs, batch_first=True, padding_value=0)
    targets = pad_sequence(targets, batch_first=True, padding_value=0)
    return inputs, targets

dataloader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)

# Initialize model, optimizer, loss function
model = SatyaGPT(
    vocab_size=5000, embed_size=256, num_layers=4, 
    heads=8, forward_expansion=4, dropout=0.1, max_length=512
).to(device)

optimizer = optim.AdamW(model.parameters(), lr=3e-4)
criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding tokens

# Check if CUDA is available before using AMP
use_amp = torch.cuda.is_available()
scaler = torch.cuda.amp.GradScaler() if use_amp else None

# Training loop with Auto-Save
epochs = 1000
for epoch in range(epochs):
    model.train()
    total_loss = 0

    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        if use_amp:
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs.view(-1, 5000), targets.view(-1))

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(inputs)
            loss = criterion(outputs.view(-1, 5000), targets.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch [{epoch+1}/{epochs}] Loss: {avg_loss:.4f}")

# Final model save
final_model_path = os.path.join(MODEL_DIR, "trained_model_final.pth")
torch.save(model.state_dict(), final_model_path)
print("Training completed and final model saved.")