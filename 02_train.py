import os
import pickle
import math
import time
import random

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm
import wandb 

from model import MusicTransformer, PositionalEncoding
import config

class MusicDataset(Dataset):
    def __init__(self, tokens, sequence_length):
        self.tokens = tokens
        self.sequence_length = sequence_length
    def __len__(self):
        return len(self.tokens) - self.sequence_length
    def __getitem__(self, idx):
        chunk = self.tokens[idx : idx + self.sequence_length + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y

def validate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validating", leave=False)
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            output = model(x)
            loss = criterion(output.view(-1, config.VOCAB_SIZE), y.view(-1))
            val_loss += loss.item()
            pbar.set_postfix({'val_loss': f"{loss.item():.4f}"})
    return val_loss / len(val_loader)


def main():
    wandb.init(
        project="music-transformer-project",
        config={
            "learning_rate": config.LEARNING_RATE,
            "batch_size": config.BATCH_SIZE,
            "epochs": config.NUM_EPOCHS,
            "sequence_length": config.SEQUENCE_LENGTH,
            "embedding_dim": config.EMBEDDING_DIM,
            "num_heads": config.NUM_HEADS,
            "num_layers": config.NUM_LAYERS,
            "ff_dim": config.FF_DIM,
            "dropout": config.DROPOUT,
            "vocab_size": config.VOCAB_SIZE,
        }
    )
    
    print("--- Обучение ---")

    device = torch.device(config.DEVICE)
    print(f"Информация: Моделът е зареден на {device}")
    models_dir = config.MODEL_SAVE_PATH
    os.makedirs(models_dir, exist_ok=True)

    print(f"Информация: Зареждам данните от {config.TOKEN_DATA_PATH}")
    with open(config.TOKEN_DATA_PATH, "rb") as f:
        data = pickle.load(f)
    
    train_tokens = data['train_tokens']
    val_tokens = data['validation_tokens']

    train_dataset = MusicDataset(train_tokens, config.SEQUENCE_LENGTH)
    full_val_dataset = MusicDataset(val_tokens, config.SEQUENCE_LENGTH)

    g = torch.Generator().manual_seed(42) 
    val_indices = torch.randperm(len(full_val_dataset), generator=g)[:8192].tolist()
    val_subset = Subset(full_val_dataset, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=4)

    model = MusicTransformer(
        vocab_size=config.VOCAB_SIZE,
        embed_dim=config.EMBEDDING_DIM,
        num_heads=config.NUM_HEADS,
        num_layers=config.NUM_LAYERS,
        ff_dim=config.FF_DIM,
        dropout=config.DROPOUT
    ).to(device)
    
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
    
    wandb.watch(model, criterion, log="all", log_freq=100)
    print(f"Информация: Моделът е инициализиран с {sum(p.numel() for p in model.parameters() if p.requires_grad):,} параметъра.")

    global_step = 0
    best_val_loss = float('inf')
    
    for epoch in range(1, config.NUM_EPOCHS + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Епоха {epoch}/{config.NUM_EPOCHS}")
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output.view(-1, config.VOCAB_SIZE), y.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            
            global_step += 1
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

            if global_step % 32 == 0:
                wandb.log({"train_loss": loss.item()}, step=global_step)

            if global_step % 256 == 0:
                val_loss = validate(model, val_loader, criterion, device)
                wandb.log({"val_loss": val_loss}, step=global_step)
                print(f"\nСтъпка {global_step}: Validation грешка: {val_loss:.4f}")
                model.train()
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    checkpoint_path = os.path.join(models_dir, f"best_model_step_{global_step}.pt")
                    torch.save(model.state_dict(), checkpoint_path)
                    print(f"  -> Най-добрия модел запазен в {checkpoint_path}")
                    wandb.log({"best_val_loss": best_val_loss}, step=global_step)

    print("\n--- Обучението завърши. ---")
    wandb.finish()


if __name__ == '__main__':
    main()