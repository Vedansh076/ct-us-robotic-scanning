#!/usr/bin/env python3
"""
train_act.py — Train an Action Chunking with Transformers (ACT) policy.
=============================================================================

Loads pre-collected expert demonstrations (.npz files) and trains a custom
Action Chunking Transformer model with CVAE style latent space.

Usage
-----
    python train_act.py --demos-dir demos/ --epochs 100 --chunk-size 20
"""

import argparse
import os
import sys
import time
import pickle
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent.resolve()))

from train_bc import load_demonstrations


# =============================================================================
# 1. Dataset definition
# =============================================================================

class ACTDataset(Dataset):
    """Dataset for training Action Chunking with Transformers (ACT).

    Prepares sliding-window observations and action chunks of size k.
    """
    def __init__(self, demos_dir, chunk_size=20):
        self.chunk_size = chunk_size
        
        # Load trajectories
        trajectories = load_demonstrations(demos_dir)
        self.samples = []
        
        # Process each trajectory
        for traj in trajectories:
            obs_list = traj.obs
            actions = traj.acts
            T = len(actions)
            
            for t in range(T):
                # Retrieve current observation
                obs_t = obs_list[t]
                image = obs_t["image"]  # (128, 128)
                force = obs_t["force"]  # (1,)
                pose = obs_t["pose"]    # (7,)
                
                # Retrieve future actions of length chunk_size
                action_chunk = actions[t : t + chunk_size]
                
                # Pad actions if window extends past the end of the episode
                if len(action_chunk) < chunk_size:
                    padding_len = chunk_size - len(action_chunk)
                    last_action = actions[T - 1]
                    padding = np.tile(last_action, (padding_len, 1))
                    action_chunk = np.concatenate([action_chunk, padding], axis=0)
                
                self.samples.append({
                    "image": image.astype(np.float32) / 255.0,  # Normalize to [0, 1]
                    "force": force.astype(np.float32),
                    "pose": pose.astype(np.float32),
                    "actions": action_chunk.astype(np.float32),
                })
                
        print(f"  [data] Created ACT dataset with {len(self.samples)} samples (chunk size: {chunk_size}).")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return (
            torch.tensor(sample["image"]).unsqueeze(0), # Add channel dim: (1, 128, 128)
            torch.tensor(sample["force"]),              # (1,)
            torch.tensor(sample["pose"]),               # (7,)
            torch.tensor(sample["actions"]),            # (k, 6)
        )


# =============================================================================
# 2. Model components
# =============================================================================

class ACTVisionEncoder(nn.Module):
    """Custom CNN vision encoder for 128x128 grayscale images."""
    def __init__(self, features_dim=256):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=8, stride=4),   # 16x31x31
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),  # 32x14x14
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1),  # 64x12x12
            nn.ReLU(),
            nn.Flatten(),
        )
        # Compute exact output dimension
        self.fc = nn.Sequential(
            nn.Linear(64 * 12 * 12, features_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.fc(self.cnn(x))


class ACTProprioEncoder(nn.Module):
    """MLP encoder for force and pose inputs."""
    def __init__(self, input_dim=8, features_dim=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, features_dim),
            nn.ReLU(),
        )

    def forward(self, force, pose):
        # Concatenate force (1) and pose (7)
        x = torch.cat([force, pose], dim=1)
        return self.mlp(x)


class ACTCvaeEncoder(nn.Module):
    """CVAE style encoder to predict mu and logvar of style latent space z."""
    def __init__(self, obs_features_dim=320, action_dim=6, chunk_size=20, latent_dim=32):
        super().__init__()
        action_seq_dim = chunk_size * action_dim
        self.mlp = nn.Sequential(
            nn.Linear(obs_features_dim + action_seq_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)

    def forward(self, obs_features, actions):
        # Flatten action sequence: (batch_size, chunk_size, 6) -> (batch_size, chunk_size * 6)
        actions_flat = actions.reshape(actions.size(0), -1)
        x = torch.cat([obs_features, actions_flat], dim=1)
        h = self.mlp(x)
        return self.fc_mu(h), self.fc_logvar(h)


class ActionChunkingTransformer(nn.Module):
    """Action Chunking with Transformers (ACT) model."""
    def __init__(self, chunk_size=20, latent_dim=32, d_model=256, nhead=8, num_encoder_layers=4, num_decoder_layers=2):
        super().__init__()
        self.chunk_size = chunk_size
        self.latent_dim = latent_dim
        self.d_model = d_model
        
        # 1. Feature encoders
        self.vision_encoder = ACTVisionEncoder(features_dim=256)
        self.proprio_encoder = ACTProprioEncoder(input_dim=8, features_dim=64)
        
        # 2. CVAE encoder
        self.cvae_encoder = ACTCvaeEncoder(obs_features_dim=320, action_dim=6, chunk_size=chunk_size, latent_dim=latent_dim)
        
        # 3. Projection heads to d_model
        self.vision_proj = nn.Linear(256, d_model)
        self.proprio_proj = nn.Linear(64, d_model)
        self.latent_proj = nn.Linear(latent_dim, d_model)
        
        # 4. Transformer components
        # Source tokens: latent z, vision, proprio (seq_len = 3)
        self.pos_emb = nn.Parameter(torch.randn(3, 1, d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=1024, activation='relu', batch_first=False)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        
        # Action queries for decoder (seq_len = chunk_size)
        self.query_emb = nn.Parameter(torch.randn(chunk_size, 1, d_model))
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=1024, activation='relu', batch_first=False)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        
        # 5. Output action head
        self.action_head = nn.Linear(d_model, 6)

    def forward(self, image, force, pose, target_actions=None):
        batch_size = image.size(0)
        
        # Extract features
        vis_feats = self.vision_encoder(image)      # (batch_size, 256)
        prop_feats = self.proprio_encoder(force, pose)  # (batch_size, 64)
        obs_feats = torch.cat([vis_feats, prop_feats], dim=1) # (batch_size, 320)
        
        # CVAE sampling
        if self.training:
            assert target_actions is not None, "Target actions required during training for CVAE"
            mu, logvar = self.cvae_encoder(obs_feats, target_actions)
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std
        else:
            # Deterministic inference: set z to prior mean
            z = torch.zeros(batch_size, self.latent_dim, device=image.device)
            mu, logvar = None, None
            
        # Project inputs to d_model
        vis_proj = self.vision_proj(vis_feats)       # (batch_size, d_model)
        prop_proj = self.proprio_proj(prop_feats)   # (batch_size, d_model)
        z_proj = self.latent_proj(z)                 # (batch_size, d_model)
        
        # Construct source sequence for Transformer Encoder: seq_len = 3
        # PyTorch Transformer defaults to (seq_len, batch_size, d_model)
        src = torch.stack([z_proj, vis_proj, prop_proj], dim=0) # (3, batch, d_model)
        src = src + self.pos_emb # Add positional embeddings
        
        # Encode
        memory = self.transformer_encoder(src) # (3, batch, d_model)
        
        # Construct target sequence (queries) for Transformer Decoder: seq_len = chunk_size
        # Repeat query embeddings across batch
        tgt = self.query_emb.repeat(1, batch_size, 1) # (chunk_size, batch, d_model)
        
        # Decode
        decoded = self.transformer_decoder(tgt, memory) # (chunk_size, batch, d_model)
        
        # Project to action dimension (6)
        pred_actions = self.action_head(decoded) # (chunk_size, batch, 6)
        
        # Transpose back to batch-first: (batch_size, chunk_size, 6)
        pred_actions = pred_actions.transpose(0, 1)
        
        return pred_actions, mu, logvar


# =============================================================================
# 3. Training logic
# =============================================================================

def train_one_epoch(model, dataloader, optimizer, kl_weight, device):
    model.train()
    total_loss = 0.0
    total_l1 = 0.0
    total_kl = 0.0
    
    for image, force, pose, target_actions in dataloader:
        image = image.to(device)
        force = force.to(device)
        pose = pose.to(device)
        target_actions = target_actions.to(device)
        
        optimizer.zero_grad()
        
        pred_actions, mu, logvar = model(image, force, pose, target_actions)
        
        # Reconstruction loss (MAE / L1 loss)
        l1_loss = F.l1_loss(pred_actions, target_actions, reduction='mean')
        
        # KL divergence loss
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        
        # Combined loss
        loss = l1_loss + kl_weight * kl_loss
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * image.size(0)
        total_l1 += l1_loss.item() * image.size(0)
        total_kl += kl_loss.item() * image.size(0)
        
    n_samples = len(dataloader.dataset)
    return total_loss / n_samples, total_l1 / n_samples, total_kl / n_samples


@torch.no_grad()
def evaluate_model(model, dataloader, kl_weight, device):
    model.eval()
    total_loss = 0.0
    total_l1 = 0.0
    
    for image, force, pose, target_actions in dataloader:
        image = image.to(device)
        force = force.to(device)
        pose = pose.to(device)
        target_actions = target_actions.to(device)
        
        # Forward pass in eval mode (z = 0)
        pred_actions, _, _ = model(image, force, pose)
        
        l1_loss = F.l1_loss(pred_actions, target_actions, reduction='mean')
        
        total_loss += l1_loss.item() * image.size(0)
        total_l1 += l1_loss.item() * image.size(0)
        
    n_samples = len(dataloader.dataset)
    return total_loss / n_samples, total_l1 / n_samples


def main():
    parser = argparse.ArgumentParser(description="Train Action Chunking with Transformers (ACT)")
    parser.add_argument("--demos-dir", type=str, required=True, help="Directory containing trajectory_*.npz files")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--kl-weight", type=float, default=10.0, help="Weight factor for KL loss")
    parser.add_argument("--chunk-size", type=int, default=20, help="Action chunk sequence size (k)")
    parser.add_argument("--latent-dim", type=int, default=32, help="CVAE latent space z dimension")
    parser.add_argument("--save-dir", type=str, default="act_checkpoints", help="Save directory")
    parser.add_argument("--device", type=str, default="auto", help="auto, cpu, or cuda")
    args = parser.parse_args()

    print("=" * 60)
    print("  Action Chunking with Transformers (ACT) — Training")
    print("=" * 60)
    print(f"  Demos dir:   {args.demos_dir}")
    print(f"  Epochs:      {args.epochs}")
    print(f"  Batch size:  {args.batch_size}")
    print(f"  LR:          {args.lr}")
    print(f"  KL weight:   {args.kl_weight}")
    print(f"  Chunk size:  {args.chunk_size}")
    print(f"  Save dir:    {args.save_dir}")
    print("=" * 60)

    # 1. Create dataset and split
    print("\n[data] Loading dataset...")
    full_dataset = ACTDataset(args.demos_dir, chunk_size=args.chunk_size)
    
    # Simple 90/10 split
    val_size = int(0.1 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # 2. Setup device & model
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"\n[device] Using training device: {device}")
    
    model = ActionChunkingTransformer(
        chunk_size=args.chunk_size,
        latent_dim=args.latent_dim,
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # 3. Training Loop
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    best_val_loss = float('inf')
    print("\n[train] Starting training...")
    
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_l1, train_kl = train_one_epoch(model, train_loader, optimizer, args.kl_weight, device)
        val_loss, val_l1 = evaluate_model(model, val_loader, args.kl_weight, device)
        dt = time.time() - t0
        
        print(f"Epoch {epoch:3d}/{args.epochs:3d} | Loss: {train_loss:.4f} (L1: {train_l1:.4f}, KL: {train_kl:.4f}) | "
              f"Val L1: {val_l1:.4f} | Time: {dt:.1f}s")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Save state dict
            torch.save(model.state_dict(), save_dir / "act_best_state.pth")
            # Save full model (including architecture config)
            torch.save(model, save_dir / "act_policy.zip")
            print(f"  [OK] Saved new best checkpoint to {save_dir / 'act_policy.zip'}")

    print("\n[train] Training complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
