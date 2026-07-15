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
        
        from imitation.data.types import maybe_unwrap_dictobs
        
        # Process each trajectory
        for traj in trajectories:
            obs_dict = maybe_unwrap_dictobs(traj.obs)
            images = obs_dict["image"]
            forces = obs_dict["force"]
            poses = obs_dict["pose"]
            actions = traj.acts
            T = len(actions)
            
            for t in range(T):
                # Retrieve current observation elements from arrays
                image = images[t]  # (128, 128)
                force = forces[t]  # (1,)
                pose = poses[t]    # (7,)
                
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
                
        # Compute dataset normalization statistics (mean and std)
        all_forces = np.array([s["force"] for s in self.samples])
        all_poses = np.array([s["pose"] for s in self.samples])
        all_actions = np.array([s["actions"] for s in self.samples])  # (N, chunk_size, 6)
        
        self.stats = {
            "force_mean": np.mean(all_forces, axis=0, keepdims=True),
            "force_std": np.std(all_forces, axis=0, keepdims=True) + 1e-6,
            "pose_mean": np.mean(all_poses, axis=0, keepdims=True),
            "pose_std": np.std(all_poses, axis=0, keepdims=True) + 1e-6,
            "action_mean": np.mean(all_actions, axis=(0, 1), keepdims=True),  # (1, 1, 6)
            "action_std": np.std(all_actions, axis=(0, 1), keepdims=True) + 1e-6,
        }
        print(f"  [data] Created ACT dataset with {len(self.samples)} samples (chunk size: {chunk_size}).")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        force_norm = (sample["force"] - self.stats["force_mean"][0]) / self.stats["force_std"][0]
        pose_norm = (sample["pose"] - self.stats["pose_mean"][0]) / self.stats["pose_std"][0]
        actions_norm = (sample["actions"] - self.stats["action_mean"][0]) / self.stats["action_std"][0]
        
        return (
            torch.tensor(sample["image"]).unsqueeze(0), # Add channel dim: (1, 128, 128)
            torch.tensor(force_norm, dtype=torch.float32),
            torch.tensor(pose_norm, dtype=torch.float32),
            torch.tensor(actions_norm, dtype=torch.float32),
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
    """Action Chunking with Transformers (ACT) model.
    
    Supports both:
      1. Standard ACT: CVAE encoder + Transformer Encoder + Transformer Decoder.
      2. ACT-Lite (no_cvae=True): Bypasses CVAE & Transformer Encoder. Uses a
         simplified Transformer Decoder cross-attending to a single combined
         observation feature vector. Recommended for fast convergence.
    """
    def __init__(self, chunk_size=20, latent_dim=32, d_model=256, nhead=8, num_encoder_layers=4, num_decoder_layers=2, no_cvae=False):
        super().__init__()
        self.chunk_size = chunk_size
        self.latent_dim = latent_dim
        self.d_model = d_model
        self.use_cvae = not no_cvae
        
        # 1. Feature encoders
        self.vision_encoder = ACTVisionEncoder(features_dim=256)
        self.proprio_encoder = ACTProprioEncoder(input_dim=8, features_dim=64)
        
        if self.use_cvae:
            # 2. CVAE components (Standard ACT)
            self.cvae_encoder = ACTCvaeEncoder(obs_features_dim=320, action_dim=6, chunk_size=chunk_size, latent_dim=latent_dim)
            self.vision_proj = nn.Linear(256, d_model)
            self.proprio_proj = nn.Linear(64, d_model)
            self.latent_proj = nn.Linear(latent_dim, d_model)
            # Positional embeddings for encoder: seq_len = 3 (z, vision, proprio)
            self.pos_emb = nn.Parameter(torch.randn(3, 1, d_model))
            encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=1024, activation='relu', batch_first=False)
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        else:
            # 2. ACT-Lite components (CVAE-less, Encoder-less)
            self.cvae_encoder = None
            self.transformer_encoder = None
            self.obs_proj = nn.Sequential(
                nn.Linear(320, d_model),
                nn.LayerNorm(d_model),
                nn.ReLU(),
            )
        
        # 3. Transformer Decoder
        self.query_emb = nn.Parameter(torch.randn(chunk_size, 1, d_model))
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=512, activation='relu', batch_first=False)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        
        # 4. Output action head
        self.action_head = nn.Linear(d_model, 6)

    def forward(self, image, force, pose, target_actions=None):
        batch_size = image.size(0)
        
        # Extract features
        vis_feats = self.vision_encoder(image)      # (batch_size, 256)
        prop_feats = self.proprio_encoder(force, pose)  # (batch_size, 64)
        
        if self.use_cvae:
            # --- Standard ACT path ---
            obs_feats = torch.cat([vis_feats, prop_feats], dim=1) # (batch_size, 320)
            if self.training:
                assert target_actions is not None, "Target actions required during training for CVAE"
                mu, logvar = self.cvae_encoder(obs_feats, target_actions)
                std = torch.exp(0.5 * logvar)
                eps = torch.randn_like(std)
                z = mu + eps * std
            else:
                z = torch.zeros(batch_size, self.latent_dim, device=image.device)
                mu, logvar = None, None
            
            # Project tokens to d_model
            vis_proj = self.vision_proj(vis_feats)
            prop_proj = self.proprio_proj(prop_feats)
            z_proj = self.latent_proj(z)
            
            # Encoder source sequence
            src = torch.stack([z_proj, vis_proj, prop_proj], dim=0) # (3, batch, d_model)
            src = src + self.pos_emb
            memory = self.transformer_encoder(src) # (3, batch, d_model)
        else:
            # --- ACT-Lite path (CVAE-less, Encoder-less) ---
            obs_feats = torch.cat([vis_feats, prop_feats], dim=1) # (batch_size, 320)
            # Directly project state to memory of sequence length 1
            memory = self.obs_proj(obs_feats).unsqueeze(0)        # (1, batch, d_model)
            mu, logvar = None, None
            
        # Transformer Decoder query construction
        tgt = self.query_emb.repeat(1, batch_size, 1) # (chunk_size, batch, d_model)
        
        # Decode
        decoded = self.transformer_decoder(tgt, memory) # (chunk_size, batch, d_model)
        
        # Project to action dimension (6)
        pred_actions = self.action_head(decoded) # (chunk_size, batch, 6)
        pred_actions = pred_actions.transpose(0, 1) # (batch_size, chunk_size, 6)
        
        return pred_actions, mu, logvar


class MlpActionChunkingPolicy(nn.Module):
    """ACT-MLP: Simplified MLP-based Action Chunking Policy.
    
    Bypasses all Transformer attention mechanisms, mapping observations
    directly to future action chunks. Highly recommended for small datasets.
    """
    def __init__(self, chunk_size=20):
        super().__init__()
        self.chunk_size = chunk_size
        self.use_cvae = False # Compatibility flag
        
        # 1. Feature encoders
        self.vision_encoder = ACTVisionEncoder(features_dim=256)
        self.proprio_encoder = ACTProprioEncoder(input_dim=8, features_dim=64)
        
        # 2. MLP Policy Head
        self.policy_head = nn.Sequential(
            nn.Linear(320, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, chunk_size * 6)
        )

    def forward(self, image, force, pose, target_actions=None):
        batch_size = image.size(0)
        
        # Extract features
        vis_feats = self.vision_encoder(image)      # (batch_size, 256)
        prop_feats = self.proprio_encoder(force, pose)  # (batch_size, 64)
        
        # Concatenate features
        obs_feats = torch.cat([vis_feats, prop_feats], dim=1) # (batch_size, 320)
        
        # Direct projection
        out = self.policy_head(obs_feats) # (batch_size, chunk_size * 6)
        pred_actions = out.reshape(batch_size, self.chunk_size, 6)
        
        return pred_actions, None, None


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
        
        # KL divergence loss (only if CVAE is enabled)
        if mu is not None and logvar is not None:
            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        else:
            kl_loss = torch.tensor(0.0, device=device)
        
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
    parser.add_argument("--no-cvae", action="store_true", help="Bypass CVAE and train a deterministic Action Chunking Transformer")
    parser.add_argument("--mlp", action="store_true", help="Use simplified MLP-based Action Chunking Policy instead of Transformer")
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
    if args.mlp:
        print("  Architecture: ACT-MLP (Simplified MLP Policy)")
    elif args.no_cvae:
        print("  Architecture: ACT-Lite (Encoder-less Transformer)")
        print(f"  CVAE mode:   Disabled")
    else:
        print("  Architecture: Standard ACT (CVAE + Transformer)")
        print(f"  CVAE mode:   Enabled")
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
    
    if args.mlp:
        model = MlpActionChunkingPolicy(chunk_size=args.chunk_size).to(device)
    else:
        model = ActionChunkingTransformer(
            chunk_size=args.chunk_size,
            latent_dim=args.latent_dim,
            no_cvae=args.no_cvae,
        ).to(device)
    
    # Use standard Adam optimizer with no weight decay (matching SB3 default)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

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
            # Save dataset normalization stats
            with open(save_dir / "norm_stats.pkl", "wb") as f:
                pickle.dump(full_dataset.stats, f)
            print(f"  [OK] Saved new best checkpoint and norm_stats to {save_dir / 'act_policy.zip'}")

    print("\n[train] Training complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
