"""
Usage:
    python3 -m homework.train_planner --model_name mlp_planner --num_epochs 50
    python3 -m homework.train_planner --model_name transformer_planner --num_epochs 80
    python3 -m homework.train_planner --model_name vit_planner --num_epochs 80
"""

import argparse
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from homework.datasets.road_dataset import load_data
from homework.metrics import PlannerMetric
from homework.models import load_model, save_model


def train_planner(
    model_name: str = "mlp_planner",
    transform_pipeline: str = "default",
    num_epochs: int = 50,
    batch_size: int = 128,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    num_workers: int = 4,
    log_dir: str = None,
):
    """
    Train a planner model.
    
    Args:
        model_name: Name of the model to train (mlp_planner, transformer_planner, vit_planner)
        transform_pipeline: Data transformation pipeline
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        lr: Learning rate
        weight_decay: Weight decay for optimizer
        num_workers: Number of data loading workers
        log_dir: Directory for tensorboard logs
    """
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Determine which transform pipeline to use
    if model_name in ["mlp_planner", "transformer_planner"]:
        # These models don't need images, only track boundaries
        transform_pipeline = "state_only"
    else:
        # ViT needs images
        transform_pipeline = "default"
    
    # Load data
    print("Loading training data...")
    train_loader = load_data(
        dataset_path="drive_data/train",
        transform_pipeline=transform_pipeline,
        return_dataloader=True,
        num_workers=num_workers,
        batch_size=batch_size,
        shuffle=True,
    )
    
    print("Loading validation data...")
    val_loader = load_data(
        dataset_path="drive_data/val",
        transform_pipeline=transform_pipeline,
        return_dataloader=True,
        num_workers=num_workers,
        batch_size=batch_size,
        shuffle=False,
    )
    
    # Create model
    print(f"Creating model: {model_name}")
    model = load_model(model_name, with_weights=False)
    model = model.to(device)
    
    # Print model size
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model has {num_params:,} trainable parameters")
    
    # Create optimizer
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Loss function (MSE for regression)
    criterion = nn.MSELoss()
    
    # Metrics
    train_metric = PlannerMetric()
    val_metric = PlannerMetric()
    
    # Tensorboard
    if log_dir is None:
        log_dir = Path("logs") / model_name / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    writer = SummaryWriter(log_dir)
    
    # Training loop
    best_val_loss = float('inf')
    global_step = 0
    
    print(f"\nStarting training for {num_epochs} epochs...")
    print("=" * 80)
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_metric.reset()
        train_loss_sum = 0.0
        
        for batch_idx, batch in enumerate(train_loader):
            # Move data to device
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device)
            
            # Forward pass
            if model_name in ["mlp_planner", "transformer_planner"]:
                pred_waypoints = model(
                    track_left=batch["track_left"],
                    track_right=batch["track_right"]
                )
            else:  # vit_planner
                pred_waypoints = model(image=batch["image"])
            
            # Compute loss (only on valid waypoints)
            target_waypoints = batch["waypoints"]
            waypoints_mask = batch["waypoints_mask"]
            
            # Masked loss
            loss = criterion(
                pred_waypoints * waypoints_mask[..., None],
                target_waypoints * waypoints_mask[..., None]
            )
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Track metrics
            train_loss_sum += loss.item()
            train_metric.add(pred_waypoints, target_waypoints, waypoints_mask)
            
            # Log to tensorboard
            writer.add_scalar("train/loss_step", loss.item(), global_step)
            global_step += 1
            
            # Print progress
            if (batch_idx + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}] Batch [{batch_idx+1}/{len(train_loader)}] Loss: {loss.item():.6f}")
        
        # Compute training metrics
        train_loss_avg = train_loss_sum / len(train_loader)
        train_metrics = train_metric.compute()
        
        # Validation phase
        model.eval()
        val_metric.reset()
        val_loss_sum = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                # Move data to device
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(device)
                
                # Forward pass
                if model_name in ["mlp_planner", "transformer_planner"]:
                    pred_waypoints = model(
                        track_left=batch["track_left"],
                        track_right=batch["track_right"]
                    )
                else:  # vit_planner
                    pred_waypoints = model(image=batch["image"])
                
                # Compute loss
                target_waypoints = batch["waypoints"]
                waypoints_mask = batch["waypoints_mask"]
                
                loss = criterion(
                    pred_waypoints * waypoints_mask[..., None],
                    target_waypoints * waypoints_mask[..., None]
                )
                
                val_loss_sum += loss.item()
                val_metric.add(pred_waypoints, target_waypoints, waypoints_mask)
        
        # Compute validation metrics
        val_loss_avg = val_loss_sum / len(val_loader)
        val_metrics = val_metric.compute()
        
        # Update learning rate
        scheduler.step(val_loss_avg)
        
        # Log to tensorboard
        writer.add_scalar("train/loss_epoch", train_loss_avg, epoch)
        writer.add_scalar("train/longitudinal_error", train_metrics["longitudinal_error"], epoch)
        writer.add_scalar("train/lateral_error", train_metrics["lateral_error"], epoch)
        writer.add_scalar("val/loss_epoch", val_loss_avg, epoch)
        writer.add_scalar("val/longitudinal_error", val_metrics["longitudinal_error"], epoch)
        writer.add_scalar("val/lateral_error", val_metrics["lateral_error"], epoch)
        writer.add_scalar("lr", optimizer.param_groups[0]['lr'], epoch)
        
        # Print epoch summary
        print("=" * 80)
        print(f"Epoch [{epoch+1}/{num_epochs}] Summary:")
        print(f"  Train Loss: {train_loss_avg:.6f} | Long Error: {train_metrics['longitudinal_error']:.4f} | Lat Error: {train_metrics['lateral_error']:.4f}")
        print(f"  Val Loss:   {val_loss_avg:.6f} | Long Error: {val_metrics['longitudinal_error']:.4f} | Lat Error: {val_metrics['lateral_error']:.4f}")
        print("=" * 80)
        
        # Save best model
        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            model_path = save_model(model)
            print(f"✓ Saved best model to {model_path}")
            print(f"  Best val loss: {best_val_loss:.6f}")
    
    print("\nTraining completed!")
    print(f"Best validation loss: {best_val_loss:.6f}")
    
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a planner model")
    
    parser.add_argument(
        "--model_name",
        type=str,
        default="mlp_planner",
        choices=["mlp_planner", "transformer_planner", "vit_planner"],
        help="Name of the model to train"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=50,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size for training"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate"
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
        help="Weight decay for optimizer"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of data loading workers"
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default=None,
        help="Directory for tensorboard logs"
    )
    
    args = parser.parse_args()
    
    train_planner(
        model_name=args.model_name,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        log_dir=args.log_dir,
    )
