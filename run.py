import os
import random
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from pathlib import Path
from transformers import AutoImageProcessor, AutoModel

# =========================
# 0. Seed & Device
# =========================
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(42)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# DINOv3 model ID (ViT-L, balanced performance/speed)
# Lighter  : "facebook/dinov3-vitb16-pretrain-lvd1689m"
# Stronger : "facebook/dinov3-vith16plus-pretrain-lvd1689m"
DINOV3_MODEL_ID = "facebook/dinov3-vitl16-pretrain-lvd1689m"

# =========================
# 1. Model Architecture
# =========================
class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE).
    Applied to the CLS token sequence to encode temporal order across frames.
    Fixed (no learnable parameters) -- generalizes beyond training length.
    """
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x):
        # x: (B, T, dim) -- sequence of CLS tokens across frames
        T = x.shape[1]
        t = torch.arange(T, device=x.device).float()
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)  # (T, dim/2)
        emb = torch.cat([freqs, freqs], dim=-1)             # (T, dim)
        cos = emb.cos()[None, :, :]                         # (1, T, dim)
        sin = emb.sin()[None, :, :]                         # (1, T, dim)
        return self._rotate(x, cos, sin)

    def _rotate(self, x, cos, sin):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        rotated = torch.cat([-x2, x1], dim=-1)
        return x * cos + rotated * sin


class VisionEncoder(nn.Module):
    """
    Extracts CLS token from each frame via frozen DINOv3 backbone.
    Projects 1024 -> proj_dim, then applies RoPE to encode temporal order.

    Input  : pixel_values (B, T, C, H, W)
    Output : (B, T, proj_dim)
    """
    def __init__(self, dino_model, proj_dim=256):
        super().__init__()
        self.backbone = dino_model
        # Freeze backbone parameters
        for p in self.backbone.parameters():
            p.requires_grad = False
        # DINOv3 ViT-L hidden_size = 1024
        hidden_size = dino_model.config.hidden_size
        # Project 1024 -> proj_dim
        self.proj = nn.Linear(hidden_size, proj_dim)
        self.norm = nn.LayerNorm(proj_dim)
        # RoPE applied to CLS token sequence for temporal position encoding
        self.rope = RotaryEmbedding(proj_dim)

    def forward(self, pixel_values):
        B, T, C, H, W = pixel_values.shape
        # Flatten frame dimension into batch -> (B*T, C, H, W)
        flat = pixel_values.view(B * T, C, H, W).to(DEVICE)
        self.backbone.eval()
        with torch.no_grad():
            out = self.backbone(pixel_values=flat)
            # index 0 = CLS token
            # index 1~4 = register tokens, 5~ = patch tokens
            cls = out.last_hidden_state[:, 0, :]  # (B*T, hidden_size)
        cls = cls.view(B, T, -1)                  # (B, T, hidden_size)
        cls = self.norm(self.proj(cls))            # (B, T, proj_dim)
        # Apply RoPE to CLS token sequence to inject temporal order
        return self.rope(cls)                      # (B, T, proj_dim)


class TextEncoder(nn.Module):
    """
    Lightweight Transformer-based text encoder (2 layers).

    Input  : token id sequence (B, L)
    Output : (B, L, dim)
    """
    def __init__(self, vocab_size, dim=256, layers=2):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb   = nn.Parameter(torch.zeros(1, 50, dim))
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=dim, nhead=8, batch_first=True, norm_first=True
            )
            for _ in range(layers)
        ])
        self.norm = nn.LayerNorm(dim)

    def forward(self, txt_ids):
        x = self.token_emb(txt_ids)
        x = x + self.pos_emb[:, :x.size(1), :]
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)


class Reasoner(nn.Module):
    """
    Fuses vision features (CLS sequence with RoPE) + text features,
    then predicts logits over text positions (3 layers).

    Input:
        v_feat : (B, T, dim)  <- DINOv3 CLS tokens with RoPE
        t_feat : (B, L, dim)  <- text token sequence (L = 5)
    Output:
        logits : (B, L, vocab_size)
    """
    def __init__(self, dim=256, layers=3, heads=8, vocab_size=35):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=dim, nhead=heads, batch_first=True, norm_first=True
            )
            for _ in range(layers)
        ])
        self.lm_head = nn.Linear(dim, vocab_size)

    def forward(self, v_feat, t_feat):
        # Concat vision (CLS sequence with RoPE) + text
        x = torch.cat([v_feat, t_feat], dim=1)  # (B, T+L, dim)
        for blk in self.layers:
            x = blk(x)
        # Predict only the last L tokens (text positions)
        return self.lm_head(x[:, -t_feat.size(1):, :])  # (B, L, vocab_size)


class FullModel(nn.Module):
    def __init__(self, dino_model, vocab_size, proj_dim=256):
        super().__init__()
        self.vision_encoder = VisionEncoder(dino_model, proj_dim=proj_dim)
        self.text_encoder   = TextEncoder(vocab_size, dim=proj_dim)
        self.reasoner       = Reasoner(dim=proj_dim, vocab_size=vocab_size)

    def forward(self, pixel_values, prompt_ids):
        v_feat = self.vision_encoder(pixel_values)  # (B, T, proj_dim)
        t_feat = self.text_encoder(prompt_ids)       # (B, L, proj_dim)
        return self.reasoner(v_feat, t_feat)          # (B, L, vocab_size)


# =========================
# 2. Dataset
# =========================
class VideoDataset(Dataset):
    """
    Frame-level Dataset for DINOv3.

    Reads JPEG frames from video directories,
    preprocesses with AutoImageProcessor -> (T, C, H, W) tensor.

    QA structure:
        Input  : ["what", "is", "it", "?", "<pad>"]
        Target : ["it", "is", "a", <class_name>, "."]
    """
    def __init__(self, root_dir, split, image_processor, num_frames=32):
        self.split_path = Path(root_dir) / split
        self.sample_dirs = sorted([
            d for d in self.split_path.rglob("*")
            if d.is_dir() and any(d.glob("*.jpg"))
        ])
        train_path = Path(root_dir) / "train"
        self.classes = sorted([d.name for d in train_path.iterdir() if d.is_dir()])
        self.itos = ["<pad>", "what", "is", "it", "?", "a", "."] + self.classes
        self.itos = list(dict.fromkeys(self.itos))
        self.stoi = {w: i for i, w in enumerate(self.itos)}
        self.image_processor = image_processor
        self.num_frames      = num_frames

    def __len__(self):
        return len(self.sample_dirs)

    def __getitem__(self, idx):
        s_dir = self.sample_dirs[idx]
        frame_files = sorted(list(s_dir.glob("*.jpg")))[:self.num_frames]
        # Load frames (BGR -> RGB)
        frames_rgb = [
            cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB)
            for f in frame_files
        ]
        # DINOv3 ImageProcessor: batch process frame list
        processed    = self.image_processor(images=frames_rgb, return_tensors="pt")
        pixel_values = processed['pixel_values']  # (T, C, H, W)
        # Build label tokens
        label_name = s_dir.parent.name
        in_tokens  = ["what", "is", "it", "?", "<pad>"]
        gt_tokens  = ["it",   "is", "a",  label_name, "."]
        in_ids = torch.tensor([self.stoi[t] for t in in_tokens], dtype=torch.long)
        gt_ids = torch.tensor([self.stoi[t] for t in gt_tokens], dtype=torch.long)
        return pixel_values, in_ids, gt_ids


# =========================
# 3. Main
# =========================
def main():
    DATASET_PATH = "video_dataset_frames"
    NUM_FRAMES   = 32
    BATCH_SIZE   = 4
    EPOCHS       = 10
    LR           = 1e-4
    PROJ_DIM     = 256

    print(f"[*] Device: {DEVICE}")
    print(f"[*] Loading DINOv3 model: {DINOV3_MODEL_ID}")

    # Load DINOv3 (requires HuggingFace access approval)
    image_processor = AutoImageProcessor.from_pretrained(DINOV3_MODEL_ID)
    dino_model      = AutoModel.from_pretrained(DINOV3_MODEL_ID).to(DEVICE)

    train_ds = VideoDataset(DATASET_PATH, "train", image_processor, num_frames=NUM_FRAMES)
    test_ds  = VideoDataset(DATASET_PATH, "test",  image_processor, num_frames=NUM_FRAMES)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=1,          shuffle=False, num_workers=2, pin_memory=True)

    print("\n" + "="*60)
    print("🔍 [Data Check] DINOv3 frame sequence structure")
    pv, sample_in, sample_gt = train_ds[0]
    print(f"▶ pixel_values shape  : {pv.shape}")
    print(f"▶ Input  (Prompt)     : {[train_ds.itos[i] for i in sample_in.tolist()]}")
    print(f"▶ Target (Answer)     : {[train_ds.itos[i] for i in sample_gt.tolist()]}")
    print(f"▶ Vocab size          : {len(train_ds.itos)}")
    print(f"▶ # Train samples     : {len(train_ds)}")
    print(f"▶ # Test  samples     : {len(test_ds)}")
    print("="*60 + "\n")

    # =========================
    # Training example visualization
    # =========================
    print("\n📷 [Train Examples] Visualizing 1-shot training samples per class...\n")
    class_samples = {}
    for idx in range(len(train_ds)):
        s_dir = train_ds.sample_dirs[idx]
        label = s_dir.parent.name
        if label not in class_samples:
            class_samples[label] = idx
        if len(class_samples) == len(train_ds.classes):
            break

    num_classes = len(class_samples)
    fig, axes = plt.subplots(1, num_classes, figsize=(4 * num_classes, 5))
    if num_classes == 1:
        axes = [axes]

    for ax, (label, idx) in zip(axes, class_samples.items()):
        pixel_values, in_ids, gt_ids = train_ds[idx]
        mid = pixel_values.shape[0] // 2
        img = pixel_values[mid]
        img_np = img.permute(1, 2, 0).numpy()
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
        img_np = img_np.clip(0, 1)
        in_str = " ".join([train_ds.itos[i] for i in in_ids.tolist()])
        gt_str = " ".join([train_ds.itos[i] for i in gt_ids.tolist()])
        ax.imshow(img_np)
        ax.axis('off')
        ax.set_title(
            f"Class : {label}\n\nQ :\n{in_str}\n\nA :\n{gt_str}",
            fontsize=8, loc='left', pad=8
        )

    plt.suptitle("1-Shot Training Examples (1 video per class)", fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.subplots_adjust(top=0.75)
    plt.show()

    # Model / Optimizer / Loss
    model     = FullModel(dino_model, len(train_ds.itos), proj_dim=PROJ_DIM).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[*] Total params    : {total_params:,}")
    print(f"[*] Trainable params: {trainable_params:,}")
    print(f"[*] Video 1-Shot + RoPE (dim={PROJ_DIM}) Training Started (Epochs={EPOCHS})\n")

    # =========================
    # Training loop
    # =========================
    for epoch in range(EPOCHS):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

        for pixel_values, in_ids, gt_ids in pbar:
            pixel_values = pixel_values.to(DEVICE)
            in_ids       = in_ids.to(DEVICE)
            gt_ids       = gt_ids.to(DEVICE)
            optimizer.zero_grad()
            logits = model(pixel_values, in_ids)
            loss   = criterion(
                logits.view(-1, logits.size(-1)),
                gt_ids.view(-1)
            )
            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        # =========================
        # Evaluation loop
        # =========================
        model.eval()
        correct, samples = 0, []

        with torch.no_grad():
            for pixel_values, in_ids, gt_ids in tqdm(test_loader, desc="Testing"):
                pixel_values = pixel_values.to(DEVICE)
                in_ids       = in_ids.to(DEVICE)
                gt_ids       = gt_ids.to(DEVICE)
                logits   = model(pixel_values, in_ids)
                pred_ids = logits[0].argmax(dim=-1).cpu().tolist()
                p_sent = " ".join([test_ds.itos[i] for i in pred_ids])
                g_sent = " ".join([test_ds.itos[i] for i in gt_ids[0].tolist()])
                if p_sent == g_sent:
                    correct += 1
                if len(samples) < 63:
                    # Extract first frame for visualization
                    vis_img = pixel_values[0, 0, :, :, :].cpu()
                    samples.append((vis_img, p_sent, g_sent))

        accuracy = (correct / len(test_ds)) * 100
        print(f"\n📊 [Epoch {epoch+1}] Strict Sentence Accuracy: {accuracy:.2f}%  ({correct}/{len(test_ds)})\n")

        # =========================
        # Visualization
        # =========================
        rows = math.ceil(len(samples) / 5)
        plt.figure(figsize=(25, 7 * rows))

        for i, (img, p, g) in enumerate(samples):
            img_np = img.permute(1, 2, 0).numpy()
            img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
            img_np = img_np.clip(0, 1)
            color = 'blue' if p == g else 'red'
            plt.subplot(rows, 5, i + 1)
            plt.imshow(img_np)
            plt.axis('off')
            plt.title(
                f"[Pred]:\n{p}\n\n[GT]:\n{g}",
                fontsize=8, loc='left', color=color, pad=8
            )

        plt.suptitle(f"Epoch {epoch+1} | Accuracy: {accuracy:.2f}%", fontsize=14)
        plt.tight_layout()
        plt.show()

    print("\n✅ Training Complete!")


if __name__ == "__main__":
    main()
