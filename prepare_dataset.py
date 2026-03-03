import os
import cv2
import shutil
import numpy as np
from pathlib import Path
from tqdm import tqdm

def save_pre_sampled_dataset(src_root, dest_root, split_type="train", samples_per_class=4, start_idx=0, num_frames=32):
    src_path = Path(src_root)
    base_dest_path = Path(dest_root) / split_type

    class_folders = sorted([d for d in src_path.iterdir() if d.is_dir() and d.name not in ['train', 'test']])

    print(f"--- [{split_type}] Frame extraction started ({samples_per_class} videos per class) ---")
    for folder in class_folders:
        label_name = folder.name
        video_files = sorted(list(folder.glob("*.mp4")))[start_idx : start_idx + samples_per_class]

        for v_file in tqdm(video_files, desc=f"{split_type}/{label_name}"):
            # Create a folder per video (e.g. dataset/train/artifact/video_01/)
            video_sample_dir = base_dest_path / label_name / v_file.stem
            video_sample_dir.mkdir(parents=True, exist_ok=True)

            # Load video and uniformly sample frames
            cap = cv2.VideoCapture(str(v_file))
            raw_frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                raw_frames.append(frame)  # Keep BGR format for cv2.imwrite
            cap.release()

            # Extract num_frames frames via uniform sampling
            total = len(raw_frames)
            indices = np.linspace(0, total - 1, num_frames, dtype=int) if total >= num_frames else range(total)

            for i, idx in enumerate(indices):
                frame_path = video_sample_dir / f"frame_{i:02d}.jpg"
                cv2.imwrite(str(frame_path), raw_frames[idx])

            # Pad with last frame if total frames < num_frames
            for i in range(len(indices), num_frames):
                last_frame_path = video_sample_dir / f"frame_{len(indices)-1:02d}.jpg"
                shutil.copy2(last_frame_path, video_sample_dir / f"frame_{i:02d}.jpg")

            # Save caption
            with open(video_sample_dir.with_suffix(".txt"), "w") as f:
                f.write(f"What is it? It is a {label_name}.")

    print(f"[*] {split_type} preprocessing complete!\n")


# Run
# Clone dataset first:
#   git clone https://huggingface.co/datasets/KyuHyunChoi/ikkaron-jeonju-1shot
#   cd ikkaron-jeonju-1shot
#   python prepare_dataset.py
ORIGIN_PATH  = "."                     # mp4 videos are in the cloned repo root
DATASET_PATH = "video_dataset_frames"  # Output path for extracted frames

save_pre_sampled_dataset(ORIGIN_PATH, DATASET_PATH, "train", 1, 0)
save_pre_sampled_dataset(ORIGIN_PATH, DATASET_PATH, "test",  5, 1)
