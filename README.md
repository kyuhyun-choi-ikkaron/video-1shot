# Video 1-Shot Grounding

340M params. 1-Shot. 100% accuracy. 3 minutes. No LLM.

## Requirements
pip install torch transformers opencv-python tqdm matplotlib

## Dataset
git clone https://huggingface.co/datasets/KyuHyunChoi/ikkaron-jeonju-1shot

## Run
'''
git clone https://github.com/kyuhyun-choi-ikkaron/video-1shot
cd video-1shot
python prepare_dataset.py
python run.py
'''

## Citation
DINOv3: https://arxiv.org/abs/2508.10104
