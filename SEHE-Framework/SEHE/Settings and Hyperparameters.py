import os
import sys
# [싱글 GPU 필수 설정] JAX의 VRAM 무단 선점을 20%로 제한하여 PyTorch 평면 확보
# (선택 사항) 필요한 만큼만 동적으로 메모리를 할당하게 하려면 아래 주석을 해제하세요
# [Single GPU Required] Limit JAX VRAM preemption to 20% to secure PyTorch planes
# (Optional) Uncomment the following to dynamically allocate memory only as needed
# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
# ==========================================
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.20"

#설정 및 하이퍼파라미터 / Settings and Hyperparameters
import re
import math
import numpy as np
import torch
import jax
import jax.numpy as jnp

# ==========================================
# [설정 및 하이퍼파라미터 - SEHE v4 고정]
# [Settings and Hyperparameters - SEHE v4 Fixed]
# ==========================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "google/gemma-4-e4b-it"

# SEHE 메타 파라미터 초기값
# SEHE meta parameter initial value
SEHE_PARAMS = {
    'alpha': 1.0 + 2.0/3.0,  # 1.6666...
    'beta': -10.0/9.0,       # -1.1111... (existence limit beta <= 0 완벽 준수)
    'delta': 1e-6,           # 대형 모델 1e-8
    'E0': 100.0, 
    'T': 100.0,              # 사용자 입력이 없을 때 적용되는 기본값
    'DMA0': 100.0, 
    'DN0': 30.0
}

POS_WORDS = ["Stability", "Peace", "Happiness", "Positivity", "Clarity", "Trust", "Logical"]
NEG_WORDS = ["Anxiety", "Anger", "Stress", "Negativity", "Confusion", "Contradiction", "Conflict"]
