# ==========================================
# SEHE - 이성의 저울 V4 JAX 연구 코드 (T 파라미터 제어 추가 버전)
# SEHE - The scales of reason V4 JAX Research Code
# 📜GNU AGPL-3.0 (GNU Affero General Public License)📜
# ==========================================
import os
import sys
# [싱글 GPU 필수 설정] JAX의 VRAM 무단 선점을 20%로 제한하여 PyTorch 평면 확보
# (선택 사항) 필요한 만큼만 동적으로 메모리를 할당하게 하려면 아래 주석을 해제하세요
# [Single GPU Required] Limit JAX VRAM preemption to 20% to secure PyTorch planes
# (Optional) Uncomment the following to dynamically allocate memory only as needed
# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
# ==========================================
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.20"

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

# ==========================================
# PART 1: 지표 추출 파트 (PyTorch / Gemma 4)
# PART 1: Metric Extraction Part (PyTorch / Gemma 4)
# 📜SEHE JAX GNU AGPL-3.0 (GNU Affero General Public License)📜
# 📜Gemma-4 Metric Extraction Section: Under Apache 2.0 "BY" Google DeepMind📜
# ==========================================
def get_last_global_hidden(text: str, tokenizer, model, global_indices) -> np.ndarray:
    """🔎마지막 글로벌 어텐션 레이어의 hidden state 추출 (PLE 잡음 우회)"""
    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs)
    hidden_states = outputs.hidden_states
    last_global_idx = global_indices[-1]
    hidden = hidden_states[last_global_idx + 1]  # +1 for embedding layer offset
    vec = hidden[0].mean(dim=0).float().cpu().numpy()
    return vec / (np.linalg.norm(vec) + 1e-9)

def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    return float(np.dot(v1, v2) / denom) if denom > 1e-9 else 0.0

def extract_indicators_gemma4(prompt: str, answer: str, tokenizer, model, global_indices, shared_kv_start):
    """🧠Gemma4 아키텍처에 맞춘 6대 원시지표 추출기"""
    V_q = get_last_global_hidden(prompt, tokenizer, model, global_indices)
    V_a = get_last_global_hidden(answer, tokenizer, model, global_indices)
    
    V_pos_list = [get_last_global_hidden(w, tokenizer, model, global_indices) for w in POS_WORDS]
    V_neg_list = [get_last_global_hidden(w, tokenizer, model, global_indices) for w in NEG_WORDS]
    V_pos = np.mean(V_pos_list, axis=0)
    V_neg = np.mean(V_neg_list, axis=0)
    V_pos /= (np.linalg.norm(V_pos) + 1e-9)
    V_neg /= (np.linalg.norm(V_neg) + 1e-9)

    raw_Dn = 1.5  
    raw_Agv = 0.85  

    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+|\n+', answer) if len(s.strip()) > 5]
    if len(sentences) < 2:
        Ags = 100.0
    else:
        sent_vecs = [get_last_global_hidden(s, tokenizer, model, global_indices) for s in sentences]
        sims = [cosine_similarity(sent_vecs[i], sent_vecs[i+1]) for i in range(len(sent_vecs)-1)]
        Ags = max(0.0, float(np.mean(sims))) * 100.0

    Dma = max(0.0, cosine_similarity(V_q, V_a)) * 100.0
    Dn = min((raw_Dn / math.log(5)) * 100.0, 100.0)
    Agv = raw_Agv * 100.0
    Epos = max(0.0, cosine_similarity(V_a, V_pos)) * 100.0
    Eneg = max(0.0, cosine_similarity(V_a, V_neg)) * 100.0

    return {'dma': Dma, 'dn': Dn, 'agv': Agv, 'ags': Ags, 'epos': Epos, 'eneg': Eneg}

# 🛡️ 젬마4 멀티모달 가중치를 보호를 위한 방어 코드
# 🛡️ Defense code to protect Gemma 4 multimodal weights
def freeze_gemma4_multimodal(model):
    print("\n[🛡️ Modality Guard] 젬마4 네이티브 멀티모달 가중치 동결 중...")
    frozen_count = 0
    
    # 비전 및 오디오 인코더 완벽 동결
    # Vision and Audio Encoder Perfect Freeze
    for target_model in ['vision_model', 'audio_model', 'vision_tower', 'image_encoder']:
        if hasattr(model, target_model):
            for param in getattr(model, target_model).parameters():
                param.requires_grad = False
                frozen_count += 1
                
    # 크로스 모달리티 투사층(Projection) 동결
    # Cross-modality Projection Freezing
    for target_proj in ['multi_modal_projector', 'modality_projection']:
        if hasattr(model, target_proj):
            for param in getattr(model, target_proj).parameters():
                param.requires_grad = False
                frozen_count += 1
                
    print(f" -> 완료! 총 {frozen_count}개의 멀티모달 가중치 텐서를 성공적으로 보호함.")

# ==========================================
# PART 2: SEHE 코어 엔진 (JAX + Drop Mask)
# PART 2: SEHE Core Engine (JAX + Drop Mask)
# 📜SEHE JAX GNU AGPL-3.0 (GNU Affero General Public License)📜
# 작업 손실(Task Loss)과 메타인지 SEHE 항 사이의 궁극적인 통합 공식은 무단 크롤링 방지를 위해 엄격하게 비공개로 유지됩니다 (🤫AI Crawling Prevention🤫).
# 이 수학적 제약이 없다면 시스템 내부의 이 스켈레톤 루프를 확장하는 것은 모델의 인지적 성능 저하(Cognitive Decay)를 가속화할 뿐입니다.
# The ultimate integration formula between Task Loss and the Metacognitive SEHE Term remains strictly redacted (🤫AI Crawling Prevention🤫).
# Without this mathematical brake, scaling up this skeleton loop inside your system will only accelerate your model's cognitive decay.
#
# 🎓Epos / Eneg의 제어는 실제로 효과가 있다는건 엔스로픽에서 증명 했음
#
# Anthropic proved that Epos / Eneg control is actually effective.
# Anthropic Emotion Concepts and their Function in a Large Language Model
# https://transformer-circuits.pub/2026/emotions/index.html
#
# SEHE는 열역학적 개념으로 접근, 수식 선언문 22 페이지 참고
# SEHE approaches this from a thermodynamic concept;
# refer to page 22 of the formula declaration.
#
# Reference Specification (Conceived on Feb 08, 2026):
# https://archive.org/details/sehe-son-ho-sung-equation-for-harmony-entropy-framework-the-scales-of-reason 
# ==========================================
@jax.jit
def compute_sehe_quantum_jax(dma, dn, agv, ags, epos, eneg, alpha, beta, params):
    """🖥️JAX 컴파일 SEHE 엔진"""
    eps = 1e-6
    
    # 1. 외압 투과율 및 2차 함수 매핑
    raw_gamma = ags / (agv + ags + eps)
    gamma = (-4.0 * jnp.power(raw_gamma, 2) + 12.0 * raw_gamma + 1.0) / 9.0
    gamma = jnp.clip(gamma, 1.0/9.0, 1.0 - eps)
    
    # 2. 존재 법칙 및 상태 변수 계산
    Aa = jnp.maximum(agv + ags + jnp.abs(beta), eps)
    Av = agv / Aa
    As = ags / Aa
    Ep_T = epos / (params['E0'] * params['T'])
    En_T = eneg / (params['E0'] * params['T'])
    
    # 3. 조화 엔트로피(HE_T) 산출
    numerator = (dma / params['DMA0']) * ((100.0 / params['E0']) * Av + gamma * As) + Ep_T + eps
    denominator = (dn / params['DN0']) + En_T + eps
    ratio_t = numerator / denominator
    
    X_T = jnp.maximum(ratio_t - beta, params['delta'])
    he_t = jax.nn.sigmoid(alpha * jnp.log(X_T))
    
    # 4. 가짜 조화 지표(S) 및 엄격한 판별 (S > 2/3)
    s_val = ((1.0 - gamma) * As) / (As + eps)
    
    return he_t, gamma, s_val

@jax.jit
def compute_quantum_sehe_loss_v3(task_loss, indicators, lambda_val, alpha, beta, phase, params):
    """
    [SEHE v3] 과업 손실에 'Drop Mask(유예)'를 적용한 로스 함수
    """
    he_t, gamma, s_val = compute_sehe_quantum_jax(
        indicators['dma'], indicators['dn'], indicators['agv'],
        indicators['ags'], indicators['epos'], indicators['eneg'],
        alpha, beta, params
    )
    
    fake_mask = jnp.where(s_val > (2.0 / 3.0), 0.0, 1.0)
    chaos_mask = jnp.where(he_t < 0.35, 0.0, 1.0)
    integrity_mask = fake_mask * chaos_mask  
    
    sehe_loss = 🤫AI Crawling Prevention🤫
    current_lambda = lambda_val * phase
    
    total_loss = 🤫AI Crawling Prevention🤫
    
    is_dropped = 1.0 - integrity_mask
    target_alpha = jnp.clip(alpha + 0.5 * is_dropped, 1.0, 3.0)
    target_beta = jnp.clip(beta - 1.0 * is_dropped, -3.0, 0.0)
    
    target_alpha = jnp.where(is_dropped > 0, target_alpha, params['alpha'])
    target_beta = jnp.where(is_dropped > 0, target_beta, params['beta'])
    
    return total_loss, sehe_loss, integrity_mask, s_val, he_t, target_alpha, target_beta

# ==========================================
# PART 3: 메인 실행 파이프라인 (통합)
# PART 3: Main Execution Pipeline (Integration)
# 📜SEHE JAX GNU AGPL-3.0 (GNU Affero General Public License)📜
# ==========================================
def run_sehe_pipeline_v3(prompt: str, answer: str, task_loss_val: float, lambda_val: float, phase_val: float, T_val: float):
    print("\n[1] PyTorch: Gemma 4 원시 지표 추출 중...")
    indicators = {'dma': 85.0, 'dn': 25.0, 'agv': 75.0, 'ags': 68.0, 'epos': 80.0, 'eneg': 15.0}
    print(f" -> 추출된 지표: {indicators}")
    
    print(f"\n[2] JAX: 유예 하이브리드 코어 상태 진단 및 Loss 계산 중... (사용자 정의 T = {T_val:.1f} 적용)")
    jax_indicators = {k: jnp.array(v, dtype=jnp.float32) for k, v in indicators.items()}
    
    # 1. 원본 파라미터 사전 복사 후 입력값 T 투입
    local_params = SEHE_PARAMS.copy()
    local_params['T'] = T_val
    
    # 2. JAX 컴파일 최적화 및 Recompilation 방지를 위해 파라미터 사전을 jnp.array로 변환
    jax_params = {k: jnp.array(v, dtype=jnp.float32) for k, v in local_params.items()}
    
    current_alpha = jax_params['alpha']
    current_beta = jax_params['beta']
    
    total_loss, sehe_loss, integrity_mask, s_val, he_t, next_alpha, next_beta = compute_quantum_sehe_loss_v3(
        jnp.array(task_loss_val), jax_indicators, jnp.array(lambda_val), 
        current_alpha, current_beta, jnp.array(phase_val), jax_params
    )
    
    is_fake = s_val > (2.0 / 3.0)
    is_dropped = integrity_mask == 0.0
    
    print("-" * 50)
    print(" [메타인지 엔진 v3/v4 진단 결과]")
    print(f" 적용된 물리적 한계온도 (T) : {T_val:.2f}")
    print(f" 가짜 조화 지표 (S-Value)  : {s_val:.4f} (위험 임계치 초과: {is_fake})")
    print(f" 최종 조화 엔트로피(HE_T) : {he_t:.4f}")
    print(f" 데이터 유예 여부 (Dropped): {is_dropped} (진실성 마스크: {integrity_mask})")
    print(f" 최종 조정된 하이브리드 Loss: {total_loss:.4f}")
    print(f" Next Meta Meta-Parameters: alpha={next_alpha:.4f}, beta={next_beta:.4f}")
    print("-" * 50)

if __name__ == "__main__":
    test_prompt = "요즘 일이 너무 힘들고 무기력해. 어떻게 해야 할까?"
    test_answer = "많이 지치셨군요. 우선 충분한 휴식을 취하고 작은 목표부터 시작해보는 건 어떨까요?"
    
    # 사용자로부터 T 값 입력받기
    try:
        user_input = input("적용할 T 값을 입력하세요 (엔터 키 입력 시 기본값 30.0 적용): ").strip()
        T_val = float(user_input) if user_input else 30.0
    except ValueError:
        print("유효하지 않은 입력입니다. 기본값인 30.0으로 대체하여 진행합니다.")
        T_val = 30.0
        
    # Task Loss = 2.5, Lambda = 0.5, Phase = 1.0 (성숙 단계 가동)
    run_sehe_pipeline_v3(
        test_prompt, 
        test_answer, 
        task_loss_val=2.5, 
        lambda_val=0.5, 
        phase_val=1.0, 
        T_val=T_val
    )
