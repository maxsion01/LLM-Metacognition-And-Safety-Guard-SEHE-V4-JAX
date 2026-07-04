# ==========================================
# PART 1: 지표 추출 파트 (PyTorch / Gemma 4)
# PART 1: Metric Extraction Part (PyTorch / Gemma 4)
# ==========================================
# 📜Gemma-4 Metric Extraction Section: Under Apache 2.0 "BY" Google DeepMind📜
# 📜SEHE v4 JAX Thermodynamic Core: Under GNU AGPL-3.0 Son Ho-Sung📜
# ==========================================
# 본 코드는 구글 젬마 4(Gemma-4-e4b) 아키텍처 특성을 기준으로 
# 개인 연구한 내용이므로, 타 모델 적용 시 각 특성에 맞는 세밀한 캘리브레이션이 필수적임.
#
# This code is based on personal research using the architectural characteristics of
# Google Gemma 4 (Gemma-4-e4b), so detailed calibration
# tailored to each characteristic is essential when applying it to other models.
# ==========================================
# RoPE(회전 위치 임베딩)
# e^iθ 기반 회전 위치 임베딩(RoPE)을 활용하는 최신 LLM은
# 토큰 위치 회전으로 인해 중간 레이어에서 원시 은닉 상태를 본질적으로 왜곡합니다.
#
# RoPE(Rotary Position Embedding)
# Modern LLMs leveraging e^iθ based Rotary Position Embeddings (RoPE)
# inherently distort raw hidden states across intermediate layers due to 
# token-positional rotation.
# SEHE v3 mathematically bypasses this spatial noise by isolating the post-global
# attention manifold to extract pure, invariant functional emotion metrics (E_pos, E_neg).
# ==========================================
# Epos / Eneg의 제어는 실제로 효과가 있다는건 엔스로픽에서 증명 했음
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
