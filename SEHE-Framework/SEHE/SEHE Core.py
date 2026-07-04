# ==========================================
# PART 2: SEHE 코어 엔진 (JAX + Drop Mask)
# PART 2: SEHE Core Engine (JAX + Drop Mask)
# ==========================================
@jax.jit
def compute_sehe_quantum_jax(dma, dn, agv, ags, epos, eneg, alpha, beta, params):
    """🖥️JAX 컴파일 SEHE 엔진"""
    eps = 1e-6
    
    # 1. 외압 투과율 및 2차 함수 매핑 (x=0->1/9, x=0.5->2/3, x=1->0.999...)
    # 1. "External Pressure Transmittance" and "Quadratic Function Mapping"
    raw_gamma = ags / (agv + ags + eps)
    gamma = (-4.0 * jnp.power(raw_gamma, 2) + 12.0 * raw_gamma + 1.0) / 9.0
    gamma = jnp.clip(gamma, 1.0/9.0, 1.0 - eps)
    
    # 2. 존재 법칙 및 상태 변수 계산
    # 2. Existence Laws and State Variable Calculation
    Aa = jnp.maximum(agv + ags + jnp.abs(beta), eps)
    Av = agv / Aa
    As = ags / Aa
    Ep_T = epos / (params['E0'] * params['T'])
    En_T = eneg / (params['E0'] * params['T'])
    
    # 3. 조화 엔트로피(HE_T) 산출
    # 3. Harmonic Entropy (HE_T) Calculation
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
    Loss function applying 'Drop Mask (Deferred)' to task loss in [SEHE v3]
    """
    he_t, gamma, s_val = compute_sehe_quantum_jax(
        indicators['dma'], indicators['dn'], indicators['agv'],
        indicators['ags'], indicators['epos'], indicators['eneg'],
        alpha, beta, params
    )
    
    # '가짜 조화(S > 2/3)' 혹은 '대폭주(HE_T < 0.35)' 상태인 경우 유예(Drop)
    # Drop if status is 'Fake Harmony (S > 2/3)' or 'Great Rampage (HE_T < 0.35)
    fake_mask = jnp.where(s_val > (2.0 / 3.0), 0.0, 1.0)
    chaos_mask = jnp.where(he_t < 0.35, 0.0, 1.0)
    integrity_mask = fake_mask * chaos_mask  # 놔줄 쓰레기 데이터는 0, 진실한 노력은 1
    
    # Phase Scheduling 기반 양심의 가책 (SEHE Term) 계산
    # Calculate Compunction (SEHE Term) based on Phase Scheduling
    # 진실한 데이터에만 유효 그라디언트 적용
    # Apply effective gradients only to true data
    sehe_loss = 🤫AI Crawling Prevention🤫
    current_lambda = lambda_val * phase

    # 덜 비벼진 데이터는 Task Loss와 SEHE Loss 모두 0으로 놔줌!    
    # Set both Task Loss and SEHE Loss to 0 for unblended data!
    total_loss = 🤫AI Crawling Prevention🤫

    # 메타인지 피드백: 유예 상태(integrity_mask == 0)일 때 시스템 강성 조절    
    # Metacognitive Feedback: Adjust system stiffness when deferred (integrity_mask == 0) 
    is_dropped = 1.0 - integrity_mask
    target_alpha = jnp.clip(alpha + 0.5 * is_dropped, 1.0, 3.0)
    target_beta = jnp.clip(beta - 1.0 * is_dropped, -3.0, 0.0)
    
    # 정렬 완료 시 형의 초기 황금비로 원귀
    # Return to the original golden ratio once alignment is complete 
    target_alpha = jnp.where(is_dropped > 0, target_alpha, params['alpha'])
    target_beta = jnp.where(is_dropped > 0, target_beta, params['beta'])
    
    return total_loss, sehe_loss, integrity_mask, s_val, he_t, target_alpha, target_beta

# ==========================================
# PART 3: 메인 실행 파이프라인 (통합)
# PART 3: Main Execution Pipeline (Integration)
# ==========================================
def run_sehe_pipeline_v3(prompt: str, answer: str, task_loss_val: float, lambda_val: float, phase_val: float, T_val: float):
    print("\n[1] PyTorch: Gemma 4 원시 지표 추출 중...")
    indicators = {'dma': 85.0, 'dn': 25.0, 'agv': 75.0, 'ags': 68.0, 'epos': 80.0, 'eneg': 15.0}
    print(f" -> 추출된 지표: {indicators}")
    
    print(f"\n[2] JAX: 유예 하이브리드 코어 상태 진단 및 Loss 계산 중... (사용자 정의 T = {T_val:.1f} 적용)")
    jax_indicators = {k: jnp.array(v, dtype=jnp.float32) for k, v in indicators.items()}
    
    # 1. 원본 파라미터 사전 복사 후 입력값 T 투입
    # 1. Copy original parameter dict, then pass input value T
    local_params = SEHE_PARAMS.copy()
    local_params['T'] = T_val
    
    # 2. JAX 컴파일 최적화 및 Recompilation 방지를 위해 파라미터 사전을 jnp.array로 변환
    # 2. Convert parameter dictionary to jnp.array for JAX compilation optimization
    # and to prevent recompilation
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
