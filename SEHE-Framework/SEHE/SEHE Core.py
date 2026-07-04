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
