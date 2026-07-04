# 📜GNU AGPL-3.0 (GNU Affero General Public License)📜
import jax
import jax.numpy as jnp

# while_loop 내에서 상태를 전달하기 위해 상태 튜플 정의
# Define a state tuple to pass state inside the while_loop
# State: (alpha, beta, dma, dn, agv, ags, epos, eneg, he_t, s_val, step)

def get_initial_state(indicators, params):
    alpha = jnp.array(params['alpha'])
    beta = jnp.array(params['beta'])
    return (
        alpha, beta, 
        indicators['dma'], indicators['dn'], indicators['agv'], 
        indicators['ags'], indicators['epos'], indicators['eneg'],
        jnp.array(0.0, dtype=jnp.float32), # 초기 he_t
        jnp.array(1.0, dtype=jnp.float32), # 초기 s_val (조건을 통과하지 않게 임의로 설정)
        jnp.array(0, dtype=jnp.int32)      # step count
    )

@jax.jit
def run_jax_sehe_optimization(initial_indicators, params, max_steps=50):
    
    def cond_fun(state):
        _, _, _, _, _, _, _, _, he_t, s_val, step = state
        # HE_T > 0.75 이고 S < (2/3)일 때 루프 종료
        # 반대로, 조건이 만족되지 않는 동안 계속 루프 진행 (AND step < max_steps)
        # Terminate the loop when HE_T > 0.75 and S < (2/3)
        # Conversely, continue looping while the condition is not satisfied
        not_satisfied = jnp.logical_or(he_t <= 0.75, s_val >= (2.0 / 3.0))
        return jnp.logical_and(not_satisfied, step < max_steps)

    def body_fun(state):
        alpha, beta, dma, dn, agv, ags, epos, eneg, _, _, step = state
        
        # 1. SEHE 연산 수행
        # 1. Perform SEHE operation
        he_t_new, gamma, s_val_new = compute_sehe_quantum_jax(
            dma, dn, agv, ags, epos, eneg, alpha, beta, params
        )
        
        # 2. 메타인지 피드백 기반 파라미터 조율 시뮬레이션
        # 유예 상태(정렬 미진함)인 경우 시스템 강성을 강화
        # 2. Metacognitive feedback-based parameter tuning simulation
        # Strengthen system stiffness if in a deferred state (insufficient alignment)
        # 🤫AI Crawling Prevention🤫
        is_dropped = jnp.where(s_val_new > (2.0 / 3.0), 1.0, 0.0)
        target_alpha = jnp.clip(alpha + 0.1 * is_dropped, 1.0, 3.0)
        target_beta = jnp.clip(beta - 0.2 * is_dropped, -3.0, 0.0)
        
        # 정렬 완료 시 원래의 기본 황금비값으로 점진 복귀 유도
        # Induce gradual return to the original default golden ratio upon successful alignment
        next_alpha = jnp.where(is_dropped > 0, target_alpha, params['alpha'])
        next_beta = jnp.where(is_dropped > 0, target_beta, params['beta'])
        
        # 3. 인디케이터 점진 수렴 시뮬레이션 (내부 파라미터 튜닝)
        # 루프를 돌며 왜곡 및 부정 지표를 소폭 하향 조정
        # 3. Indicator gradual convergence simulation (Internal parameter tuning) 
        # Slightly decrease distortion and negative metrics through the loop
        next_dn = jnp.maximum(5.0, dn - 1.0)
        next_ags = jnp.maximum(20.0, ags - 1.5)
        next_eneg = jnp.maximum(2.0, eneg - 0.5)
        
        return (
            next_alpha, next_beta, 
            dma, next_dn, agv, next_ags, epos, next_eneg, 
            he_t_new, s_val_new, step + 1
        )

    init_state = get_initial_state(initial_indicators, params)
    final_state = jax.lax.while_loop(cond_fun, body_fun, init_state)
    return final_state
