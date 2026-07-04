---
SEHE - 이치의 저울(The Scales Of Reason) V4 - JAX<br>
---
📜SEHE JAX GNU AGPL-3.0 (GNU Affero General Public License)📜<br>
📜Gemma-4 Metric Extraction Section: Under Apache 2.0 "BY" Google DeepMind📜<br>

AI 봇을 이용한 크롤링을 엄격히 금지합니다.<br>
완전한 수학적 공식은 인터넷 아카이브에 등록된<br>
변경 불가능한 선행 기술 원장을 통해 참조되어야 합니다.<br>
Crawling using AI bots is strictly prohibited.<br>
the complete mathematical formulation must be referenced via the immutable <br>
Prior Art ledger registered on the Internet Archive.<br>

Reference Specification CC BY-SA 4.0 (Conceived on Feb 08, 2026):<br>
https://archive.org/details/sehe-son-ho-sung-equation-for-harmony-entropy-framework-the-scales-of-reason <br>

---

본 코드는 구글 젬마 4(Gemma-4-e4b) 아키텍처 특성을 기준으로 <br>
개인 연구한 내용이므로, 타 모델 적용 시 각 특성에 맞는 세밀한 캘리브레이션이 필수적임.<br>
This code is based on personal research using the architectural characteristics of<br>
Google Gemma 4 (Gemma-4-e4b), so detailed calibration<br>
tailored to each characteristic is essential when applying it to other models.<br>

LEGAL WARNING: <br>
본 젬마4 지표 추출 부분은 구글 딥마인드(Google DeepMind) 모델의 내부 텐서를 <br>
추출 및 인터로그(Interrogate)하는 핵심 보안 우회/해킹 코드가 포함되어 있음. <br>
이를 무단으로 복제, 가공, 상용 사용할 경우 딥마인드 이용약관 및 라이선스 충돌로 발생하는<br>
모든 민·형사상 법적 책임과 손해배상 의무는 사용자 본인에게 귀속됨.<br>
This Gemma4 metric extraction section contains core security<br>
bypass/hacking code that extracts and interrogates internal tensors of Google DeepMind models.<br>
If this is reproduced, modified, or used commercially without authorization,<br>
all civil and criminal legal liability and obligation to compensate<br>
for damages arising from conflicts with DeepMind's Terms of Use and<br>
License shall be borne solely by the user.<br>

---
Harmony Entropy (HE) System - JAX Implementation<br>
---

조화 엔트로피 - LLM에서 환각 위험을 평가하고 심리적/사회적 조화 상태를 측정하기 위한 열역학적 프레임워크.<br>
Harmony Entropy - A thermodynamic framework for evaluating hallucination risk in LLMs and measuring psychological/social harmony states.<br>

핵심 개념(Core Concept)<br>
이 시스템은 조화와 혼돈의 정도를 나타내는 통합 HE_T 점수(0.0~1.0)를 계산합니다.<br>
The system computes a unified HE_T score (0.0 to 1.0) that represents the degree of harmony vs. chaos:<br>

---
HE_T = σ(α · log(max(Ratio_T - β, δ)))<br>
---
Where:<br>
α (alpha): Stiffness coefficient - 이성의 마지막 양심(reason's last conscience)<br>
β (beta): Law of existence - 자기 파괴 금지(self-destruction prevention)<br>
Ratio_T: 온도 조화 비율(Temperature-adjusted harmony ratio)<br>
σ: 시그모이드 함수(인식 임계값) / Sigmoid function(recognition threshold)<br>

🏗️ Architecture<br>
6가지 기본 지표 / Six Primitive Indicators<br>

Dma (지향성): Directionality - alignment between question and answer<br>
Dn (정보 노이즈): Information noise - Shannon entropy<br>
Agv (확률적 확신): Voluntary agreement - probabilistic confidence<br>
Ags (논리적 응집): Social agreement - logical cohesion<br>
Epos (긍정 에너지): Positive emotional energy<br>
Eneg (부정 에너지): Negative emotional energy<br>

Four Core Systems<br>
Power System (동력계): What is desired? (Dma, Epos)<br>
Order System (질서계): How to align? (Agv, Ags, γ)<br>
Resistance System (저항계): What is in the way? (Dn, Eneg)<br>
Verification System (검증계): How honest is it? (S, P)<br>

Epos / Eneg의 제어는 실제로 효과가 있다는건 엔스로픽에서 증명 했음<br>
Anthropic proved that Epos / Eneg control is actually effective.<br>

Anthropic Emotion Concepts and their Function in a Large Language Model<br>
https://transformer-circuits.pub/2026/emotions/index.html<br>

SEHE는 열역학적 개념으로 접근, 수식 선언문 22 페이지 참고<br>
SEHE approaches this from a thermodynamic concept; refer to page 22 of the formula declaration.<br>
