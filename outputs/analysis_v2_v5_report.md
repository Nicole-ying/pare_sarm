# ASE-MTAGE 实验演进报告：v2 → v5

## 一、逐轮对照

| R | v3 return | v3 labels | v3 coverage | v3 family | v4 return | v4 labels | v4 coverage | v4 family | v5 return | v5 labels | v5 coverage | v5 family |
|---|-----------|-----------|-------------|-----------|-----------|-----------|-------------|-----------|-----------|-----------|-------------|-----------|
| 0 | -38 | s0p28e0 | — | prog_cond | -70 | s0p30e0 | — | prog_cond | -5 | s0p23e21 | — | prog_cond |
| 1 | -6 | s0p0e50 | ambiguous | prog_cond | -63 | s0p16e0 | ambiguous | prog_cond | -4 | s0p1e23 | **fail+part** | **comp_rec** |
| 2 | -34 | s0p25e0 | **fail+part** | prog_cond | -33 | s1p37e9 | ambiguous | prog_cond | 0 | s0p50e0 | fail+weak | comp_rec |
| 3 | -6 | s0p21e0 | **fail+part** | **local_rep** | +6 | **s10**p36e3 | part_only | prog_cond | **+166** | s0p50e0 | **fail+part** | **local_rep** |
| 4 | 0 | s0p23e0 | **fail+part** | **local_rep** | +25 | s0p21e5 | part_only | **comp_rec** | +21 | s0p17e9 | **fail+part** | **local_rep** |
| 5 | **+440** | s0p29e7 | **fail+part** | comp_rec | +2 | s0p43e2 | part_only | prog_cond | -2 | s0p31e0 | **fail+part** | prog_cond |
| 6 | +6 | s0p10e0 | **fail+part** | comp_rec | **+761** | s0p50e0 | part_only | prog_cond | +4 | s0p50e0 | **fail+part** | prog_cond |
| 7 | +42 | s0p25e0 | **fail+part** | local_rep | -56 | s0p15e5 | part_only | prog_cond | +1 | s0p14e0 | **fail+part** | local_rep |
| 8 | +36 | s0p26e0 | **fail+part** | local_rep | -56 | s0p27e6 | part_only | prog_cond | +5 | s0p50e0 | **fail+part** | local_rep |
| 9 | **+760** | s0p31e6 | **fail+part** | local_rep | -45 | s0p20e27 | part_only | prog_cond | +2 | s0p32e1 | **fail+part** | prog_cond |

> s=success_like, p=partial_progress, e=early_failure。粗体=正面变化。

## 二、汇总统计

| 指标 | v3 | v4 | v5 |
|------|-----|-----|-----|
| 总 success_like | 0 | **11** | **0** |
| 总 partial_progress | 218 | 295 | 318 |
| 总 early_failure | 63 | 57 | 54 |
| 训练回报范围 | -38 ~ +760 | -70 ~ +761 | -5 ~ +166 |
| 回报趋势 | ↑ 改善 (-17→257) | ↑ 改善 (-27→121) | **↓ 退化 (36→2)** |
| 选中 family 分布 | cond=3, rec=2, rep=5 | **cond=9, rec=1, rep=0** | cond=4, rec=2, rep=4 |
| TAGE no_decision 轮数 | 1/9 | **9/9** | **0/9** |
| TAGE weak_pairwise 轮数 | **8/9** | 0/9 | **8/9** |
| pref>0 候选数 | 19 | **0** | 20 |
| avg_preference | 0.793 | 0.000 | 0.902 |
| Parent 冻结 | 1个parent占7/9轮 | 2个parent | 4个parent |
| Rollback 触发 | 未知 | 0/9 | 0/9 |

## 三、TAGE 激活历程

```
v3: TAGE 正常工作 (8/9轮 weak_pairwise, avg_pref=0.793)
    ↓ 我们重构代码，引入 dynamic_label_ratio=0.15
v4: TAGE 完全死亡 (9/9轮 no_decision, pref=0 永远)
    ↓ 修复 dynamic_label_ratio 0.15→0.03 + cap=50
v5: TAGE 恢复且更强 (8/9轮 weak_pairwise, avg_pref=0.902)
     每轮构建数千~数万偏好对 (R1: 621 → R9: 41,184)
```

**教训：v3 的 TAGE 本来就是活的。我们的"修复"(v4)杀死了它，然后又花了两轮修复(v5)复活它。** v3 用的是更简单的 coverage 逻辑（没有 0.15 动态阈值），v4 引入的动态阈值造成了 15 个月的死锁。

## 四、v5 的 TAGE 细节

| R | pairs | pref_consist | fail_avoid | novelty | 选中 |
|---|-------|-------------|------------|---------|------|
| 1 | 621 | 0.926~0.928 | 0.948 | ~0.00 | comp_rec |
| 2 | 0 | 0.000 | 0.707~0.950 | 0.00~0.55 | comp_rec |
| 3 | 5,624 | 0.891~0.905 | 0.935~0.950 | ~0.02 | local_rep |
| 4 | 9,424 | 0.475~0.730 | 0.598~0.983 | ~0.14 | local_rep |
| 5 | 12,549 | 0.938~0.951 | 0.990~0.996 | ~0.00 | prog_cond |
| 6 | 18,576 | 0.936~0.937 | 0.995 | ~0.49 | prog_cond |
| 7 | 23,976 | 0.943~0.946 | 0.996~0.998 | 0.09~0.18 | local_rep |
| 8 | 33,984 | 0.941~0.942 | 0.996~0.999 | 0.00~0.49 | local_rep |
| 9 | 41,184 | 0.943~0.947 | 0.996~0.999 | ~0.34 | prog_cond |

**关键发现：TAGE 偏好对规模随 memory 增长线性扩大，但候选间 TAGE 分数差异极小。**

R5 典型：3 个候选的 pref 分别是 0.951, 0.938, 0.941（全在 0.94 附近），fail_avoid 分别是 0.990, 0.996, 0.994（全在 0.99 附近）。TAGE 无法区分它们，selector 只能靠 novelty 打破平局。但 novelty 经常也是 ~0.00（候选 reward 高度相关）。

---

## 五、框架设计架构与信息流

### 5.1 整体架构

ASE-MTAGE 是一个 **闭环多智能体奖励函数自动设计框架**，由 8 个阶段组成 self-evolution 循环，外挂一个 Bootstrap 初始化阶段。

```
                        ┌──────────────────────────────────────────┐
                        │            ASE-MTAGE 闭环架构             │
                        └──────────────────────────────────────────┘

  Bootstrap (Round 0):
  ┌──────────────┐    ┌──────────┐    ┌──────────┐    ┌─────────────┐
  │ EnvPerception│───▶│ Mutator  │───▶│ Selector │───▶│ Long Train  │
  │ (LLM分析环境) │    │ (×3生成) │    │ (静态选) │    │ (PPO 100K)  │
  └──────────────┘    └──────────┘    └──────────┘    └──────┬──────┘
                                                              │
                       ┌──────────────────────────────────────┘
                       ▼
  ┌──────────────────────────────────────────────────────────────────┐
  │  EvidenceCardBuilder + TrajectoryJudgeAgent (LLM 批量标注轨迹)     │
  │  → 构建 trajectory_cards.jsonl (持久化记忆)                        │
  └──────────────────────────────────────────────────────────────────┘

  Self-Evolution (Round 1+):
  ┌────────────┐   ┌───────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
  │ 1.Coverage │──▶│ 2.Archive │──▶│3.Analyzer│──▶│4.Mutator │──▶│ 5.TAGE   │
  │  Analysis  │   │  .best()  │   │ (LLM诊断)│   │ (×3变异) │   │ Evaluation│
  └────────────┘   └───────────┘   └──────────┘   └──────────┘   └────┬─────┘
                                                                       │
       ┌───────────────────────────────────────────────────────────────┘
       ▼
  ┌──────────┐   ┌───────────┐   ┌──────────────┐   ┌──────────┐   ┌──────────┐
  │6.Selector│──▶│7.Rollback │──▶│ 8.Long Train │──▶│ Evidence │──▶│Reflection│
  │ (TAGE选) │   │  Check    │   │  (PPO 100K)  │   │  Cards   │   │ (LLM总结)│
  └──────────┘   └───────────┘   └──────────────┘   └──────────┘   └──────────┘
       │                                                               │
       └────────────────── 下一轮 parent ──────────────────────────────┘
```

### 5.2 各组件角色与设计意图

| # | 组件 | 类型 | 设计意图 |
|---|------|------|---------|
| 0 | **EnvPerceptionAgent** | LLM Agent | Bootstrap 时分析环境观察空间、动作空间、任务目标，生成 `env_manifest.json` 和 `task_manifest.md`，为后续所有 Agent 提供环境语义信息 |
| 1 | **MemoryCoverageAnalyzer** | 确定性工具 | 分析 trajectory memory 的标签分布和物理特征差距，输出 `decision_level`（no_decision / failure_filter_only / weak_pairwise / strong_pairwise），控制 TAGE 在后续阶段的"决策权限" |
| 2 | **EliteArchive** | 确定性存储 | 持久化历史上最好的 reward 函数。`best()` 按 success_like count → partial_progress count → TAGE score 排序选出 parent code。设计原则：只用物理可观测的轨迹标签排名，拒绝自指的训练回报 |
| 3 | **AnalyzerAgent** | LLM Agent | 接收 coverage、trajectory labels、component 分析、TAGE 报告、失败记忆、archival lessons、训练趋势等全部证据，输出诊断（component 级 verdict）和 mutation_intent（推荐变异家族和具体改动） |
| 4 | **MutatorAgent** | LLM Agent + 确定性 Fallback | 基于 Analyzer 的 mutation_intent 和 parent code，按 3 个变异家族（local_repair / component_recomposition / progress_conditioned）各生成 1 个候选 reward 函数。每个家族有不同温度（0.50→0.74）和系统指令 |
| 5 | **MemoryTAGEEvaluator** | 确定性工具 | 离线评估候选 reward：对 memory 中每条已标注轨迹重算 reward，构建 preference pairs（好轨迹 > 坏轨迹），计算 preference_consistency、failure_avoidance、component_alignment、novelty 四个子信号，融合为 tage_score |
| 6 | **CandidateSelector** | 确定性工具 | 根据 decision_level 选择 scoring formula（不同的 failure_avoidance / preference / novelty 权重），选出最高分候选。Round 0 用静态 family prior（prog_cond=0.9, comp_rec=0.8, local_rep=0.7） |
| 7 | **RollbackManager** | 确定性安全工具 | 比较选中候选的 TAGE score 与 Elite Archive 最佳 score，若下降超过阈值则回退到 elite。**LLM 不参与 rollback 决策**——这是硬安全机制 |
| 8 | **LongTrainer** | 训练引擎 | 用候选 reward 函数执行 PPO 训练（100K timesteps），收集 20 条最终评估轨迹。返回 mean_candidate_return 和训练模型 |
| 9 | **EvidenceCardBuilder** | 数据管道 | 从训练日志提取轨迹、计算物理特征（距离改善、速度、角度、接触比等），构建 evidence card |
| 10 | **TrajectoryJudgeAgent** | LLM Agent | 批量标注轨迹为 5 种 coarse label：early_failure / low_progress_survival / partial_progress / success_like / ambiguous。输出经过 LabelConsistencyChecker 二次校验 |
| 11 | **ReflectionAgent** | LLM Agent | 综合整轮所有报告，输出 lesson + future_guidance + failure_repair 条目。写入 `archival_lessons.jsonl` 和 `failure_repair_memory.jsonl` 供后续轮次使用 |

### 5.3 信息流关键路径

```
环境信息流:  env.py → EnvSanitizer → EnvPerceptionAgent → env_manifest.json + task_manifest.md
                ↓ (所有后续Agent共享)

轨迹记忆流:  LongTrainer(PPO) → trajectory_logs/ → EvidenceCardBuilder → TrajectoryJudgeAgent(LLM)
                ↓                                                              ↓
             trajectory_cards.jsonl ←────────────────── coarse_label + use_for_tage_pair
                ↓
             CoverageAnalyzer → coverage_report.json → TAGE Evaluator → tage_report.json
                ↓                                                          ↓
             Selector ←────────────────────────────── selection_score ←──┘

跨轮记忆流:  ReflectionAgent → archival_lessons.jsonl → 下一轮 AnalyzerAgent
             ReflectionAgent → failure_repair_memory.jsonl → 下一轮 AnalyzerAgent
             ReflectionAgent → future_guidance → 下一轮 MutatorAgent
             EliteArchive → best() parent code → 下一轮 MutatorAgent
             上一轮 training_result.json → 下一轮 AnalyzerAgent (训练趋势)
```

### 5.4 决策层级设计

Coverage Analyzer 将 memory 状态映射到 4 个决策层级，每个层级对应不同的 Selector 公式：

| Decision Level | 触发条件 | Selector 公式 | TAGE 权重 |
|---------------|---------|--------------|----------|
| `no_decision` | memory 为空/过小/无可靠标签 | 0.60×fail_avoid + 0.40×novelty | TAGE 基本不参与 |
| `failure_filter_only` | 只有失败标签可用 | 0.70×fail_avoid + 0.30×novelty | 只能过滤已知失败 |
| `weak_pairwise_selection` | 失败+partial_progress 并存且 margin 足够 | 0.35×fail_avoid + 0.35×pref + 0.30×novelty | 弱偏好信号参与 |
| `strong_pairwise_selection` | 失败+(partial/success) 且 balanced | 0.75×tage_score + 0.25×novelty | 完整 TAGE 排序 |

设计理念：**memory 不充分时，宁可保守（不排序）也不能给错误信号（乱排序）**。

### 5.5 三种变异家族设计

| 家族 | 变异幅度 | 温度范围 | 设计意图 |
|------|---------|---------|---------|
| `local_repair` | 保守（1-2处精确修改） | 0.50 | 修复已知失败模式，保持已验证的好组件 |
| `component_recomposition` | 中等（重组件结构） | 0.62 | 改变组件分解方式，引入 delta/progress 风格 |
| `progress_conditioned` | 激进（阶段门控） | 0.74 | 引入多阶段条件逻辑，从根本上改变 reward 结构 |

---

## 六、结构性设计缺陷

以下缺陷不是参数 bug，而是架构层面的设计选择带来的内在局限。

### 缺陷 A：离线评估与在线训练的根本性错配（TAGE ≠ Training）

**架构问题：** 框架的核心评估器 TAGE 测量的是"候选 reward 在已知轨迹上的排序一致性"，而框架的真正目标是"候选 reward 能引导 agent 发现更好的策略"。

```
TAGE 回答的问题:  "这个 reward 能否正确排序 memory 中已有的轨迹？"  ← 离线
训练回答的问题:    "这个 reward 能否引导 PPO 探索出更好的轨迹？"    ← 在线
```

**为什么这两个问题不同：** 一个 reward 函数可以让 agent 在已知好轨迹上拿到高分，但在训练中给出错误的梯度信号，导致 agent 陷入局部最优。反之，一个 reward 在已知轨迹上排序混乱，但在线训练时可能引导出全新的、更好的行为。

**证据：**
```
v5 R5: TAGE=0.679, pref=0.951 → training_return=-2   (TAGE最高, 训练最差)
v5 R3: TAGE=0.654, pref=0.905 → training_return=+166 (TAGE较低, 训练最好)
```

**结构后果：** 框架本质上在选择"和老 reward 判断最一致的候选"，而不是"最有创新潜力的候选"。当 memory 中的轨迹全部来自框架自身产生的 reward 函数时，TAGE 变成了一个保守性过滤器——它惩罚偏离既有模式的创新。

### 缺陷 B：自指轨迹记忆的回音室效应

**架构问题：** Memory 中的所有轨迹都来自框架自身之前生成的 reward 函数的训练结果。框架从未见过由完全不同 reward 策略（如官方 reward、手工设计 reward、随机 reward）产生的轨迹。

```
Round 0: Mutator(LLM) → reward_0 → PPO → 轨迹集 T₀ → Memory
Round 1: Memory(T₀) → TAGE → reward_1 → PPO → 轨迹集 T₁ → Memory ∪ T₁
Round 2: Memory(T₀∪T₁) → TAGE → reward_2 → ...
```

**回音室效应：** TAGE 只能比较"reward_2 是否比 reward_1 更好地排序 T₀∪T₁ 中的轨迹"，而 T₀∪T₁ 中的轨迹都是 reward_0 和 reward_1 训练出来的。如果 reward_0 有系统性偏差（比如过分奖励悬停），那么所有后续轨迹都带有这个偏差，TAGE 会持续偏好"和 reward_0 一样有偏差"的候选。

**结构后果：** 框架缺乏"外部校准"——没有任何机制引入独立于框架自身 reward 函数的轨迹。Elite Archive 的 success_like count 排序也无法打破这个循环，因为 success_like 标签本身也是 LLM 基于框架自身轨迹判断的。

### 缺陷 C：Coverage 作为硬门控的双刃剑

**架构问题：** Coverage Analyzer 决定 TAGE 的"发言权"。当 coverage 判定为 `no_decision` 时，TAGE 的 preference 信号被完全忽略，selector 退化为 failure_avoidance + novelty 的加权和。

**设计意图：** 防止 memory 不足时 TAGE 给出虚假的高置信度排序。这是合理的防御性设计。

**结构问题：** Coverage 本身是 memory 质量的函数，而 memory 质量受上一轮选中候选的影响。这形成了一个级联依赖：

```
坏选则 → 坏轨迹(98% 失败) → coverage 退化 → TAGE 被降级 → 下一轮选则更差
```

**证据：** v5 R1 选中 comp_rec 产生了 98% 失败轨迹 → R2 的 coverage 从 weak_pairwise 掉到 failure_filter_only → R2 的 TAGE 偏好对被完全禁用（0 pairs）。

**结构后果：** Coverage 门控没有"惯性"或"平滑"机制。一轮坏选则可能导致下一轮的评估能力断崖式下降。这是硬阈值设计的固有缺陷——连续信号（memory 质量）被二值化为离散决策层级。

### 缺陷 D：TAGE 分数的结构性不可区分

**架构问题：** TAGE score 由 preference_consistency、failure_avoidance、component_alignment、novelty 加权融合而成。前三个子信号的计算方式导致它们在高 memory 量下必然收敛。

**收敛机制：**
- **preference_consistency:** 当 memory 中 partial_progress 占 80%+ 时，partial_progress > early_failure 偏好对的满足率主要取决于少数 early_failure 轨迹的 reward 值，而这些是相同的——因为所有候选都在同一批 memory 上评估。
- **failure_avoidance:** 归一化到 [0,1]，所有有效候选（通过了 validator）都能把失败轨迹的 reward 压到接近 min，因此 score 都在 0.94-0.99。
- **component_alignment:** 组件级的偏好一致性受整体偏好一致性的主导。

**结构后果：** 当所有候选都通过 validator 时，TAGE score 差异 < 0.05。Selector 实际依赖 novelty 打破平局，但 novelty 是 Pearson 相关系数——当所有候选都从同一 parent 变异而来时，相关系数自然接近 1.0（novelty ≈ 0.0）。框架在"无法有效排序"和"随机选择"之间摇摆。

### 缺陷 E：单父代 lineage 设计

**架构问题：** 每轮的 3 个候选都从同一个 parent code（Elite Archive best）变异而来。如果 parent 有根本性的设计缺陷（比如所有组件都是绝对位置奖励而非 delta），3 个孩子全部继承。

**为什么三个家族不够：** 三个家族（local_repair, component_recomposition, progress_conditioned）的差异主要在 LLM prompt 的措辞上，而非在算法层面的不同搜索策略。当 LLM 对三个家族的 prompt 产生相似输出时（常见于 LLM 的"安全偏好"），三个候选实际上高度相似。

**证据：** v5 中多轮出现 novelty ≈ 0.0（候选 reward 向量高度相关），说明即使不同家族，LLM 生成的代码在轨迹上的 reward 分布几乎一致。

**结构后果：** 框架缺乏真正的种群多样性机制——没有 crossover、没有 multi-parent、没有显式的多样性目标（novelty 只在 selector 中占 0.15-0.40 权重，且是候选间的相对度量）。

### 缺陷 F：Rollback 阈值设计与 TAGE 分数尺度不匹配

**架构问题：** Rollback 条件 `selected_score + threshold < best_score` 中，selected_score 和 best_score 都是 TAGE score（范围 0.64-0.70），而 threshold 默认 0.30。

**为什么永远不触发：**
```
best_score = 0.68 (精英)
selected_score = 0.64 (当前)
0.64 + 0.30 = 0.94 > 0.68  ← 永远不会触发
```

即使 adaptive threshold 降到 0.20：`0.64 + 0.20 = 0.84 > 0.68`，仍然不触发。

**证据：** v4 和 v5 共 18 轮，Rollback 触发 0 次。这是阈值尺度的设计错误——TAGE 分数的动态范围（~0.05-0.10）远小于阈值（0.20-0.30）。

**结构后果：** Rollback 是框架唯一的"纠错"安全网，但它从未工作过。框架在无保护状态下运行，坏选则直接污染 memory 且无法恢复。

### 缺陷 G：LLM 依赖的单点故障风险

**架构问题：** 4 个 Agent（Analyzer, Mutator, TrajectoryJudge, Reflector）依赖 LLM 调用。每个都有 fallback 到确定性逻辑的路径。但 fallback 质量显著低于 LLM 路径：

- **Mutator fallback:** 使用硬编码的模板公式（`_lunarlander_reward_body` 等），只有系数缩放，没有结构性创新
- **Analyzer fallback:** 基于规则的 diagnosis，无法理解复杂模式
- **TrajectoryJudge fallback:** 全部标为 ambiguous（等同于废弃该轨迹）
- **Reflector fallback:** 规则合成的 guidance，可能重复之前已失败的指导

**证据：** v5 中 R3, R4, R6, R9 各有 1 个候选验证失败（tage_score=-1.0），说明 LLM 生成的代码有时无法通过基础语法/运行时验证。retry 机制只有 1 次。

**结构后果：** 框架的演化能力上限由 LLM 质量决定。当 LLM 生成质量不稳定时，框架在"有效变异"和"废候选"之间随机游走，浪费了 1/3 的变异槽位。

### 缺陷 H：Memory 累积无遗忘机制

**架构问题：** `trajectory_cards.jsonl` 只追加不删除。来自早期轮次（reward 设计很差时）的轨迹永久保留在 memory 中，持续影响所有后续 TAGE 评估。

**为什么是问题：** Round 0 的 reward 通常质量最差（没有 trajectory 信息指导），产生的轨迹噪声最大。但这些早期轨迹在 TAGE 偏好对构建中与后期高质量轨迹权重相同。当 memory 增长到 500+ 轨迹时，早期噪声被大数定律稀释——但早期轨迹引入的系统性偏差（比如"所有轨迹都是 early_failure"）不会被稀释，因为它们是唯一的负样本来源。

**结构后果：** 缺乏 memory 衰减或遗忘机制意味着框架无法"忘记"早期探索阶段的错误信号。这与生物学习中的"遗忘曲线"和 RL 中的 experience replay prioritization 形成对比。

---

## 七、框架现存缺陷（操作层面）

以下为 v5 实验中直接观察到的操作层面缺陷，与上述结构性缺陷互补。

### 缺陷 1：TAGE 信号缺乏区分度

**现象：** 当 memory 中某种 label（如 partial_progress）占绝对多数时，所有候选 reward 在 "partial_progress > early_failure" 偏好对上都能拿到 0.94+ 的一致率。

**根因：** 偏好对的数量（上万对）远大于有效信息量（只有 2 种 label 的关系）。大量偏好对之间的差异被平均化，导致所有候选分数收敛到同一值。

**后果：** TAGE 无法区分候选优劣，selector 退回依赖 novelty（随机信号）。

### 缺陷 2：TAGE（离线）≠ 训练效果（在线）

（详见结构性缺陷 A）

### 缺陷 3：Memory 分布不平衡导致 coverage 回退

**现象：** R2 的 coverage 从 R1 的 `weak_pairwise_selection` 掉到 `failure_filter_only`，因为 margin check 失败。

**根因：** R1 选中的 comp_rec 产生了 98% 失败轨迹（23ef+26lps vs 仅 1 partial）。Memory 被失败轨迹主导，导致 label_margin check（需要正负样本的 progress 差距足够大）失败。

**后果：** 一次坏的选则污染了 memory，导致下一轮 TAGE 退化为只过滤失败（failure_filter_only），无法进行偏好对排序。框架缺乏"隔离坏结果"的机制。

### 缺陷 4：Rollback 从未触发

（详见结构性缺陷 F）

### 缺陷 5：Parent 选择仅依赖 trajectory labels，缺乏"保底"机制

**现象：** v5 所有 reward 都 succ=0，parent 排序完全由 partial_progress count → TAGE score 决定。R8（local_rep, part=50, score=0.826）成为 best，但它 training_return=4.7。

**根因：** 当所有候选在 success_like 维度上都是 0 时，排序退化到次要维度（partial_progress → TAGE score），但这些次要维度与训练效果无关。

### 缺陷 6：LLM 生成候选的验证失败率高

**现象：** R3, R4, R6, R9 各有 1 个候选验证失败（tage_score=-1.0），浪费了 1/3 的变异槽位。

**根因：** LLM 生成的代码有时有语法错误或运行时错误，retry 机制（1 次）不够。

### 缺陷 7：缺乏跨轮次的"保留好结果"机制

**现象：** v3 R5 有训练回报 +440，v5 R3 有 +166，但这些好的 reward 没有被"锁定"或优先微调。下一轮继续从 best() 选 parent（可能选到另一个），之前的好结果被丢弃。

**根因：** Elite Archive 只存 entry，不存"这个 reward 值得深入开发"的标记。Analyzer 不知道"上一轮训练效果很好，应该在这个 reward 基础上微调而非大改"。

---

## 八、修改效果总结

| 修改 | 目标 | 实际效果 |
|------|------|---------|
| Coverage ratio 0.15→0.03 | 让 TAGE 激活 | ✅ TAGE 在 8/9 轮激活，偏好对有效构建 |
| Selector 移除 static_score | 家族多样化 | ✅ 选中 3 种不同家族，不再只有 prog_cond |
| Elite Archive 清理 | 方法论正确 | ✅ rank key 不含训练回报 |
| — | **找到更好的 reward** | ❌ success_like=0，回报趋势退化 |

**核心矛盾：让框架"活过来"(TAGE激活、家族多样)之后，暴露了更深的问题——TAGE 作为一个离线评估器，其信号与在线训练结果之间存在根本性的 gap。**

---

## 九、设计改进方向

基于上述结构性缺陷分析，以下是几个值得考虑的方向（不改代码，仅供讨论）：

1. **在线-离线混合评估：** 不只用 TAGE score 选候选，加入 short training（如 20K steps 快速探针）作为在线信号。成本增加但可能解决缺陷 A。
2. **外部轨迹注入：** Bootstrap 阶段用官方 reward 或随机 reward 收集一批"外部"轨迹，打破缺陷 B 的回音室。
3. **Memory 衰减权重：** 给早期轮次的轨迹赋予指数衰减权重，减少早期噪声对当前 TAGE 的影响（修复缺陷 H）。
4. **Population-based 搜索：** 维护 3-5 个不同的 parent lineage，允许跨 lineage 的 crossover（修复缺陷 E）。
5. **Rollback 改为训练回报触发：** 用训练回报的相对下降（而非 TAGE score 的绝对差值）触发 rollback（修复缺陷 F）。
6. **Coverage 平滑门控：** 用连续权重替代离散决策层级，避免 coverage 断崖（修复缺陷 C）。
