# ASE-MTAGE 保守化重构计划

## 0. 文档目的

本文档用于记录 ASE-MTAGE 后续重构计划，说明每个模块为什么要改、应该怎么改、改完希望达到什么效果。

当前目标不是推翻已有 `ase_mtage/` 框架，也不是另建新仓库或新文件夹，而是在原有框架内部做简化、降权和纠偏，使系统从复杂的全自动闭环，收敛为更稳定、更容易验证的 conservative ASE-MTAGE（保守版 ASE-MTAGE）。

本轮重构的核心原则是：

```text
1. TAGE 不再从早期就承担“排序选最优 reward”的职责。
2. Memory 不再无限累加后直接喂给 TAGE，而是区分 full memory 与 working memory。
3. 轨迹标签分成通用粗标签和环境自适应细标签，避免写死 LunarLander。
4. Parent reward 不再训练完就自动晋升，必须通过保守晋升门控。
5. Analyzer 必须记住跨轮行为模式，而不是只看标签数量或 TAGE 分数。
6. Mutator 必须按失败重复程度逐步升级变异强度，避免每轮自由大改。
```

---

## 1. 当前框架的主要问题

### 1.1 TAGE 过早承担排序职责

当前框架中，Round 1 之后只要存在 trajectory memory，TAGE 就可能参与候选 reward 的选择。即使已经设计了 `decision_level`，但实际流程仍容易变成：

```text
生成 K=3 reward candidates
→ 用已有 trajectory memory 计算 TAGE 分数
→ Selector 根据 TAGE 派生分数选 top-1
→ 对 top-1 进行长训
→ 新轨迹继续进入 memory
```

如果早期 memory 很差，TAGE 的判断也会很不可靠。此时让 TAGE 排序，相当于把噪声反馈变成选择依据，容易导致错误 reward 被长训、错误轨迹继续污染下一轮 memory。

### 1.2 Memory 无限累加，失败轨迹容易淹没有效轨迹

当前完整 memory 会持续追加每一轮轨迹。如果前几轮 reward 都不好，memory 中会积累大量 early failure（早期失败）、low progress survival（低进展存活）、hover-like（悬停类）等失败轨迹。后续即使出现少量 partial progress（部分进展）或 success-like（类成功）轨迹，也可能在统计上被大量失败轨迹淹没。

因此，轨迹越多不一定越好。对 TAGE 来说，更重要的是 memory 的行为分布是否均衡、是否具有对比性、是否有足够高置信度样本。

### 1.3 粗标签不足以支持 Analyzer 诊断

目前通用粗标签包括：

```text
early_failure
low_progress_survival
partial_progress
success_like
ambiguous
```

这些标签适合 TAGE 构造弱偏好，但不足以让 Analyzer 判断具体失败行为。例如 `low_progress_survival` 可能对应 LunarLander 的悬停，也可能对应 BipedalWalker 的原地站立不前。它们在不同环境中的修复策略不同。

因此需要保留通用 coarse label（粗标签），同时引入由 LLM 根据环境自动生成的 fine label（细标签）或 behavior pattern（行为模式）。

### 1.4 Parent reward 晋升过于激进

当前训练过的 reward 可能根据标签数量或 TAGE score 进入 elite archive，并被作为下一轮 parent reward。若 trajectory label 错误，或者 partial progress 被误判，那么一个方向错误的 reward 可能被当成父奖励函数继续变异，导致搜索在错误局部区域内反复震荡。

需要引入 parent promotion gate（父奖励晋升门控），只有新 reward 明显优于当前 parent 时才允许晋升。

### 1.5 Mutator 变异过自由，搜索不连续

如果每轮都允许 LLM 自由重写 reward，系统很难判断行为变化来自哪个修改，也容易在不同 reward 结构之间乱跳。更稳定的方式是分阶段变异：先局部修复，重复失败后再结构重组，最后才进行阶段化 reward 或 reseed。

---

## 2. 总体重构目标

重构后的 ASE-MTAGE 应该满足以下目标：

```text
1. TAGE early stage 只做 failure filter，不做强排序。
2. TAGE ranking 只有在 memory 足够丰富、标签足够可靠后才开启。
3. Full memory 保留所有轨迹，working memory 给 TAGE 使用，避免失败轨迹淹没少量有效轨迹。
4. coarse_label 固定通用，fine_label 由 LLM 根据环境自适应生成。
5. Analyzer 以 behavior pattern 为核心做跨轮诊断。
6. Parent reward 只有明显改善才晋升，保持搜索连续性。
7. Mutator 根据 Analyzer 的 mutation escalation 控制变异强度。
8. Reflection 记录每轮“行为模式—reward 原因—修复尝试—结果”。
```

整体期望是让框架更保守、更可解释、更容易跑出稳定趋势，而不是每轮依赖高噪声自动选择。

---

## 3. TAGE 重构计划

### 3.1 为什么要改 TAGE

TAGE 本质上不应该是 ground-truth fitness（真实适应度函数）。它只能根据已有轨迹判断 candidate reward 是否可能重复已知失败，或者是否能把已知较好轨迹排在较差轨迹之前。

早期 memory 不充分时，TAGE 没有资格判断“哪个 reward 最好”。如果它过早排序，就会放大错误标签和错误 progress proxy。

### 3.2 怎么改

修改文件：

```text
ase_mtage/tools/memory_coverage.py
ase_mtage/tools/mtage_evaluator.py
ase_mtage/tools/selector.py
```

新增 TAGE 权限字段：

```json
{
  "tage_authority": "disabled | filter_only | weak_rank | strong_rank",
  "can_rank": false,
  "can_reject": true,
  "reject_candidate": false,
  "reject_reason": "",
  "tage_score": 0.0,
  "tage_confidence": 0.0,
  "recommended_use": "do_not_use | filter_only | auxiliary_rank | main_rank"
}
```

权限含义：

```text
disabled:
  TAGE 不参与选择，只记录诊断。

filter_only:
  TAGE 只能否决明显奖励已知失败模式的 candidate，不能排序。

weak_rank:
  TAGE 可以作为辅助排序信号，但不能单独决定选择。

strong_rank:
  TAGE 才能作为主要排序信号。
```

### 3.3 Selector 新规则

Selector 不再无脑选择最大 `tage_score`。

```text
tage_authority = disabled:
  不看 TAGE 分数，按静态策略或 mutation policy 选择。

tage_authority = filter_only:
  先剔除 reject_candidate=True 的候选；
  剩余候选按 mutation policy 选择；
  不按 TAGE 分数排序。

tage_authority = weak_rank:
  TAGE 只作为辅助分数，例如 20%～30% 权重。

tage_authority = strong_rank:
  TAGE 才能作为主要排序依据。
```

### 3.4 预期效果

```text
1. 早期 memory 不足时，避免 TAGE 过早选错 reward。
2. 只有失败轨迹时，TAGE 只负责过滤继续奖励失败的候选。
3. memory 中有 failure + partial progress 后，才允许弱偏好选择。
4. memory 中有足够 success-like / near-success 后，才允许强排序。
```

---

## 4. Memory 重构计划

### 4.1 为什么要改 Memory

完整 memory 需要保留，因为它对复现、画图和论文分析很重要。但 TAGE 不应该直接读取所有历史轨迹。否则失败轨迹过多会淹没少量有效轨迹。

因此需要区分：

```text
full memory:
  完整日志，所有轨迹都保留。

working memory:
  给 TAGE 和 Analyzer 当前轮使用，经过分层采样和限量控制。
```

### 4.2 怎么改

新增文件：

```text
ase_mtage/memory/working_memory.py
```

保留原始文件：

```text
memory/trajectory_cards.jsonl
```

新增工作记忆文件：

```text
memory/working_trajectory_cards.jsonl
memory/working_memory_summary.json
```

WorkingMemoryBuilder 逻辑：

```text
1. 读取 full trajectory_cards.jsonl。
2. 按 coarse_label 分组。
3. 按 fine_label / behavior_pattern 再分组。
4. 每个 coarse_label 最多保留 N 条。
5. 每个 fine_label 最多保留 M 条。
6. 最近轮次优先。
7. 高置信度优先。
8. success_like 和 partial_progress 稀有时尽量全保留。
9. ambiguous 可给 Analyzer 看，但不进入 TAGE preference pair。
```

建议默认参数：

```json
{
  "max_per_coarse_label": 30,
  "max_per_fine_label": 10,
  "recent_round_window": 3,
  "prefer_high_confidence": true,
  "keep_ambiguous_for_analyzer": true,
  "use_ambiguous_for_tage": false
}
```

### 4.3 Pipeline 接入方式

当前：

```text
TAGE 读取 memory/trajectory_cards.jsonl
```

改成：

```text
每轮开始先从 full memory 构建 working memory；
TAGE 读取 memory/working_trajectory_cards.jsonl；
Analyzer 可以读取 working_memory_summary + full memory 统计摘要。
```

### 4.4 预期效果

```text
1. 防止 early_failure 等失败轨迹数量过大，压制 partial_progress 信号。
2. 保持 memory 多样性，让 TAGE 看到更均衡的行为集合。
3. 让 Analyzer 能看到具体行为模式，而不是只看到粗标签数量。
4. 保留完整日志，不影响论文分析和复现实验。
```

---

## 5. 轨迹标签重构计划

### 5.1 为什么要改标签

粗标签适合 TAGE，但不适合 Analyzer 做具体诊断。比如 `low_progress_survival` 在不同环境中含义不同：

```text
LunarLander:
  可能是悬停不着陆。

BipedalWalker:
  可能是原地站立不前。

CartPole:
  可能是活得较久但接近失败边界。
```

因此需要区分通用 coarse label 和环境自适应 fine label。

### 5.2 coarse_label 固定通用

固定 coarse labels：

```text
early_failure
low_progress_survival
partial_progress
success_like
ambiguous
```

这些标签用于：

```text
MemoryCoverage
TAGE
Selector
```

### 5.3 fine_label 由 LLM 根据环境生成

修改文件：

```text
ase_mtage/agents/env_perception.py
ase_mtage/prompts/env_perception.md
```

让 EnvPerceptionAgent 输出：

```json
{
  "behavior_label_schema": [
    {
      "fine_label": "string",
      "definition": "string",
      "mapped_coarse_label": "early_failure | low_progress_survival | partial_progress | success_like | ambiguous",
      "observable_evidence": ["string"]
    }
  ]
}
```

示例，LunarLander 可以由 LLM 生成：

```text
fast_crash
hover_without_landing
approach_unstable
touchdown_unstable
near_success_landing
```

示例，BipedalWalker 可以由 LLM 生成：

```text
early_fall
standing_without_forward_progress
unstable_short_walk
stable_forward_walking
```

这些 fine labels 不是代码写死，而是由 LLM 根据 `env_manifest` 和 `task_manifest` 自动生成。

### 5.4 TrajectoryJudge 输出粗细标签

修改文件：

```text
ase_mtage/agents/trajectory_judge.py
ase_mtage/prompts/trajectory_judge.md
```

TrajectoryJudge 输入增加：

```text
behavior_label_schema
```

输出格式改成：

```json
{
  "coarse_label": "partial_progress",
  "fine_label": "approach_unstable",
  "fine_label_description": "The agent approaches the target but remains unstable.",
  "confidence": 0.78,
  "evidence_used": [],
  "use_for_tage_pair": true
}
```

### 5.5 预期效果

```text
1. TAGE 仍然使用通用粗标签，保证跨环境通用性。
2. Analyzer 使用 fine_label，能够识别具体失败模式。
3. 避免把 LunarLander 的 hover / crash 逻辑硬编码到框架里。
4. 后续扩展 BipedalWalker、CartPole 时，不需要重写标签代码。
```

---

## 6. Analyzer 重构计划

### 6.1 为什么要改 Analyzer

Analyzer 不能只读 coverage report 或 label counts。它应该真正分析跨轮行为变化：

```text
这一轮主导行为是什么？
是否和上一轮重复？
reward 修改后行为变好了还是变坏了？
哪个 reward component 可能奖励了失败行为？
是否需要升级 mutation 强度？
```

### 6.2 怎么改

修改文件：

```text
ase_mtage/agents/analyzer.py
ase_mtage/prompts/analyzer.md
```

新增输入：

```text
behavior_pattern_summary
round_behavior_summary
fine_label_counts
parent_reward_performance
previous_round_result
working_memory_summary
```

新增输出：

```json
{
  "behavior_diagnosis": {
    "dominant_failure_pattern": "hover_without_task_completion",
    "repeated_from_previous_round": true,
    "new_failure_pattern": false,
    "result_vs_parent": "worse | similar | better | unclear",
    "evidence": []
  },
  "mutation_escalation": {
    "recommended_level": "local_repair | component_recomposition | progress_conditioned | reseed",
    "reason": "same behavior pattern repeated for two rounds"
  }
}
```

### 6.3 mutation escalation 规则

```text
第一次出现失败：
  local_repair

同一失败连续两轮：
  component_recomposition

同一失败连续三轮：
  progress_conditioned

多轮仍失败且 parent 很差：
  reseed 或 rollback
```

### 6.4 预期效果

```text
1. Analyzer 能记住每轮行为模式，而不是只看标签数量。
2. LLM 不会每轮重复同样的修复建议。
3. 变异强度有依据，不再每轮自由大改。
4. 跨轮 memory 真正参与 reward evolution。
```

---

## 7. Mutator 重构计划

### 7.1 为什么要改 Mutator

当前如果 LLM 每轮都自由重写 reward，会导致搜索不连续，无法判断哪个修改带来了行为变化。需要让 Mutator 按 Analyzer 的 `mutation_escalation` 控制变异强度。

### 7.2 怎么改

修改文件：

```text
ase_mtage/agents/mutator.py
ase_mtage/prompts/mutator.md
```

Mutator 读取：

```text
analyzer_report.mutation_escalation.recommended_level
```

候选生成策略：

```text
local_repair:
  生成 3 个 local repair 候选；
  只改少量 component，保持主结构。

component_recomposition:
  生成 2 个 component recomposition + 1 个 local repair；
  允许删除、gate、替换部分 component。

progress_conditioned:
  生成 2 个 progress-conditioned + 1 个 component recomposition；
  引入阶段化或条件化 reward。

reseed:
  不继承 parent，重新生成 3 个 seed rewards。
```

### 7.3 禁止纯系数缩放

Prompt 增加要求：

```text
Do not only multiply all coefficients.
Every candidate must include at least one structural change:
- add/remove/gate a component
- condition a component on behavior stage
- change terminal handling
- separate progress and stability
```

中文含义：

```text
不要只把所有系数乘一个倍数。
每个候选必须包含至少一个结构性变化。
```

### 7.4 预期效果

```text
1. reward 搜索更连续。
2. 避免每轮大改导致行为随机漂移。
3. 当局部修复无效时，才升级到结构重组。
4. 让 LLM 的修改更可追踪、更可解释。
```

---

## 8. Parent Reward / Elite Archive 重构计划

### 8.1 为什么要改 Parent Reward

如果新 reward 训练完就直接进入 elite archive，并可能成为 parent，那么错误标签或偶然 partial_progress 会导致坏 reward 被继承。之后所有变异都围绕坏 parent 展开，容易长期不收敛。

### 8.2 怎么改

修改文件：

```text
ase_mtage/memory/elite_archive.py
ase_mtage/pipeline.py
```

新增 parent promotion gate：

```json
{
  "parent_promotion": {
    "promote": true,
    "reason": "partial_progress increased and early_failure decreased",
    "result_vs_parent": "better"
  }
}
```

### 8.3 晋升条件

满足任一正向条件：

```text
1. success_like 数量增加；
2. partial_progress 明显增加；
3. dominant failure pattern 消失；
4. mean behavior quality 改善；
5. Analyzer 判断 result_vs_parent = better，且给出证据。
```

同时不能违反：

```text
1. early_failure 明显增加；
2. 出现新的 dominant failure pattern；
3. TAGE 判定继续奖励已知失败；
4. label confidence 太低；
5. improvement 主要来自 ambiguous 轨迹。
```

### 8.4 未晋升 reward 也要记录

未晋升 reward 不应被丢弃，而应写入：

```text
failure_repair_memory
round_behavior_summary
candidate_attempt_log
```

### 8.5 预期效果

```text
1. 防止坏 reward 被错误晋升为 parent。
2. 保持 reward 搜索方向连续。
3. 让失败尝试成为 memory，而不是成为下一轮起点。
4. 减少 hover / crash 等失败模式之间的来回振荡。
```

---

## 9. Rollback 重构计划

### 9.1 为什么要改 Rollback

Rollback 不应只看分数，也应看行为模式。如果新 reward 带来新的 dominant failure pattern，即使某些离线分数看起来不错，也应该回退。

### 9.2 怎么改

修改文件：

```text
ase_mtage/tools/rollback.py
```

新增 rollback 条件：

```text
1. new_dominant_failure_pattern = true；
2. result_vs_parent = worse；
3. num_usable_tage_pairs = 0 且训练表现差；
4. early_failure_ratio 明显升高；
5. low confidence / ambiguous 轨迹占比过高。
```

### 9.3 回退后行为

```text
1. 新 reward 不晋升 parent；
2. 下一轮继续从旧 parent 变异；
3. 当前失败原因写入 failure-repair memory；
4. Analyzer 下一轮读取该失败案例。
```

### 9.4 预期效果

```text
1. 防止明显退化 reward 继续主导训练。
2. 让系统在错误方向上及时停止。
3. 增强 parent stability。
```

---

## 10. Pipeline 重构计划

### 10.1 为什么要改 Pipeline

Pipeline 是整个闭环的调度器。前面所有机制都需要在 pipeline 中接线，否则模块虽然存在，但实际不会改变实验行为。

### 10.2 新 Round 流程

```text
Round 0:
  LLM 生成 K=3 seed reward；
  静态选择一个；
  长训；
  构建 full memory 和 behavior summary。

Round 1:
  memory 通常不足；
  TAGE 不排序；
  Analyzer 根据 Round 0 behavior summary 做 local repair；
  Selector 选择安全候选长训。

Round 2:
  如果同一失败重复，升级 mutation；
  如果有 partial progress，TAGE 可做 weak filter；
  仍不强排序。

Round 3+:
  memory 足够多样后；
  TAGE 才逐步参与排序。
```

### 10.3 每轮新增文件

```text
round_behavior_summary.json
working_memory_summary.json
parent_promotion_report.json
candidate_attempt_log.jsonl
```

### 10.4 预期效果

```text
1. 每一轮的训练行为、选择原因和 parent 变化都可追踪。
2. TAGE 的权限随着 memory 质量逐步提升。
3. Analyzer 能利用前几轮行为模式做真正跨轮诊断。
4. 实验失败时更容易定位是哪一环出问题。
```

---

## 11. Prompt 重构计划

### 11.1 EnvPerception Prompt

新增要求：

```text
Generate environment-adaptive fine behavior labels.
Do not hard-code task-specific official reward.
Map each fine label to one coarse label.
```

中文含义：

```text
根据当前环境生成细粒度行为标签；
不要使用或推断官方奖励；
每个细标签必须映射到一个通用粗标签。
```

### 11.2 TrajectoryJudge Prompt

新增要求：

```text
Use behavior_label_schema.
Output both coarse_label and fine_label.
If evidence is insufficient, output ambiguous.
Do not force every trajectory into a useful category.
```

中文含义：

```text
使用环境自适应标签表；
同时输出粗标签和细标签；
证据不足就输出 ambiguous；
不要强行分类。
```

### 11.3 Analyzer Prompt

新增要求：

```text
Focus on behavior pattern transitions across rounds.
Compare current reward with parent reward.
Identify repeated failure patterns.
Recommend mutation escalation only when repeated failure is observed.
```

中文含义：

```text
关注跨轮行为模式变化；
比较当前 reward 和父 reward；
识别重复失败模式；
只有重复失败时才升级变异强度。
```

### 11.4 Mutator Prompt

新增要求：

```text
Follow mutation_escalation level.
Do not rewrite reward freely unless instructed.
Do not only scale coefficients.
Respect preserve_components and remove_or_gate_components.
```

中文含义：

```text
按照 Analyzer 给出的变异强度；
不要自由重写 reward；
不要只调系数；
必须遵守保留和删除组件列表。
```

---

## 12. Config 重构计划

### 12.1 新增 conservative mode

在 config 中新增：

```json
{
  "conservative_mode": {
    "enabled": true,
    "delay_tage_ranking": true,
    "use_working_memory": true,
    "enable_parent_promotion_gate": true,
    "max_per_coarse_label": 30,
    "max_per_fine_label": 10,
    "recent_round_window": 3
  }
}
```

### 12.2 正式实验要求

主实验配置必须满足：

```json
{
  "llm": {
    "enabled": true,
    "fallback_on_error": false,
    "api_key_env": "DEEPSEEK_API_KEY"
  }
}
```

注意：

```text
不要在 config 中写明文 API key。
只保留 api_key_env。
```

### 12.3 预期效果

```text
1. 可以一键开启保守版 ASE-MTAGE。
2. 主实验不允许静默 fallback。
3. 便于后续做 ablation，比如关闭 working memory 或关闭 parent promotion gate。
```

---

## 13. 实验计划

### 13.1 第一阶段：保守版短轮数验证

先不要直接跑 10 轮，先跑：

```text
Environment: LunarLander-v2
Full timesteps per round: 500000
Candidates per round: K=3
Rounds: 3～5
conservative_mode: true
```

观察重点：

```text
1. dominant_behavior_pattern 是否从 crash / hover 中逐步变化；
2. TAGE 是否没有过早强排序；
3. working memory 是否保持均衡；
4. parent reward 是否稳定；
5. Analyzer 是否识别重复失败；
6. mutation escalation 是否按预期逐步升级。
```

### 13.2 每轮重点检查文件

```text
round_behavior_summary.json
trajectory_judgment_summary.json
coverage_report.json
working_memory_summary.json
tage_summary.json
selection_report.json
parent_promotion_report.json
analyzer/self_evaluation.json
reflection/reflection.json
```

---

## 14. 优先级安排

### 第一批必须先改

```text
1. EnvPerception 输出 behavior_label_schema。
2. TrajectoryJudge 输出 coarse_label + fine_label。
3. EvidenceCard 保存 fine_label / behavior_pattern。
4. WorkingMemoryBuilder 分层采样。
5. TAGE 改成 filter-first，不能早期排序。
6. Selector 根据 tage_authority 选择。
```

解决的问题：

```text
标签不够细；
memory 被失败轨迹淹没；
TAGE 过早排序；
选择逻辑不稳。
```

### 第二批接着改

```text
7. 生成 round_behavior_summary.json。
8. Analyzer 读取跨轮 behavior pattern。
9. Mutator 根据 mutation_escalation 控制变异强度。
10. Parent promotion gate。
11. Rollback 加行为模式判断。
```

解决的问题：

```text
父奖励函数乱换；
变异不连续；
LLM 反复犯同样错误；
跨轮 memory 不够结构化。
```

### 第三批最后优化

```text
12. Config 增加 conservative_mode。
13. 画图统计 working memory / full memory 分布。
14. 消融实验配置。
15. 文档和论文描述更新。
```

---

## 15. 最小稳定闭环

重构后的最小稳定闭环为：

```text
EnvPerception 生成环境自适应 fine label schema
        ↓
Mutator 生成 K=3 reward
        ↓
LongTrainer 训练一个 reward
        ↓
EvidenceCard 记录轨迹证据 + coarse/fine label
        ↓
WorkingMemoryBuilder 分层采样
        ↓
MemoryCoverage 判断 TAGE 权限
        ↓
TAGE 前期只过滤，后期才排序
        ↓
Selector 选择候选
        ↓
Analyzer 总结行为模式和失败原因
        ↓
ParentPromotionGate 决定是否晋升父奖励
        ↓
Reflection 记录 failure-repair memory
```

---

## 16. 预期论文故事

本次重构后的论文故事可以更清晰地表述为：

```text
现有 LLM reward design 方法已经证明 LLM 能生成 reward function，
但轨迹反馈本身有噪声，过早使用 trajectory preference ranking 会放大错误反馈。

因此我们提出 conservative ASE-MTAGE：
1. 用环境自适应行为标签描述轨迹；
2. 用分层 working memory 防止失败轨迹淹没有效轨迹；
3. 让 TAGE 从 failure filter 逐步过渡到 preference ranker；
4. 用 parent promotion gate 保证 reward 搜索连续；
5. 用 failure-repair memory 记录跨轮行为模式和修复结果。
```

这种故事比“多 Agent + 多 memory + TAGE 打分”更稳，因为它清楚说明了问题来源：**轨迹反馈有噪声，TAGE 不能过早被当成真实 fitness。**

---

## 17. 最终目标

本轮重构最终希望达到：

```text
1. 框架更简单，不再所有模块同时强决策。
2. TAGE 更保守，先过滤，后排序。
3. Memory 更均衡，失败轨迹不会淹没有效轨迹。
4. 标签更通用，细标签由 LLM 根据环境生成。
5. Parent reward 更稳定，不再因噪声标签被错误晋升。
6. Analyzer 真正做跨轮行为诊断。
7. Mutator 修改更连续、更可解释。
8. 实验结果更容易分析，即使失败也能定位原因。
```

这份计划作为后续改代码和跑实验的总路线。