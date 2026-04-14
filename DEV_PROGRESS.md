# 俄罗斯方块棋 - 开发进度

最后更新: 2026-04-15

---

## 当前主线状态

当前主线已经从“死区/封锁格”玩法切换为 Go-style 规则版本。

当前推荐主线:

1. 棋盘大小: `10x10`
2. 发牌方式: `bag7`
   中文: 共享七袋发牌
3. 终局判定: `pieces_only`
   中文: 仅比较终局存活棋子数
4. 主动跳过: `false`
   中文: 禁止主动跳过
5. 终局触发: `double_forced_pass`
   中文: 连续两次被动无着结束
6. 无合法着法处理: `reroll_once_then_pass`
   中文: 无合法着法时先重抽一次，再被动停手
7. 结算顺序: `capture_then_clear_once`
   中文: 先提子，再消行，不做第二次提子复查

死区主线思路已封存，作为未来扩展玩法候选，不再作为当前训练与规则优化的主线依据。

---

## 已完成的工作

### Phase 1: 核心玩法

- [x] 10x10 棋盘基础对弈系统
- [x] 7种标准俄罗斯方块 + 旋转/放置
- [x] 围棋式围杀机制
- [x] 俄罗斯方块消行机制
- [x] 人机/双人模式切换
- [x] 基础浏览器交互与棋谱导出

### Phase 2: 当前主线规则确立

已完成 Go-style 主线下的关键规则筛选:

1. 无合法着法处理
   - `reroll_once_then_pass` 优于 `pass_and_redraw`
2. 结算顺序
   - `capture_then_clear_once` 优于 `capture_then_clear_recheck`
   - 也优于当前测试中的 `clear_then_capture`
3. 发牌方式
   - `bag7` 明显优于 `bag7_independent`
4. 终局判定
   - 当前继续保留 `pieces_only`
   - 旧 `pieces_then_deadzones` / `area_like` 不再是 Go-style 主线下的有效候选

详细测试记录见 `TESTING_STATUS.md`。

### Phase 3: 训练与评估管线

- [x] AlphaZero-Lite: ResNet 6层, 128通道
- [x] 自对弈 + MCTS + 评估 + best checkpoint 流程
- [x] 结构化评估历史输出
- [x] head-to-head 评估支持
- [x] 浏览器部署所需模型导出能力

### Phase 4: 规则引擎性能优化

已完成多轮“语义等价”的性能优化:

- [x] 方块旋转与边界缓存
- [x] 合法落点范围裁剪
- [x] 自杀检查局部化
- [x] 提子检查局部化
- [x] 消行检查局部化
- [x] 候选区域标记优化

这些优化显著提升了 4070 Ti SUPER 上的自对弈吞吐。

### Phase 5: 一致性护栏

- [x] 新增 `cli/check_rule_consistency.py`
- [x] 优化后可对随机局面和抽样合法着法做参考实现对照
- [x] 已用 `100` 个随机局面通过一致性检查

推荐命令:

```powershell
python cli\check_rule_consistency.py --states 100 --move-checks 3 --seed 20260413
```

---

## 当前已知情况

### 1. 规则主线已切换

旧文档中大量“死区主线”结论已不再直接适用于当前主线。

当前主线是:

1. 提子后直接清空为 `EMPTY`
2. 不再围绕死区转化展开核心循环
3. 规则优化与训练比较应只基于 Go-style 主线重新判断

### 2. 性能已经大幅改善

在本地 4070 Ti SUPER 环境中，规则引擎优化后吞吐曾达到:

- `23.20 pos/s`

说明当前主瓶颈已经从“大范围整盘规则扫描”收缩到了更细碎的 Python 开销和环境波动问题。

### 3. Benchmark 环境波动需要注意

后续参数 sweep 中观察到同配置吞吐出现明显漂移，因此:

1. 不应仅凭一次 benchmark 结果就更换主训练参数
2. 更可靠的做法是:
   - 固定 case 连续复测
   - 保持系统环境稳定
   - 再决定最终参数

---

## 当前训练基线

当前推荐训练入口:

`run_train_4070tis.ps1`

当前脚本已经更新为 Go-style 主线规则:

1. `piece_distribution = bag7`
2. `terminal_mode = pieces_only`
3. `allow_voluntary_skip = false`
4. `end_condition_mode = double_forced_pass`
5. `no_legal_move_mode = reroll_once_then_pass`
6. `resolution_mode = capture_then_clear_once`

当前推荐的 4070 Ti SUPER 训练配比进一步更新为:

1. `games-per-iter = 24`
2. `num-simulations = 16`
3. `selfplay-parallel-games = 10`
4. `inference-batch-size = 40`
5. `batch-size = 512`
6. `lr = 0.0015`
7. `lr-step-size = 15`
8. `lr-gamma = 0.9`
9. `dirichlet-alpha = 0.05`
10. `dirichlet-epsilon = 0.10`
11. `train-steps-per-iter = 16`

这是当前“修复后的可信训练栈”下更稳的主训练基线。

### 4. Go-style 主线下仍有训练优化空间

围绕固定主线规则，已完成一轮定向参数复调短测:

1. 基线短测
   - `24 games/iter`
   - `24 sims`
   - `min_train_batches = 16`
   - `lr = 0.002`
   - 最佳评估胜率: `32.5%`
2. 强训练短测
   - `24 games/iter`
   - `24 sims`
   - `train_steps_per_iter = 32`
   - `lr = 0.0015`
   - 最佳评估胜率: `35.0%`
3. 强数据短测
   - `36 games/iter`
   - `24 sims`
   - `train_steps_per_iter = 24`
   - `lr = 0.0015`
   - 最佳评估胜率: `37.5%`

当前结论:

1. Go-style 主线还没有练满
2. 单纯多训有帮助，但“更多新自对弈样本 + 适度固定训练步数”更有效
3. 学习率从 `0.002` 下调到 `0.0015` 后，短测表现更稳定
4. 因此当前更值得推进的是训练参数复调，而不是继续新增规则复杂度

附加交叉验证:

1. `checkpoints_tune_go_more_data/best.pt`
2. `checkpoints_mainline_c_v2/best.pt`

在当前 Go-style 规则下做 `80` 局 head-to-head 时，前者对后者取得:

- `42.5%` 胜率
- `29` 负
- `17` 和

反向座次下，旧主配置最佳模型只有:

- `33.8%` 胜率

这说明新的训练配比已经表现出可见优势，尽管仍需要更长训练来进一步确认。

### 5. 更新后的长训练已经验证训练配比改进有效

基于更新后的主训练脚本，已完成一轮 `20` 轮长训练，关键评估结果如下:

1. Iter `5`: `35.0%`
2. Iter `10`: `41.7%`
3. Iter `15`: `41.7%`
4. Iter `20`: `43.3%`

这说明:

1. 新训练配比已经显著超过此前这条 Go-style 主线自己的长训练结果
2. 当前提升是真实的，不只是短测噪声
3. 在 `20` 轮时仍有上升迹象，因此还不能认为这条线已经训练饱和

### 6. 但当前 Go-style 最佳模型仍未稳定超过旧代表模型

固定种子复核对战结果:

1. 新 Go-style 最佳模型 vs 旧代表模型
   - `46-57-17` / `120` 局
   - 胜率 `38.3%`
2. 旧代表模型 vs 新 Go-style 最佳模型
   - `62-41-17` / `120` 局
   - 胜率 `51.7%`

对同一 heuristic Lv2 的固定评估:

1. 新 Go-style 最佳模型: `35.0%`
2. 旧代表模型: `39.2%`

当前更稳妥的结论是:

1. Go-style 主线训练参数优化已经成功
2. 但“新主线模型已经全面取代旧代表模型”这一点还不能下结论
3. 下一阶段应继续在固定规则下拉长训练预算，而不是再扩展新规则复杂度

### 7. 训练恢复语义已修正

`cli/train_alphazero.py` 已补强 checkpoint 机制:

1. 保存 `scheduler` 状态
2. 保存 `best_model_state`
3. `resume` 时恢复 scheduler
4. 明确提示 replay buffer 不会恢复

这意味着:

1. 旧的 `resume` 续训结果属于暖启动续训
2. 不能把此前的 `20 -> 40` 回落简单归因为模型本身后期必然退化

### 8. 中后期稳定性更值得继续优化学习率调度

基于固定规则，已完成一轮训练策略对比:

1. 基线组
   - `lr-step-size = 10`
   - `lr-gamma = 0.8`
   - `buffer-size = 100000`
   - 最佳评估胜率: `38.8%`
2. 慢衰减组
   - `lr-step-size = 15`
   - `lr-gamma = 0.9`
   - `buffer-size = 100000`
   - 最佳评估胜率: `41.2%`
3. 小回放池组
   - `lr-step-size = 10`
   - `lr-gamma = 0.8`
   - `buffer-size = 30000`
   - 最佳评估胜率: `36.2%`

当前结论:

1. 当前主线的中后期波动更像是学习率调度问题
2. 更慢、更平滑的学习率衰减比缩小 replay buffer 更有效
3. 因此主训练脚本已更新为慢衰减配置

### 9. 搜索正确性缺口已修正，但训练需要重新适配

已在 `cli/train_alphazero.py` 中修复三项会直接影响棋力的训练问题:

1. MCTS clone 现在复制 RNG 状态
2. 自对弈根节点支持 Dirichlet 探索噪声
3. 合法动作先验全零时改为均匀兜底

这意味着:

1. 当前搜索实现比之前更可信
2. 但修复后的训练动态和旧实验不再完全等价
3. 因此不能简单要求“修复后立刻复现旧 best 数字”

### 10. 根节点噪声应保留，但强度要更轻

后修复阶段已完成一轮噪声强度对比:

1. 标准噪声
   - `dirichlet-alpha = 0.3`
   - `dirichlet-epsilon = 0.25`
   - 最佳评估胜率: `40.0%`
2. 中等噪声
   - `dirichlet-alpha = 0.15`
   - `dirichlet-epsilon = 0.15`
   - 最佳评估胜率: `28.3%`
3. 轻噪声
   - `dirichlet-alpha = 0.05`
   - `dirichlet-epsilon = 0.10`
   - 最佳评估胜率: `45.0%`

当前结论:

1. 根节点噪声不是问题本身，问题在于之前默认值过重
2. 当前主线更适合更轻的探索噪声
3. 因此主训练脚本已切到轻噪声配置

### 11. 轻噪声正式 20 轮结果仍未超过旧高点

基于修复后的搜索实现 + 轻噪声配置，已完成一轮正式 `20` 轮训练:

1. Iter `5`: `29.2%`
2. Iter `10`: `30.8%`
3. Iter `15`: `30.0%`
4. Iter `20`: `34.2%`

这说明:

1. 轻噪声优于较重噪声
2. 但当前“修复后的可信训练栈”仍未重现旧版本的 `43.3%` 峰值
3. 下一阶段更像是需要继续做训练适配，而不是继续改规则

### 12. 修复后训练栈的更稳基线已经确认

围绕修复后的训练栈，又完成了一轮定向复调:

1. 基线
   - `24 games`
   - `16 sims`
   - `16 train_steps`
   - best: `40.0%`
2. 更多数据
   - `36 games`
   - `16 sims`
   - `16 train_steps`
   - best: `30.0%`
3. 更强搜索
   - `24 games`
   - `24 sims`
   - `16 train_steps`
   - best: `41.7%`
4. 更多训练
   - `24 games`
   - `16 sims`
   - `24 train_steps`
   - best: `41.7%`

随后又验证了把“更强搜索 + 更多训练”直接叠加到正式 `20` 轮长训并不稳定，最终 best 只有 `33.3%`。

当前更稳妥的处理是:

1. 保留修复后的搜索实现
2. 保留轻噪声和慢衰减
3. 先把默认训练入口保持在更稳的修复后基线，而不是直接推高所有参数

### 13. 长期目标已从“刷单一胜率”重新转向“策略成长型 AI”

现在已经明确:

1. `vs_random` 只是最低层 sanity check，不是长期目标
2. 最终目标是能和人类对弈、持续自我进化、策略越来越清晰的 AI
3. 因此后续工作的重点应从“单一 benchmark 数字”转向“策略成长与多维评估”

当前判断:

1. 现有 heuristic 对手仍有参考价值，但不能再作为唯一主指标
2. 后续更应关注:
   - 对历史 best 的自我进化
   - 固定策略探针局面的表现
   - 围杀、防守、取舍等策略行为是否更成熟
3. 训练方向应继续围绕“可信训练栈 + 多维评估”推进

---

## 下一步计划

### P0

1. 将训练评估从单一 benchmark 升级为多维评估
2. 在稳定的修复后基线上继续观察模型自我进化
3. 补充更接近人类对弈需求的策略探针与分层评估

### P1

4. 继续复查历史 best 与当前 best 的 head-to-head 演化
5. 评估是否需要设计更适合策略成长的训练报告与可解释指标

### P2

6. 将“死区主线”整理为独立扩展玩法分支
7. 继续推进网页端 AI 与 UX 改进

---

## 相关文件

```
TetrisWeiqi/
├── index.html
├── DEV_PROGRESS.md
├── TESTING_STATUS.md
├── run_train_4070tis.ps1
└── cli/
    ├── tetris_weiqi.py
    ├── train_alphazero.py
    ├── analyze_rules.py
    └── check_rule_consistency.py
```
