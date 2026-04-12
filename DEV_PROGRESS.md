# 俄罗斯方块棋 - 开发进度

最后更新: 2026-04-13

---

## 已完成的工作

### Phase 1: 核心玩法 (已完成)

- [x] 10x10 棋盘基础对弈系统
- [x] 7种标准俄罗斯方块 + 旋转/放置
- [x] 围棋围杀机制 (四方向气判定)
- [x] 俄罗斯方块消行机制
- [x] 死区系统 (被围杀的方块转为死区)
- [x] 3级启发式AI (简单/中等/困难)
- [x] 移动端触控适配
- [x] 多棋盘尺寸支持 (8x8, 10x10, 12x12, 15x15)
- [x] 人机/双人模式切换

### Phase 2: 规则体系确立 (已完成)

通过系统性测试 (A/B/C/D对比、单参数消融、交叉实验) 确定了主线规则:

| 规则参数 | 确定值 | 重要度 |
|---------|--------|--------|
| 棋盘大小 | 10x10 | - |
| 发牌模式 | 共享 bag7 | 低 |
| 死区参与消行 | true | **关键** |
| 终局计分 | pieces_only | - |
| 主动跳过 | 禁止 | - |
| 终局条件 | double_forced_pass | **高** |
| 无合法着法处理 | reroll_once_then_pass | **高** |
| 结算顺序 | capture_then_clear_recheck | 低 |
| 死区转换时机 | immediate | 低 |
| 重抽次数 | 1 | 中 |

详细测试数据见 `TESTING_STATUS.md`。

### Phase 3: AlphaZero 训练 (基础完成)

#### 网络架构
- AlphaZero-Lite: ResNet 6层, 128通道, 3.17M参数
- 输入编码: 22通道 (5棋盘one-hot + 7当前方块 + 7 bag计数 + 1规则 + 2玩家指示)
- 输出: 策略头 (400维, 4旋转×10×10) + 价值头 (标量tanh)

#### 训练环境
| 环境 | 设备 | 最佳吞吐量 | 最佳胜率 |
|------|------|-----------|---------|
| Windows | RTX 4070 Ti SUPER | 12.23 pos/s | 40.8% (iter 20) |
| macOS | Apple M5 MPS | 21.23 pos/s (+73.5%) | 45.0% (iter 10) |

#### MPS 优化
- 启用 float16 autocast (此前仅CUDA支持)
- 禁用 GradScaler (MPS统一内存不需要)
- 优化并行/批量参数适配统一内存架构

#### 训练参数调优
- 大缓冲区 (200k) 降低过拟合回归: 从 -6.7% 降至 -1.8%
- 400局评估降低采样噪声, 实际模型能力约 38%

### Phase 4: 浏览器AI部署 (已完成)

- [x] ONNX模型导出 (opset 17, 单文件 12.2MB)
- [x] onnxruntime-web v1.21.0 集成
- [x] 22通道状态编码 JavaScript 实现 (与Python训练端一致)
- [x] bag7 共享发牌系统 JavaScript 实现
- [x] 神经网络AI选项 (Level 4: 神经网络)
- [x] Softmax采样推理 (temperature=0.5)
- [x] nextTurn() reroll逻辑与主线规则同步
- [x] 浏览器端实际验证通过

### Phase 5: 游戏体验功能 (进行中)

- [x] 棋谱记录系统 (全局 moveHistory 追踪)
- [x] 结算画面导出棋谱 (.txt文件下载)
  - 对局基本信息 (日期/棋盘/模式/AI等级)
  - 完整落子过程 (棋子/旋转/坐标/围杀/消行)
  - 终局棋盘ASCII图示
- [x] 初始加载render()报错修复 (board未初始化保护)

---

## 当前已知问题

### AI棋力不足

实际对局测试表明, 当前神经网络AI的策略性明显不足:
- 评估胜率仅约 38-45%, 无法稳定赢过随机AI
- 缺乏中心控制、围杀规划、防守意识
- 浏览器端推理无MCTS搜索, 纯网络直觉出招

**根本原因: 训练量级不足**

| 参数 | 当前值 | 有效训练估计 |
|------|--------|------------|
| 每轮自对弈局数 | 20-32 | 200-500 |
| MCTS模拟次数 | 24-50 | 200-800 |
| 训练迭代轮数 | 10-20 | 100-500 |
| 总自对弈局数 | ~200-400 | 数万-数十万 |

---

## 下一步计划

### 近期 (P0 - 提升AI棋力)

1. **大规模训练**
   - 目标参数: `--iterations 100 --games-per-iter 100 --num-simulations 200`
   - 预计在M5 MPS上需要数小时到十余小时
   - 目标: 评估胜率提升至 60%+ (稳定赢过启发式AI)

2. **浏览器端轻量MCTS**
   - 在onnxruntime-web推理基础上实现Web端MCTS搜索
   - 每步做20-50次模拟, 用WebWorker后台运行避免UI阻塞
   - 预期可显著提升落子质量

3. **推理优化**
   - 降低temperature (0.5→0.1) 减少随机性
   - 考虑ONNX模型量化 (float32→float16/int8) 加速浏览器推理
   - 添加"思考中..."状态提示

### 中期 (P1 - 游戏体验)

4. **落子动画与音效**
   - 方块放置动画
   - 围杀/消行特效
   - 基础音效系统

5. **AI复盘功能**
   - 对局结束后标注好手/疑问手
   - 展示AI推荐的最佳着法

6. **本地存档**
   - IndexedDB/localStorage保存对局状态
   - 悔棋功能

### 远期 (P2 - RPG与联网)

7. **迁移到游戏引擎 (Phaser 3)**
   - 为RPG系统打基础
   - 更丰富的动画和交互

8. **RPG系统原型**
   - 第一个职业实现 (先知: 预览下一个方块)
   - 剧情战役第一章 (10关)

9. **多人联网**
   - WebSocket实时对战
   - 排位匹配系统

---

## 项目文件结构

```
TetrisWeiqi/
├── index.html                  # 主游戏页面 (含浏览器AI)
├── DEV_PROGRESS.md             # 本文件 - 开发进度
├── PROJECT_PLAN.md             # 完整项目规划
├── TESTING_STATUS.md           # 规则测试详细记录
├── cli/
│   ├── tetris_weiqi.py         # 游戏核心引擎 (Python)
│   ├── train_alphazero.py      # AlphaZero训练脚本
│   ├── analyze_rules.py        # 规则分析工具
│   ├── benchmark_m5_mps.json   # M5 MPS基准测试结果
│   └── benchmark_mainline_c_v2.json  # 4070TiS基准测试结果
├── checkpoints_m5_mps/
│   ├── best.pt                 # 最佳模型检查点 (38MB)
│   ├── best_browser.onnx       # 浏览器部署ONNX模型 (12.2MB)
│   └── eval_history.jsonl      # 评估历史
├── checkpoints_mainline_c_v2/  # RTX 4070TiS训练结果
├── checkpoints_D_single_reroll/# D变体测试结果
└── checkpoints_mainline_large_buffer/ # 大缓冲区测试结果
```

---

*文档版本: v1.0 | 创建日期: 2026-04-13*
