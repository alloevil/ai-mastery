# AI Mastery — 从零到精通

> 基于 [ai-engineering-from-scratch](https://github.com/rohitg00/ai-engineering-from-scratch)，逐节展开讲解，每个概念都对应到大模型中的真实应用。

## 定位

原课程的风格是简洁直觉 + 代码练习。本专题在此基础上：

1. **原文展开**：以原课程内容为主线，用中文展开讲透
2. **大模型融汇**：每个概念都对应到 Transformer、Attention、LoRA、RAG 等真实场景
3. **代码验证**：每节都有可运行的 Python 代码，动手验证

## 目录结构

```
01-math/       — 数学基础（线性代数、微积分、概率论...）
02-ml/         — 机器学习基础
03-dl/         — 深度学习核心
04-cv/         — 计算机视觉
05-nlp/        — 自然语言处理
07-transformers/ — Transformer 深度解析
10-llms/       — 大语言模型
...
```

每个主题下：
```
topic/
  docs/        — 逐节讲解（Markdown）
  code/        — 代码验证脚本
  outputs/     — 学习产出（笔记、总结）
```

## 当前进度

### 01 数学基础 — 线性代数直觉

对照原课程 `phases/01-math-foundations/01-linear-algebra-intuition/`：

| 小节 | 原课程标题 | 状态 |
|------|-----------|------|
| 1.1 | Vectors Are Points (and Directions) | ✅ |
| 1.2 | Matrices Are Transformations | ⬜ |
| 1.3 | The Dot Product Measures Similarity | ⬜ |
| 1.4 | Linear Independence | ⬜ |
| 1.5 | Basis and Rank | ⬜ |
| 1.6 | Projection | ⬜ |
| 1.7 | Gram-Schmidt Process | ⬜ |

## 使用方式

1. 先读原课程（`en.md`）
2. 再读本专题对应的小节（`docs/`）
3. 运行代码验证（`code/`）
4. 做练习题，写学习笔记（`outputs/`）
