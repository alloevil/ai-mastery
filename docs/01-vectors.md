# 1.1 向量 = 空间中的点（也是方向）

> 原课程：`phases/01-math-foundations/01-linear-algebra-intuition/docs/en.md`
> 主题：Vectors Are Points (and Directions)

---

## 原课程核心观点

> A vector is just a list of numbers. But those numbers mean something — they're coordinates in space.

向量就是一组数。但这组数有意义——它们是空间中的坐标。

---

## 展开讲：两个视角，同一个东西

向量可以从两个角度看：

**代数视角：** `[3, 2]` 就是两个数，按顺序排列。

**几何视角：** `[3, 2]` 是平面上从原点 (0,0) 到 (3,2) 这个点的一个箭头。

这两个视角是等价的。`[3, 2]` 既是"两个数"，也是"空间中的一个位置"和"一个方向"。

```python
import numpy as np

# 2D 向量
v = np.array([3, 2])

# 长度（模）= 原点到这个点的距离
magnitude = np.linalg.norm(v)
print(f"向量: {v}")
print(f"长度: √(3² + 2²) = √13 = {magnitude:.4f}")
```

```
向量: [3 2]
长度: √(3² + 2²) = √13 = 3.6056
```

**向量有两个身份：**

- **位置向量**：从原点到这个点 → "它在哪里"
- **方向向量**：从 A 到 B → "它指向哪里"

`[3, 2]` 作为位置向量，表示空间中 (3,2) 这个点。作为方向向量，表示"向右 3，向上 2"这个方向。

---

## 原课程的 AI 映射

> In AI, vectors represent everything:
> - A word → a vector of 768 numbers (its "meaning" in embedding space)
> - An image → a vector of millions of pixel values
> - A user → a vector of preferences

展开讲：

### 一个词 → 768 维向量

模型不认识文字。第一步是把每个词变成一组数——这就是 Embedding 层做的事。

```python
# Embedding 本质上是一个大查找表：词表大小 × 向量维度
# GPT-2: 50257 × 768
# GPT-3: 50257 × 12288

# 假设我们有一个简化版
vocab = {"国王": 0, "王后": 1, "苹果": 2}
d_model = 5

# 每个词对应一行向量（训练出来的）
embeddings = np.array([
    [0.2, -0.5, 0.8, 0.1, -0.3],      # 国王
    [0.18, -0.45, 0.82, 0.12, -0.28],  # 王后
    [-0.3, 0.7, 0.1, -0.6, 0.5],       # 苹果
])

king = embeddings[vocab["国王"]]
queen = embeddings[vocab["王后"]]
apple = embeddings[vocab["苹果"]]

# 语义相近的词，向量距离近
print(f"国王↔王后: {np.linalg.norm(king - queen):.3f}")  # 0.066
print(f"国王↔苹果: {np.linalg.norm(king - apple):.3f}")  # 1.887
```

```
国王↔王后: 0.066
国王↔苹果: 1.887
```

**为什么语义相近的词向量也相近？**

因为模型在训练时，"国王"和"王后"出现在相似的上下文里（比如"___下令"、"___加冕"）。模型为了让预测更准，就把它们放在向量空间中相近的位置。

这就是 **Distributional Hypothesis**（分布假说）：一个词的含义由它的上下文决定。出现在相似上下文中的词，含义相似，向量也相似。

### 一张图 → 百万维向量

一张 224×224 的 RGB 图片，展平后就是 224×224×3 = 150528 维的向量。模型在这个高维空间里学习"猫"和"狗"的区别。

实际中不会真的展平——CNN 用卷积核提取局部特征，逐步压缩成更小的向量。最终一张图变成一个 2048 维的向量（比如 ResNet 的最后一层），这个向量编码了图片的全部语义信息。

### 一个用户 → 偏好向量

推荐系统把用户表示为向量：`[喜欢科幻, 喜欢动作, 年龄段, ...]`。用户向量和物品向量的点积/余弦相似度越高，推荐得分越高。

---

## 向量运算的语义

原课程后面会讲点积和余弦相似度，但这里先给一个直觉：

```python
# 向量减法 = 提取方向
king_to_queen = queen - king
print(f"国王→王后的方向: {king_to_queen}")
```

```
国王→王后的方向: [-0.02  0.05  0.02  0.02  0.02]
```

`国王 - 男人 + 女人 ≈ 王后` 不是魔法，是向量空间中的几何运算。减法提取了"从男性到女性的方向"，加法沿着这个方向移动。

---

## 大模型中向量的完整流转

这是原课程没讲但你需要理解的：

```
用户输入 "国王"
    ↓
Tokenizer: "国王" → token ID [1234]
    ↓
Embedding: [1234] → 取出第 1234 行 → [0.2, -0.5, 0.8, ...] (768维)
    ↓
+ 位置编码: 告诉模型这个词在第几个位置
    ↓
进入 Transformer 第 1 层 → 输出 768 维向量
    ↓
进入 Transformer 第 2 层 → 输出 768 维向量
    ↓
...（重复 96 层）...
    ↓
最后一层输出 768 维向量
    ↓
LM Head (768 → 50257): 变成词表大小的向量
    ↓
Softmax: 变成概率分布
    ↓
采样: 选出概率最高的 token → "王后"
```

**每一步输入输出都是向量。** 向量是模型内部流动的"货币"。理解了向量在每一层是怎么变换的，就理解了大模型在做什么。

---

## 原课程代码练习

原课程让你自己实现一个 Vector 类：

```python
class Vector:
    def __init__(self, components):
        self.components = list(components)
        self.dim = len(self.components)

    def __add__(self, other):
        return Vector([a + b for a, b in zip(self.components, other.components)])

    def __sub__(self, other):
        return Vector([a - b for a, b in zip(self.components, other.components)])

    def dot(self, other):
        return sum(a * b for a, b in zip(self.components, other.components))

    def magnitude(self):
        return sum(x**2 for x in self.components) ** 0.5

    def normalize(self):
        mag = self.magnitude()
        return Vector([x / mag for x in self.components])

    def cosine_similarity(self, other):
        return self.dot(other) / (self.magnitude() * other.magnitude())

    def __repr__(self):
        return f"Vector({self.components})"


# 测试
a = Vector([1, 2, 3])
b = Vector([4, 5, 6])

print(f"a + b = {a + b}")           # [5, 7, 9]
print(f"a · b = {a.dot(b)}")        # 32
print(f"|a| = {a.magnitude():.4f}") # 3.7417
print(f"cosine = {a.cosine_similarity(b):.4f}")  # 0.9746
```

**为什么从零实现？** 因为 NumPy/PyTorch 的一行代码 `np.dot(a, b)` 背后就是这些循环。你手写一遍，就知道 `@` 运算符到底在做什么。

---

## 小结

| 原课程要点 | 大模型里的对应 |
|-----------|-------------|
| 向量是一组数 | Embedding、隐藏状态、输出 logits |
| 向量是空间中的点 | 语义空间中的一个位置 |
| 向量是方向 | 语义关系（国王→王后） |
| AI 里一切都可以表示为向量 | 模型内部流动的全部是向量 |

---

## 练习题

1. 实现 `Vector.angle_between(other)` — 返回两个向量之间的角度（度）
2. 创建一个 2D 缩放矩阵，x 方向放大 2 倍，y 方向放大 3 倍，作用到向量 [1, 1] 上
3. 给定 5 个随机的 50 维向量，用余弦相似度找出最相似的两个

---

下一节：[1.2 Matrices Are Transformations](02-matrices.md)
