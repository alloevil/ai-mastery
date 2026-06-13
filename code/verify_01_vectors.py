"""
1.1 向量 — 代码验证
对应 docs/01-vectors.md 中的所有代码示例
"""
import numpy as np


def section_1_basic_vector():
    """基本向量操作"""
    print("=" * 50)
    print("1. 基本向量")
    print("=" * 50)

    v = np.array([3, 2])
    magnitude = np.linalg.norm(v)
    print(f"向量: {v}")
    print(f"长度: √(3² + 2²) = √13 = {magnitude:.4f}")
    print()


def section_2_embedding_similarity():
    """Embedding 向量的语义距离"""
    print("=" * 50)
    print("2. Embedding 语义距离")
    print("=" * 50)

    vocab = {"国王": 0, "王后": 1, "苹果": 2}
    embeddings = np.array([
        [0.2, -0.5, 0.8, 0.1, -0.3],      # 国王
        [0.18, -0.45, 0.82, 0.12, -0.28],  # 王后
        [-0.3, 0.7, 0.1, -0.6, 0.5],       # 苹果
    ])

    king = embeddings[vocab["国王"]]
    queen = embeddings[vocab["王后"]]
    apple = embeddings[vocab["苹果"]]

    print(f"国王↔王后: {np.linalg.norm(king - queen):.3f}")
    print(f"国王↔苹果: {np.linalg.norm(king - apple):.3f}")
    print()


def section_3_vector_subtraction():
    """向量减法 = 提取方向"""
    print("=" * 50)
    print("3. 向量减法 = 提取方向")
    print("=" * 50)

    king = np.array([0.2, -0.5, 0.8, 0.1, -0.3])
    queen = np.array([0.18, -0.45, 0.82, 0.12, -0.28])

    direction = queen - king
    print(f"国王→王后的方向: {direction}")
    print()


class Vector:
    """从零实现的 Vector 类（原课程练习）"""

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

    def angle_between(self, other):
        """练习题 1：返回两个向量之间的角度（度）"""
        cos = self.cosine_similarity(other)
        # clamp to [-1, 1] to avoid numerical errors
        cos = max(-1, min(1, cos))
        return np.degrees(np.arccos(cos))

    def __repr__(self):
        return f"Vector({self.components})"


def section_4_vector_class():
    """从零实现 Vector 类"""
    print("=" * 50)
    print("4. 从零实现 Vector 类")
    print("=" * 50)

    a = Vector([1, 2, 3])
    b = Vector([4, 5, 6])

    print(f"a + b = {a + b}")
    print(f"a · b = {a.dot(b)}")
    print(f"|a| = {a.magnitude():.4f}")
    print(f"cosine = {a.cosine_similarity(b):.4f}")
    print(f"角度 = {a.angle_between(b):.2f}°")
    print()


def section_5_exercises():
    """练习题验证"""
    print("=" * 50)
    print("5. 练习题")
    print("=" * 50)

    # 练习题 1：angle_between
    a = Vector([1, 0])
    b = Vector([0, 1])
    print(f"[1,0] 和 [0,1] 的角度: {a.angle_between(b):.1f}°")  # 90°

    a = Vector([1, 1])
    b = Vector([1, 1])
    print(f"[1,1] 和 [1,1] 的角度: {a.angle_between(b):.1f}°")  # 0°

    # 练习题 2：缩放矩阵
    scale = np.array([[2, 0], [0, 3]])
    v = np.array([1, 1])
    result = scale @ v
    print(f"\n缩放矩阵 [[2,0],[0,3]] × [1,1] = {result}")  # [2, 3]

    # 练习题 3：找最相似的两个向量
    np.random.seed(42)
    vectors = np.random.randn(5, 50)
    best_sim = -1
    best_pair = (0, 0)
    for i in range(5):
        for j in range(i + 1, 5):
            cos = np.dot(vectors[i], vectors[j]) / (
                np.linalg.norm(vectors[i]) * np.linalg.norm(vectors[j])
            )
            if cos > best_sim:
                best_sim = cos
                best_pair = (i, j)
    print(f"\n5 个 50 维向量中最相似的两个: #{best_pair[0]} 和 #{best_pair[1]}")
    print(f"余弦相似度: {best_sim:.4f}")
    print()


if __name__ == "__main__":
    section_1_basic_vector()
    section_2_embedding_similarity()
    section_3_vector_subtraction()
    section_4_vector_class()
    section_5_exercises()
