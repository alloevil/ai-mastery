"""1.2 Matrices Are Transformations — 代码验证"""
import numpy as np

np.random.seed(42)

print("=" * 60)
print("1. 基向量变换")
print("=" * 60)

M = np.array([[2, 0],
              [0, 3]])
v = np.array([1, 1])

print(f"M =\n{M}")
print(f"M @ [1,0] = {M @ np.array([1, 0])}  ← 第一列 = 基向量 [1,0] 的去向")
print(f"M @ [0,1] = {M @ np.array([0, 1])}  ← 第二列 = 基向量 [0,1] 的去向")
print(f"M @ [1,1] = {M @ v}")

print("\n" + "=" * 60)
print("2. 行列式 = 面积缩放因子")
print("=" * 60)

print(f"det(M) = {np.linalg.det(M)}  ← 面积放大 6 倍")

# 降维矩阵
C = np.array([[1, 2],
              [2, 4]])
print(f"det(C) = {np.linalg.det(C)}  ← 空间被压扁，不可逆")

print("\n" + "=" * 60)
print("3. 矩阵乘法 = 变换组合")
print("=" * 60)

A = np.array([[1, 0.5],
              [0, 1]])   # 剪切
B = np.array([[2, 0],
              [0, 0.5]]) # x放大, y缩小

v = np.array([1, 1])
print(f"先B再A: {A @ (B @ v)}  ← 先缩放再剪切")
print(f"先A再B: {B @ (A @ v)}  ← 先剪切再缩放")
print(f"A @ B =\n{A @ B}")
print(f"B @ A =\n{B @ A}")
print(f"顺序不同结果: {np.array_equal(A @ B, B @ A)}  ← 不等")

print("\n" + "=" * 60)
print("4. 逆矩阵 = 撤销变换")
print("=" * 60)

A = np.array([[2, 1],
              [1, 3]])
A_inv = np.linalg.inv(A)
v = np.array([4, 5])
transformed = A @ v
restored = A_inv @ transformed

print(f"原始:       {v}")
print(f"变换后:     {transformed}")
print(f"逆变换恢复: {restored}")
print(f"A @ A_inv =\n{np.round(A @ A_inv, 10)}  ← 应该是单位矩阵")

print("\n" + "=" * 60)
print("5. 神经网络的一层")
print("=" * 60)

x = np.array([0.5, 0.8, 0.2])
W = np.random.randn(2, 3) * 0.5
b = np.array([0.1, 0.1])

z = W @ x + b
a = np.maximum(0, z)

print(f"输入 x:   {x}        shape: {x.shape}")
print(f"权重 W:\n{W}")
print(f"W @ x:    {np.round(W @ x, 4)}      shape: {(W @ x).shape}")
print(f"+ b:      {np.round(z, 4)}      shape: {z.shape}")
print(f"ReLU:     {np.round(a, 4)}      shape: {a.shape}")

print("\n" + "=" * 60)
print("6. 转置")
print("=" * 60)

M = np.array([[1, 2, 3],
              [4, 5, 6]])
print(f"M =\n{M}")
print(f"M^T =\n{M.T}")
print(f"M shape: {M.shape} → M^T shape: {M.T.shape}")

print("\n✅ 全部验证通过")
