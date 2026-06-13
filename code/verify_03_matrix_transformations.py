"""1.3 Matrix Transformations — 代码验证"""
import numpy as np

print("=" * 60)
print("1. 四种基本变换")
print("=" * 60)

# 旋转 45°
theta = np.pi / 4
R = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta),  np.cos(theta)]])
v = np.array([1, 0])
print(f"[1,0] 旋转45°: {np.round(R @ v, 4)}")
print(f"  det(R) = {np.linalg.det(R):.1f}  ← 面积不变")

# 缩放
S = np.array([[2, 0], [0, 0.5]])
print(f"[1,1] 缩放: {S @ np.array([1, 1])}")
print(f"  det(S) = {np.linalg.det(S):.1f}  ← 面积不变 (2×0.5=1)")

# 剪切
Sh = np.array([[1, 0.5], [0, 1]])
print(f"[1,1] 剪切: {Sh @ np.array([1, 1])}")
print(f"  det(Sh) = {np.linalg.det(Sh):.1f}  ← 面积不变")

# 镜像
Ref = np.array([[-1, 0], [0, 1]])
print(f"[2,1] 镜像: {Ref @ np.array([2, 1])}")
print(f"  det(Ref) = {np.linalg.det(Ref):.1f}  ← 方向翻转")

print("\n" + "=" * 60)
print("2. 组合变换：顺序不同结果不同")
print("=" * 60)

R90 = np.array([[0, -1], [1, 0]])
S = np.array([[2, 0], [0, 0.5]])
v = np.array([1, 0])

print(f"先旋转再缩放: {S @ (R90 @ v)}")
print(f"先缩放再旋转: {R90 @ (S @ v)}")
print(f"S @ R =\n{S @ R90}")
print(f"R @ S =\n{R90 @ S}")
print(f"相同? {np.array_equal(S @ R90, R90 @ S)}")

print("\n" + "=" * 60)
print("3. 特征值和特征向量")
print("=" * 60)

A = np.array([[2, 1], [1, 2]])

v1 = np.array([1, 1])
v2 = np.array([1, -1])
print(f"A @ [1,1] = {A @ v1},  3×[1,1] = {3 * v1}  ← 特征值 3")
print(f"A @ [1,-1] = {A @ v2},  1×[1,-1] = {1 * v2}  ← 特征值 1")

eigenvalues, eigenvectors = np.linalg.eig(A)
print(f"\nnp.linalg.eig 结果:")
print(f"  特征值: {eigenvalues}")
print(f"  特征向量:\n{eigenvectors}")

print("\n" + "=" * 60)
print("4. 特征分解 A = V @ D @ V⁻¹")
print("=" * 60)

A = np.array([[2, 1], [1, 2]], dtype=float)
eigenvalues, eigenvectors = np.linalg.eig(A)

V = eigenvectors
D = np.diag(eigenvalues)
V_inv = np.linalg.inv(V)

reconstructed = V @ D @ V_inv
print(f"原始矩阵:\n{A}")
print(f"重建矩阵:\n{np.round(reconstructed, 10)}")
print(f"相同? {np.allclose(A, reconstructed)}")

print("\n" + "=" * 60)
print("5. 练习：求 [[4,2],[1,3]] 的特征值")
print("=" * 60)

B = np.array([[4, 2], [1, 3]], dtype=float)
eigenvalues_B, eigenvectors_B = np.linalg.eig(B)
print(f"矩阵 B:\n{B}")
print(f"特征值: {eigenvalues_B}")
print(f"特征向量:\n{eigenvectors_B}")

# 验证
for i in range(2):
    v = eigenvectors_B[:, i]
    lam = eigenvalues_B[i]
    Av = B @ v
    lv = lam * v
    match = np.allclose(Av, lv)
    print(f"  λ={lam:.1f}: B@v = {np.round(Av, 4)}, λ*v = {np.round(lv, 4)} → 匹配={match}")

print("\n✅ 全部验证通过")
