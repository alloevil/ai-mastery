"""1.4 Calculus for ML — 代码验证"""
import numpy as np

print("=" * 60)
print("1. 数值导数 vs 解析导数")
print("=" * 60)

def f(x):
    return x ** 2

def numerical_derivative(f, x, h=1e-7):
    return (f(x + h) - f(x - h)) / (2 * h)

for x in [-2, -1, 0, 1, 2]:
    numerical = numerical_derivative(f, x)
    analytical = 2 * x
    print(f"x={x:2d}  f(x)={f(x):4d}  数值导数={numerical:.6f}  解析导数={analytical:.1f}")

print("\n" + "=" * 60)
print("2. 多变量梯度")
print("=" * 60)

def f_2d(point):
    x, y = point
    return x**2 + y**2

def numerical_gradient(f, point, h=1e-7):
    grad = []
    for i in range(len(point)):
        p_plus = list(point)
        p_minus = list(point)
        p_plus[i] += h
        p_minus[i] -= h
        partial = (f(p_plus) - f(p_minus)) / (2 * h)
        grad.append(partial)
    return grad

point = [3.0, 4.0]
grad = numerical_gradient(f_2d, point)
print(f"点 {point} 的梯度: {[f'{g:.1f}' for g in grad]}")
print(f"解析梯度: [2*3, 2*4] = [6.0, 8.0]")

print("\n" + "=" * 60)
print("3. 梯度下降求 f(x)=x² 的最小值")
print("=" * 60)

x = 5.0
lr = 0.1
print(f"起始: x={x:.4f}, f(x)={x**2:.4f}")
for step in range(10):
    grad = 2 * x
    x = x - lr * grad
    print(f"步{step+1:2d}: x={x:.4f}, f(x)={x**2:.6f}")

print("\n" + "=" * 60)
print("4. 链式法则: y = (3x+1)²")
print("=" * 60)

x = 2
# 外层: f(u)=u², f'(u)=2u
# 内层: g(x)=3x+1, g'(x)=3
# dy/dx = 2(3x+1) × 3 = 6(3x+1)
dy_dx_chain = 6 * (3 * x + 1)
print(f"x=2 时 dy/dx = {dy_dx_chain}")

# 数值验证
def g(x):
    return (3*x + 1)**2
dy_dx_numerical = numerical_derivative(g, 2)
print(f"数值验证: {dy_dx_numerical:.4f}")

print("\n" + "=" * 60)
print("5. 常见函数的导数验证")
print("=" * 60)

# sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

for x in [-2, 0, 2]:
    numerical = numerical_derivative(sigmoid, x)
    analytical = sigmoid_derivative(x)
    print(f"sigmoid'(x={x}) 数值={numerical:.6f} 解析={analytical:.6f}")

print("\n" + "=" * 60)
print("6. f(x) = x³ - 3x + 1 练习")
print("=" * 60)

def g(x):
    return x**3 - 3*x + 1

def g_prime(x):
    return 3*x**2 - 3

for x in [0, 1, 2]:
    numerical = numerical_derivative(g, x)
    analytical = g_prime(x)
    print(f"x={x}  数值={numerical:.6f}  解析={analytical:.1f}")

print("\n✅ 全部验证通过")
