import numpy as np
import math
from numpy.polynomial.chebyshev import chebfit


# ==========================================
# 1. Mock HEAAN Library (模拟器)
# ==========================================
class MockCiphertext:
    def __init__(self, value, scale, level, v, B):
        self.value = value
        self.scale = scale
        self.level = level
        self.v = v
        self.B = B

    def __repr__(self):
        return (f"Ctx(Lvl={self.level}, Scale=2^{math.log2(self.scale):.1f}, "
                f"v={self.v:.4f}, B={self.B:.6f})")


class MockEvaluator:
    def __init__(self, log_p, B_scale):
        self.Delta = 2 ** log_p
        self.B_scale = B_scale

    def multiply(self, c1, c2):
        new_val = c1.value * c2.value
        new_scale = c1.scale * c2.scale
        new_level = min(c1.level, c2.level)
        new_v = c1.v * c2.v
        new_B = c1.v * c2.B + c2.v * c1.B #B_new=v1B2+v2B1+B1B2+B_mult,Table1，省略小量
        return MockCiphertext(new_val, new_scale, new_level, new_v, new_B)

    def multiply_const(self, c1, const_val):
        const_scale = self.Delta
        new_val = c1.value * const_val
        new_scale = c1.scale * const_scale
        new_v = c1.v * abs(const_val)
        new_B = c1.B * abs(const_val)
        return MockCiphertext(new_val, new_scale, c1.level, new_v, new_B)

    def rescale(self, c1):
        if c1.level <= 0:
            raise ValueError("Level exhausted! Cannot rescale.")
        new_val = c1.value
        new_scale = c1.scale / self.Delta
        new_level = c1.level - 1
        new_v = c1.v / self.Delta
        new_B = (c1.B / self.Delta) + self.B_scale#lemma2
        return MockCiphertext(new_val, new_scale, new_level, new_v, new_B)

    def mod_down(self, ctx, target_level):
        if ctx.level < target_level:
            raise ValueError(f"Cannot mod_down to higher level: {ctx.level} -> {target_level}")
        return MockCiphertext(ctx.value, ctx.scale, target_level, ctx.v, ctx.B)

    def add(self, c1, c2):
        if abs(math.log2(c1.scale) - math.log2(c2.scale)) > 0.1:
            raise ValueError(f"Scale Mismatch: {math.log2(c1.scale)} vs {math.log2(c2.scale)}")#可删去，防止忘记缩放
        if c1.level != c2.level:
            target = min(c1.level, c2.level)
            if c1.level > target: c1 = self.mod_down(c1, target)
            if c2.level > target: c2 = self.mod_down(c2, target)
        new_val = c1.value + c2.value
        new_v = c1.v + c2.v
        new_B = c1.B + c2.B
        return MockCiphertext(new_val, c1.scale, c1.level, new_v, new_B)


# ==========================================
# 2. 切比雪夫多项式评估
# ==========================================
def eval_chebyshev_poly(ctx_x, coeffs, eval):
    # --- 1. 初始化 T0, T1 ---
    # T0 = 1 (常数密文)
    T_prev2 = MockCiphertext(1.0, ctx_x.scale, ctx_x.level, 1.0, 0.0)
    # T1 = x
    T_prev1 = ctx_x

    # --- 2. 计算前两项: c0*T0 + c1*T1 ---
    T0_ctx = MockCiphertext(1.0, ctx_x.scale, ctx_x.level, 1.0, 0.0)
    term0 = eval.multiply_const(T0_ctx, coeffs[0])
    term0 = eval.rescale(term0)

    term1 = eval.multiply_const(ctx_x, coeffs[1])
    term1 = eval.rescale(term1)

    final_sum = eval.add(term0, term1)

    # --- 3. 递归计算 ---
    for k in range(2, len(coeffs)):
        # A. 2x
        two_x = eval.multiply_const(ctx_x, 2.0)
        two_x = eval.rescale(two_x)

        # B. 2x * T_{k-1}
        term_mult = eval.multiply(two_x, T_prev1)
        term_mult = eval.rescale(term_mult)

        # C. - T_{k-2} (修复 Scale 问题的关键步)
        neg_prev2 = eval.multiply_const(T_prev2, -1.0)
        neg_prev2 = eval.rescale(neg_prev2)  # 必须 Rescale 对齐 term_mult

        T_k = eval.add(term_mult, neg_prev2)

        # D. 累加
        term_k = eval.multiply_const(T_k, coeffs[k])
        term_k = eval.rescale(term_k)

        final_sum = eval.add(final_sum, term_k)

        # 更新
        T_prev2 = T_prev1
        T_prev1 = T_k

    return final_sum


# ==========================================
# 3. 混合方法：倍角公式恢复 (2y^2 - 1)
# ==========================================
def hybrid_double_angle(ctx, r, eval):
    """
    执行 r 次: y = 2 * y^2 - 1
    """
    current_ctx = ctx
    for i in range(r):
        print(f"  [Hybrid] Iter {i + 1}/{r} (Level {current_ctx.level})")

        # 1. y^2 (密文乘密文)
        y_sq = eval.multiply(current_ctx, current_ctx)
        y_sq = eval.rescale(y_sq)  # Scale: Delta^2 -> Delta, Level - 1

        # 2. 2 * y^2 (密文加法代替乘法，节省 Rescale，或者用 mult_const)
        # 用加法更省：2*A = A + A (不需要 Rescale)
        two_y_sq = eval.add(y_sq, y_sq)

        # 3. - 1
        # 构造常数 1 的密文，Scale 必须和 two_y_sq 一致
        # two_y_sq 的 Scale 是 Delta
        one_ctx = MockCiphertext(1.0, two_y_sq.scale, two_y_sq.level, 1.0, 0.0)

        # 此时 two_y_sq 的 Level 可能已经比 one_ctx 低了（因为 rescale）
        # 但 MockCiphertext 构造时 Level 是手动指定的，这里要小心
        # 用 mult_const(-1) 更稳健，它会自动处理 Scale 并 Rescale

        # result = 2*y^2 + (-1)
        # 构造 -1 常数
        neg_one = eval.multiply_const(one_ctx, -1.0)  # Scale -> Delta^2
        neg_one = eval.rescale(neg_one)  # Scale -> Delta, Level 匹配 two_y_sq

        # 相加
        current_ctx = eval.add(two_y_sq, neg_one)

    return current_ctx


# ==========================================
# 4. 主程序
# ==========================================

# 参数设置 (Han '20 Table 4 参数)
log_p = 40
# 我们需要足够的层级：
# 1. 切比雪夫递归 (30阶, 朴素算法): 约 30 层
# 2. 混合方法恢复 (r=2): 2 层 (主要是 y^2)
# 总共给 40 层比较安全
L = 40
cheb_degree = 30  # 实际在10阶左右就能做到小数点后8位精确，层数剩余约20层，30层最后仅剩余5层
hybrid_r = 2  # 缩放次数 r=2 (Scale down 4倍)

evaluator = MockEvaluator(log_p, B_scale=2 ** (-40))

# --- 1. 预计算系数 (在缩放后的区间拟合) ---
print(f"--- 1. Fitting Coefficients (Hybrid r={hybrid_r}) ---")
# 1. 直接在标准区间 [-1, 1] 上撒点
nodes = np.linspace(-1, 1, 1000)
y = np.cos(2 * np.pi * (nodes / (2 ** hybrid_r)))# 拟合的是 cos(2*pi * (x / 2^r))
coeffs = chebfit(nodes, y, deg=cheb_degree)

# --- 2. 加密输入 (直接使用原始输入) ---
real_input = 0.6666
ctx_x = MockCiphertext(real_input, 2**log_p, L, 1.0, 2**(-20))
print(f"\n--- 2. Encrypted Input : {ctx_x} ---")

# --- 3. 切比雪夫评估 (Evaluation) ---
print("\n--- 3. Chebyshev Evaluation ---")
approx_ctx = eval_chebyshev_poly(ctx_x, coeffs, evaluator)
target_val = np.cos(2 * np.pi * (real_input / (2**hybrid_r)))
print(f"Approximation Result: {approx_ctx.value:.6f} (Target: {target_val:.6f})")

# --- 4. 混合方法恢复 (Double Angle) ---
print("\n--- 4. Hybrid Method Recovery ---")
final_ctx = hybrid_double_angle(approx_ctx, hybrid_r, evaluator)# 通过 r 次 2y^2 - 1，还原回 cos(2 * pi * x)

# --- 5. 验证 ---
print(f"\n--- 5. Final Result ---")
real_val = np.cos(2 * np.pi * real_input)
print(f"Real Value : {real_val:.8f}")
print(f"Decrypted  : {final_ctx.value:.8f}")
print(f"Error      : {abs(final_ctx.value - real_val):.8f}")

if final_ctx.level < 0:
    print("WARNING: Levels exhausted!")
else:
    print(f"Remaining Levels: {final_ctx.level}")