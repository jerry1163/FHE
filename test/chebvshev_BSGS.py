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
        new_B = c1.v * c2.B + c2.v * c1.B
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
        new_B = (c1.B / self.Delta) + self.B_scale
        return MockCiphertext(new_val, new_scale, new_level, new_v, new_B)

    def mod_down(self, ctx, target_level):
        if ctx.level < target_level:
            raise ValueError(f"Cannot mod_down to higher level: {ctx.level} -> {target_level}")
        return MockCiphertext(ctx.value, ctx.scale, target_level, ctx.v, ctx.B)

    def add(self, c1, c2):
        if abs(math.log2(c1.scale) - math.log2(c2.scale)) > 0.1:
            raise ValueError(f"Scale Mismatch: {math.log2(c1.scale)} vs {math.log2(c2.scale)}")
        if c1.level != c2.level:
            target = min(c1.level, c2.level)
            if c1.level > target: c1 = self.mod_down(c1, target)
            if c2.level > target: c2 = self.mod_down(c2, target)
        new_val = c1.value + c2.value
        new_v = c1.v + c2.v
        new_B = c1.B + c2.B
        return MockCiphertext(new_val, c1.scale, c1.level, new_v, new_B)


# ==========================================
# 2. BSGS 优化切比雪夫评估
# ==========================================
def _compute_chebyshev_power_of_2(ctx_x, target_k, eval):
    """
    计算 T_{2^k}(x) 利用倍角公式 T_{2n} = 2 T_n^2 - 1
    """
    p = int(math.log2(target_k))
    curr = ctx_x  # T_1

    for _ in range(p):
        # T_{2n} = 2 * (T_n)^2 - 1
        sq = eval.multiply(curr, curr)
        sq = eval.rescale(sq)

        two_sq = eval.add(sq, sq)

        one = MockCiphertext(1.0, two_sq.scale, two_sq.level, 1.0, 0.0)
        neg_one = eval.multiply_const(one, -1.0)
        neg_one = eval.rescale(neg_one)

        if neg_one.level > two_sq.level: neg_one = eval.mod_down(neg_one, two_sq.level)
        curr = eval.add(two_sq, neg_one)

    return curr


def _recurse_chebyshev(ctx_x, coeffs, eval):
    """
    BSGS 核心递归: P(x) = U(x) * T_m(x) + V(x)
    """
    degree = len(coeffs) - 1

    # Base Case: 0阶
    if degree == 0:
        T0 = MockCiphertext(1.0, ctx_x.scale, ctx_x.level, 1.0, 0.0)
        res = eval.multiply_const(T0, coeffs[0])
        res = eval.rescale(res)
        return res

    # Base Case: 1阶
    if degree == 1:
        T0 = MockCiphertext(1.0, ctx_x.scale, ctx_x.level, 1.0, 0.0)
        term0 = eval.multiply_const(T0, coeffs[0])
        term0 = eval.rescale(term0)

        term1 = eval.multiply_const(ctx_x, coeffs[1])
        term1 = eval.rescale(term1)
        return eval.add(term0, term1)

    # Giant Step 分解
    k = math.floor(math.log2(degree))
    split_point = 2 ** k

    coeffs_R = coeffs[:split_point]  # 低阶部分 V(x)
    coeffs_Q = coeffs[split_point:]  # 高阶部分 U(x)

    # 递归计算 U(x) 和 V(x)
    res_R = _recurse_chebyshev(ctx_x, coeffs_R, eval)
    res_Q = _recurse_chebyshev(ctx_x, coeffs_Q, eval)

    # 计算 T_{split}
    T_pow2 = _compute_chebyshev_power_of_2(ctx_x, split_point, eval)

    # 利用性质: T_{m+j} = 2 T_m T_j - T_{|m-j|} 进行折叠
    # 这里使用简化版 BSGS: P = T_m * Q + R' (需调整系数)
    # 为了演示清晰，使用标准分解: Res = T_m * res_Q + res_R_corrected

    # 1. 2 * T_m * Q
    term_mult = eval.multiply(T_pow2, res_Q)
    term_mult = eval.rescale(term_mult)
    term_mult = eval.add(term_mult, term_mult)

    # 2. 折叠系数到 R
    new_coeffs_R = list(coeffs_R)
    for j, coef in enumerate(coeffs_Q):
        if j == 0: continue
        fold_idx = split_point - j
        if fold_idx < len(new_coeffs_R):
            new_coeffs_R[fold_idx] -= coef

    # 3. 重新计算 R'
    res_R_folded = _recurse_chebyshev(ctx_x, new_coeffs_R, eval)

    return eval.add(term_mult, res_R_folded)


def eval_chebyshev_bsgs(ctx_x, coeffs, eval):
    return _recurse_chebyshev(ctx_x, coeffs, eval)


# ==========================================
# 3. 混合方法：倍角公式恢复
# ==========================================
def hybrid_double_angle(ctx, r, eval):
    current_ctx = ctx
    for i in range(r):
        print(f"  [Hybrid] Iter {i + 1}/{r} (Level {current_ctx.level})")

        # y^2
        y_sq = eval.multiply(current_ctx, current_ctx)
        y_sq = eval.rescale(y_sq)

        # 2 * y^2
        two_y_sq = eval.add(y_sq, y_sq)

        # - 1
        one_ctx = MockCiphertext(1.0, two_y_sq.scale, two_y_sq.level, 1.0, 0.0)
        neg_one = eval.multiply_const(one_ctx, -1.0)
        neg_one = eval.rescale(neg_one)

        current_ctx = eval.add(two_y_sq, neg_one)
    return current_ctx


# ==========================================
# 4. 主程序
# ==========================================
if __name__ == "__main__":
    # 参数设置 (Han '20)
    log_p = 40
    L = 40  # 这里的层数足够 BSGS 跑很高阶
    cheb_degree = 30  # 30阶
    hybrid_r = 2  # 缩放次数 r=2

    evaluator = MockEvaluator(log_p, B_scale=2 ** (-40))

    # --- 1. 预计算系数 (在 [-1, 1] 上拟合缩放后的函数) ---
    print(f"--- 1. Fitting Coefficients (Degree {cheb_degree}) ---")
    nodes = np.linspace(-1, 1, 1000)
    # 目标函数: cos(2*pi * (x / 2^r))
    y = np.cos(2 * np.pi * (nodes / (2 ** hybrid_r)))
    coeffs = chebfit(nodes, y, deg=cheb_degree)
    print(f"Coeffs[0]: {coeffs[0]}")

    # --- 2. 加密输入 ---
    real_input = 0.1  # 真实输入 x
    # 直接传入真实值，视为已归一化到 [-1, 1]
    ctx_x = MockCiphertext(real_input, 2 ** log_p, L, 1.0, 2 ** (-20))
    print(f"\n--- 2. Encrypted Input: {ctx_x} ---")

    # --- 3. BSGS 评估 ---
    print("\n--- 3. Chebyshev Evaluation (BSGS) ---")
    approx_ctx = eval_chebyshev_bsgs(ctx_x, coeffs, evaluator)

    target_val = np.cos(2 * np.pi * (real_input / (2 ** hybrid_r)))
    print(f"Approximation Result: {approx_ctx.value:.8f} (Target: {target_val:.8f})")
    print(f"Remaining Level after BSGS: {approx_ctx.level}")

    # --- 4. 混合方法恢复 ---
    print("\n--- 4. Hybrid Method Recovery ---")
    final_ctx = hybrid_double_angle(approx_ctx, hybrid_r, evaluator)

    # --- 5. 验证 ---
    print(f"\n--- 5. Final Result ---")
    real_val = np.cos(2 * np.pi * real_input)
    print(f"Real Value : {real_val:.8f}")
    print(f"Decrypted  : {final_ctx.value:.8f}")
    print(f"Error      : {abs(final_ctx.value - real_val):.8f}")
    print(f"Final Level: {final_ctx.level}")