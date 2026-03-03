import numpy as np
import math
from numpy.polynomial.chebyshev import chebfit


# ==========================================
# 1. Mock HEAAN Library (模拟同态加密环境)
# ==========================================
# 师兄，这个类是为了模拟 C++ HEAAN 库的行为
# 在真实环境中，这些是由库（如 SEAL, Lattigo）提供的
class MockCiphertext:
    def __init__(self, value, scale, level):
        self.value = value  # 真实值 (用于验证)
        self.scale = scale  # 缩放因子 Delta
        self.level = level  # 当前模数链层级

    def __repr__(self):
        return f"Ctx(val={self.value:.4f}, scale=2^{math.log2(self.scale):.1f}, lvl={self.level})"


class MockEvaluator:
    def __init__(self, log_p=40):
        self.p = 2 ** log_p  # Rescaling 基底 (Delta)

    def multiply(self, ctx1, ctx2):
        # 密文乘法：值相乘，Scale 相乘
        new_val = ctx1.value * ctx2.value
        new_scale = ctx1.scale * ctx2.scale
        # 层级取两者中较低的（实际库中通常要求层级一致）
        new_level = min(ctx1.level, ctx2.level)
        return MockCiphertext(new_val, new_scale, new_level)

    def multiply_const(self, ctx, const_val):
        # 密文乘常数：值相乘，Scale 变大 (因为 const_val 被视为编码后的整数多项式)
        # 实际上 const_val 应该先编码成 Plaintext，带有 scale P
        # 这里简化模拟：假设 const_val 已经包含了 scale P
        new_val = ctx.value * const_val
        new_scale = ctx.scale * self.p  # 常数也贡献了 scale
        return MockCiphertext(new_val, new_scale, ctx.level)

    def add(self, ctx1, ctx2):
        # 密文加法：要求 Scale 和 Level 必须一致
        if abs(math.log2(ctx1.scale) - math.log2(ctx2.scale)) > 0.1:
            raise ValueError(f"Scale mismatch in Add: {math.log2(ctx1.scale)} vs {math.log2(ctx2.scale)}")
        if ctx1.level != ctx2.level:
            # 在真实库中，这里需要手动做 mod_down (简单模约简)
            print(f"  [Warning] Level mismatch in Add ({ctx1.level} vs {ctx2.level}), simulating ModDown...")
            target_level = min(ctx1.level, ctx2.level)
            ctx1.level = target_level
            ctx2.level = target_level

        return MockCiphertext(ctx1.value + ctx2.value, ctx1.scale, ctx1.level)

    def rescale(self, ctx):
        # Rescaling: Scale 除以 P，Level 减 1
        if ctx.level <= 0:
            raise ValueError("Cannot rescale: Level is 0")
        new_val = ctx.value  # 值本身不变（在 CKKS 逻辑中，我们认为它是 m = val * scale）
        new_scale = ctx.scale / self.p
        new_level = ctx.level - 1
        return MockCiphertext(new_val, new_scale, new_level)

    def mod_down(self, ctx, target_level):
        # 简单模约简：只降级，不改 Scale
        if ctx.level < target_level:
            raise ValueError("Cannot mod_down to a higher level")
        return MockCiphertext(ctx.value, ctx.scale, target_level)


# ==========================================
# 2. Parameters & Setup
# ==========================================
log_N = 14
log_Q = 300
log_p = 40  # Scaling factor bits (Delta)
L = 10  # Max Level

# 初始化模拟器
evaluator = MockEvaluator(log_p)
scale = 2 ** log_p

# 目标函数：cos(2 * pi * x)
cheb_degree = 16  # 切比雪夫阶数 (Han '20 建议低阶)

# ==========================================
# 3. Pre-computation (Plaintext Domain)
# ==========================================
print("--- 1. Pre-computing Chebyshev Coefficients (Plaintext) ---")
# 在 [-1, 1] 区间拟合 cos(2*pi*x)
# 注意：Han '20 论文中输入 x 已经是缩放后的 (t/q - 0.25)，范围在 [-1, 1]
nodes = np.linspace(-1, 1, 1000)
target_y = np.cos(2 * np.pi * nodes)
coeffs = chebfit(nodes, target_y, deg=cheb_degree)

print(f"Coeffs (first 5): {coeffs[:5]}")
print("-" * 50)


# ==========================================
# 4. Ciphertext Evaluation Function (The Core)
# ==========================================
def eval_chebyshev_encrypted(ctx_x, coeffs, evaluator):
    """
    在密文域评估 Sum(c_k * T_k(x))
    利用递归: T_n = 2x * T_{n-1} - T_{n-2}
    """

    # T_0(x) = 1
    # 这是一个常数密文，Scale 应该与 ctx_x 一致以便加法
    # 在真实代码中，我们通常用一个全 1 的明文多项式
    T_prev2 = MockCiphertext(1.0, ctx_x.scale, ctx_x.level)

    # T_1(x) = x
    T_prev1 = ctx_x

    # 结果累加器: result = c_0 * T_0 + c_1 * T_1
    # 注意：c_0 乘常数后，Scale 会变成 Scale^2，需要处理
    # 这里为了模拟简单，我们假设 c_k 乘法后不 Rescale (因为是常数乘法，噪声增长小)
    # 但为了加法，Scale 必须对齐。

    # 初始化 sum = c0 * T0 + c1 * T1
    # c0 * T0
    term0 = evaluator.multiply_const(T_prev2, coeffs[0])
    term0 = evaluator.rescale(term0)  # 保持 Scale 为 Delta

    # c1 * T1
    term1 = evaluator.multiply_const(T_prev1, coeffs[1])
    term1 = evaluator.rescale(term1)

    # 累加 (需对齐 Level)
    if term0.level > term1.level:
        term0 = evaluator.mod_down(term0, term1.level)
    final_sum = evaluator.add(term0, term1)

    print(f"Iter 0-1: Result Level {final_sum.level}")

    # 递归计算 T_k
    for k in range(2, len(coeffs)):
        # 公式: T_k = 2 * x * T_{k-1} - T_{k-2}

        # 1. 计算 2x
        # x 是密文，2 是常数。 2 * x
        ctx_2x = evaluator.multiply_const(ctx_x, 2.0)
        ctx_2x = evaluator.rescale(ctx_2x)  # Scale 回到 Delta

        # 2. 计算 (2x) * T_{k-1}
        # 这是 密文 * 密文 -> Scale 变成 Delta^2 -> 需要 Rescale
        term_mult = evaluator.multiply(ctx_2x, T_prev1)
        term_mult = evaluator.rescale(term_mult)  # Scale 回到 Delta, Level - 1

        # 3. 减去 T_{k-2}
        # 问题：term_mult 的 Level 降低了 1，而 T_prev2 还是老 Level
        # 必须把 T_prev2 降级 (ModDown)
        if T_prev2.level > term_mult.level:
            T_prev2 = evaluator.mod_down(T_prev2, term_mult.level)

        # T_k = ...
        # 注意：减法其实就是加 (-1 * T_{k-2})
        T_k = evaluator.add(term_mult, evaluator.multiply_const(T_prev2, -1.0 / scale))  # 模拟减法

        # 4. 累加到总结果: sum += c_k * T_k
        term_k = evaluator.multiply_const(T_k, coeffs[k])
        term_k = evaluator.rescale(term_k)

        if final_sum.level > term_k.level:
            final_sum = evaluator.mod_down(final_sum, term_k.level)

        final_sum = evaluator.add(final_sum, term_k)

        # 更新递归项
        T_prev2 = T_prev1
        T_prev1 = T_k

        print(f"Iter {k}: T_k Level {T_k.level}, Sum Level {final_sum.level}")

    return final_sum


# ==========================================
# 5. Main Test Loop (Ciphertext Domain)
# ==========================================
print("\n--- 2. Starting Ciphertext Evaluation ---")

# 模拟输入数据 t
# Han '20 论文输入: x = (t/q - 0.25)
# 假设 t/q = 0.3, 那么 x = 0.05 (在 [-1, 1] 之间)
input_val = 0.05
print(f"Input x (normalized): {input_val}")
real_cos = np.cos(2 * np.pi * input_val)
print(f"True cos(2*pi*x): {real_cos:.6f}")

# 1. 加密 (Encoding + Encryption)
# 初始密文，Scale = Delta (2^40), Level = 10
ctx_x = MockCiphertext(input_val, scale, L)
print(f"Encrypted input: {ctx_x}")

# 2. 执行同态切比雪夫近似
# 这里包含了大量的 Mult, Rescale, ModDown
encrypted_result = eval_chebyshev_encrypted(ctx_x, coeffs, evaluator)

# 3. 解密与验证
print("\n--- 3. Decryption & Verification ---")
print(f"Final Ciphertext: {encrypted_result}")
print(f"Decrypted Value:  {encrypted_result.value:.6f}")
print(f"Error:            {abs(encrypted_result.value - real_cos):.6f}")

# 检查是否耗尽了层级
if encrypted_result.level < 0:
    print("\n[CRITICAL] Level exhausted! Need larger L or better algorithm (like BSGS).")
else:
    print(f"\n[SUCCESS] Remaining Levels: {encrypted_result.level}")