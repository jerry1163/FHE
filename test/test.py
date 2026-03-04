import numpy as np
import math


# ==========================================
# 辅助函数：数组移位与广义对角线提取
# ==========================================
def rotate_plaintext(vec, shift_steps):
    """
    对一维数组进行循环左移。正数向左，负数向右。
    """
    n = len(vec)
    shift_steps = shift_steps % n
    return np.concatenate((vec[shift_steps:], vec[:shift_steps]))


def extract_diagonal(matrix, k):
    """提取矩阵的第 k 条广义对角线。"""
    n = matrix.shape[0]
    diag = np.zeros(n, dtype=float)
    for i in range(n):
        diag[i] = matrix[i, (i + k) % n]
    return diag


# ==========================================
# 密文对象与同态指令模拟器
# ==========================================
class Ciphertext:
    def __init__(self, data, depth=0):
        self.data = np.array(data, dtype=float)
        self.depth = depth


class FheSimulator:
    def __init__(self):
        self.rotations = 0
        self.multiplications = 0
        self.key_switches = 0  # 新增：用于记录底层密钥切换的次数

    def homomorphic_add(self, ctx1, ctx2):
        new_data = ctx1.data + ctx2.data
        return Ciphertext(new_data, max(ctx1.depth, ctx2.depth))

    def homomorphic_constant_multiply(self, plain_vec, ctx):
        self.multiplications += 1
        new_data = plain_vec * ctx.data
        return Ciphertext(new_data, ctx.depth + 1)

    def homomorphic_rotate(self, ctx, shift_steps):
        n = len(ctx.data)
        shift_steps = shift_steps % n
        if shift_steps == 0:
            return ctx
        self.rotations += 1
        self.key_switches += 1

        new_data = rotate_plaintext(ctx.data, shift_steps)
        return Ciphertext(new_data, ctx.depth)


# ==========================================
# 核心：生成纯实数的稠密测试矩阵
# ==========================================
def generate_real_dense_matrix(N):
    mat = np.zeros((N, N), dtype=float)
    for i in range(N):
        for j in range(N):
            mat[i, j] = (i + 1) + (j + 1) * 0.1
    return mat


# ==========================================
# 策略一：Naive 朴素方法
# ==========================================
def eval_naive_matrix_mult(simulator, matrix, ctxt):
    N = matrix.shape[0]
    result_ctxt = None
    for k in range(N):
        diag = extract_diagonal(matrix, k)
        if np.all(np.abs(diag) < 1e-10):
            continue

        rot_ctxt = simulator.homomorphic_rotate(ctxt, k)
        mult_ctxt = simulator.homomorphic_constant_multiply(diag, rot_ctxt)

        if result_ctxt is None:
            result_ctxt = mult_ctxt
        else:
            result_ctxt = simulator.homomorphic_add(result_ctxt, mult_ctxt)
    return result_ctxt


# ==========================================
# 策略二：BSGS 大步小步法
# ==========================================
def eval_bsgs_matrix_mult(simulator, matrix, ctxt):
    """
    大步小步法 (Baby-Step Giant-Step)
    将线性的对角线循环重组为 2D 的大步和小步网格执行。
    """
    N = matrix.shape[0]

    # 计算网格维度，尽量接近 N 的平方根
    k1 = int(math.ceil(math.sqrt(N)))  # 外循环大步数量
    k2 = int(math.ceil(N / k1))  # 内循环小步数量

    print(f"\n[BSGS 调度器] 启动大步小步网格分配: {k1} 行 (Giant) x {k2} 列 (Baby)")

    # 为了直观展示，我们打印出这 N 条对角线是如何被塞进网格的
    grid = [[" " for _ in range(k2)] for _ in range(k1)]
    for i in range(k1):
        for j in range(k2):
            idx = i * k2 + j
            if idx < N:
                grid[i][j] = f"diag_{idx}"
            else:
                grid[i][j] = "空填充"

    for row_idx, row in enumerate(grid):
        print(f"  大步 {row_idx}: {row}")

    result_ctxt = None

    # ---------------------------------------------
    # 步骤 1: 预计算所有小步旋转并存起来
    # ---------------------------------------------
    v_rots = []
    for j in range(k2):
        v_rots.append(simulator.homomorphic_rotate(ctxt, j))

    # ---------------------------------------------
    # 步骤 2: 双层网格计算与大步旋转
    # ---------------------------------------------
    for i in range(k1):
        inner_sum_ctxt = None

        # 内循环：算大步 i 的内部累加
        for j in range(k2):
            k = i * k2 + j
            if k >= N:
                continue

            diag_k = extract_diagonal(matrix, k)
            if np.all(np.abs(diag_k) < 1e-10):
                continue

            # 明文对角线必须提前向右平移来进行预补偿
            diag_shifted = rotate_plaintext(diag_k, -(i * k2))

            # 使用已经复用好的小步旋转密文进行哈达玛乘积
            mult_ctxt = simulator.homomorphic_constant_multiply(diag_shifted, v_rots[j])

            if inner_sum_ctxt is None:
                inner_sum_ctxt = mult_ctxt
            else:
                inner_sum_ctxt = simulator.homomorphic_add(inner_sum_ctxt, mult_ctxt)

        # 外循环：对累加好的内层块，执行一次整体的跨越旋转
        if inner_sum_ctxt is not None:
            outer_rot_ctxt = simulator.homomorphic_rotate(inner_sum_ctxt, i * k2)

            if result_ctxt is None:
                result_ctxt = outer_rot_ctxt
            else:
                result_ctxt = simulator.homomorphic_add(result_ctxt, outer_rot_ctxt)

    return result_ctxt


# ==========================================
# 主流程：运行与对比
# ==========================================
def run_simulation():
    N = 8
    print(f"========== FHE 稠密矩阵仿真: Naive vs BSGS (N={N}) ==========\n")

    # 生成初始矩阵
    dense_matrix = generate_real_dense_matrix(N)

    # 打印初始进行乘法的矩阵
    print("【初始状态】进行乘法的初始稠密矩阵：")
    print(np.array_str(dense_matrix, precision=2, suppress_small=True))
    print("-" * 50)

    test_vec = np.arange(1, N + 1, dtype=float)
    initial_ctxt = Ciphertext(test_vec, depth=0)

    # ------------------------------------
    # 1. 运行 Naive
    # ------------------------------------
    print("\n【1】开始执行 Naive 方法...")
    sim_naive = FheSimulator()
    final_naive = eval_naive_matrix_mult(sim_naive, dense_matrix, initial_ctxt)

    print(f" -> 旋转次数: {sim_naive.rotations} 次")
    print(f" -> 乘法次数: {sim_naive.multiplications} 次")
    print(f" -> 底层触发 Key-Switch: {sim_naive.key_switches} 次")
    print(f" -> 消耗深度: {final_naive.depth} 层")

    # ------------------------------------
    # 2. 运行 BSGS
    # ------------------------------------
    print("\n【2】开始执行 BSGS 方法...")
    sim_bsgs = FheSimulator()
    final_bsgs = eval_bsgs_matrix_mult(sim_bsgs, dense_matrix, initial_ctxt)

    print(f"\n -> 旋转次数: {sim_bsgs.rotations} 次")
    print(f" -> 乘法次数: {sim_bsgs.multiplications} 次")
    print(f" -> 底层触发 Key-Switch: {sim_bsgs.key_switches} 次")
    print(f" -> 消耗深度: {final_bsgs.depth} 层\n")

    # ------------------------------------
    # 3. 校验正确性
    # ------------------------------------
    expected_result = np.dot(dense_matrix, test_vec)

    diff_naive = np.max(np.abs(final_naive.data - expected_result))
    diff_bsgs = np.max(np.abs(final_bsgs.data - expected_result))

    print("【正确性比对】")
    print(f"Naive 与标准误差: {diff_naive:.2e}")
    print(f"BSGS 与标准误差 : {diff_bsgs:.2e}")

    if diff_bsgs < 1e-10:
        print("\n结论：BSGS 在不影响层数的条件下成功成倍减少了极其耗时的底层密钥切换操作次数！")


if __name__ == "__main__":
    run_simulation()