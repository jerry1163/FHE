import math


# ==========================================
# 第一部分：模拟同态加密的底层宏观指令
# ==========================================

def homomorphic_add(vec1, vec2):
    """模拟同态加法，将两个密文向量对应元素相加，基本不消耗电路深度。"""
    return [a + b for a, b in zip(vec1, vec2)]


def homomorphic_constant_multiply(diag_vec, ctxt_vec):
    """模拟同态哈达玛乘积，将明文常数向量与密文向量对应元素相乘，消耗1层电路深度。"""
    return [plain * cipher for plain, cipher in zip(diag_vec, ctxt_vec)]


def homomorphic_rotate(ctxt_vec, shift_steps):
    """
    模拟同态密文循环移位。
    正数表示向左平移，负数表示向右平移。
    这是一个极其耗时且消耗特定旋转密钥的操作。
    """
    n = len(ctxt_vec)
    shift_steps = shift_steps % n
    return ctxt_vec[shift_steps:] + ctxt_vec[:shift_steps]


# ==========================================
# 第二部分：支持基数调节与混合大步小步法的核心引擎
# ==========================================

class TunableRadixHomomorphicDFT:
    def __init__(self, vector_length, radix):
        """
        初始化基数可调的引擎。

        参数解释：
        vector_length: 输入密文向量的长度 n。
        radix: 基数 r。控制我们把多少层稀疏矩阵合并在一起。
               比如 n=1024，radix=2，需要 10 层电路深度。
               如果 radix=4，就把每两层合并，只需要 5 层电路深度。
        """
        self.n = vector_length
        self.radix = radix
        # 计算合并后的总层数，即 log 以 radix 为底 n 的对数
        self.num_layers = int(math.log(self.n, self.radix))

        # 预先生成合并后带有等差数列规律的矩阵对角线
        self.merged_layers = self._precompute_merged_diagonals()

    def _precompute_merged_diagonals(self):
        """
        预计算引擎：模拟在明文状态下合并矩阵的过程。
        根据论文引理 3，合并 k 层稀疏矩阵后，新矩阵会包含最多 2r-1 条非零对角线，
        并且这些对角线的索引严格构成等差数列。
        """
        layers = {}
        for layer_idx in range(1, self.num_layers + 1):
            # 计算这一层对角线的固定等差步长 gap
            gap = self.n // (self.radix ** layer_idx)

            # 合并后对角线的总数约为 2 * radix - 1
            num_diagonals = 2 * self.radix - 1

            # 记录这 num_diagonals 条对角线的结构
            # 为了演示工程逻辑，我们用字典模拟，键是偏移的步数，值是生成的明文对角线向量
            diagonals = {}

            # 模拟生成等差数列排布的对角线（正向、负向及主对角线）
            half_count = num_diagonals // 2
            for step in range(-half_count, half_count + 1):
                offset = step * gap
                # 这里用全 1 向量模拟真实复数单位根对角线，主要为了展示控制流
                diagonals[offset] = [complex(1, 0)] * self.n

            layers[layer_idx] = {
                "gap": gap,
                "diagonals": diagonals,
                "num_diagonals": num_diagonals
            }

        return layers

    def bsgs_matrix_vector_multiply(self, layer_info, ctxt):
        """
        核心大步小步法（BSGS）执行器。
        专门用来极速处理带有等差数列对角线规律的矩阵与密文相乘。
        """
        gap = layer_info["gap"]
        diagonals = layer_info["diagonals"]
        t = layer_info["num_diagonals"]

        # 1. 动态计算网格大小 k1 (大步数量) 和 k2 (小步数量)
        # 我们希望 k1 和 k2 尽量接近 t 的平方根，这样整体旋转次数最少
        k1 = math.ceil(math.sqrt(t))
        k2 = math.ceil(t / k1)

        # 2. 提取所有存在的对角线偏移量，并按照从小到大排序
        offsets = sorted(list(diagonals.keys()))

        # 3. 核心补零策略：如果 k1 * k2 大于实际的对角线数量 t，
        # 我们就人为在网格里填入全零的假对角线，以凑齐完美的二维网格。
        padded_diagonals = []
        for i in range(k1 * k2):
            if i < t:
                padded_diagonals.append((offsets[i], diagonals[offsets[i]]))
            else:
                # 填充根本不需要平移的零向量
                padded_diagonals.append((0, [complex(0, 0)] * self.n))

        # 4. 开始执行大步小步法双层循环
        result_ctxt = [complex(0, 0)] * self.n

        # 外层循环：大步走 (Giant Steps)
        for i in range(k1):
            inner_sum_ctxt = [complex(0, 0)] * self.n

            # 内层循环：小步走 (Baby Steps)
            for j in range(k2):
                # 获取当前网格坐标对应的对角线信息
                grid_index = i * k2 + j
                offset, plain_diag = padded_diagonals[grid_index]

                # 小步旋转：将密文旋转所需的偏移量
                rotated_ctxt = homomorphic_rotate(ctxt, offset)

                # 哈达玛乘积：将提前算好且已经反向旋转对齐的明文对角线与密文相乘
                # 注意：真实的明文预处理会包含 rot_{-lk2i}(m)，这里为简化直接相乘
                multiplied_ctxt = homomorphic_constant_multiply(plain_diag, rotated_ctxt)

                # 将小步计算结果累加到内层求和器中
                inner_sum_ctxt = homomorphic_add(inner_sum_ctxt, multiplied_ctxt)

            # 大步旋转：将内层累加好的结果整体进行一次大跨度旋转
            # 大步的跨度是 k2 乘以基准步长 gap
            giant_step_offset = i * k2 * gap
            giant_rotated_ctxt = homomorphic_rotate(inner_sum_ctxt, giant_step_offset)

            # 将大步结果累加到最终结果中
            result_ctxt = homomorphic_add(result_ctxt, giant_rotated_ctxt)

        return result_ctxt

    def run_tunable_dft(self, ctxt):
        """
        启动基数可调的同态离散傅里叶变换。
        """
        current_ctxt = ctxt

        # 逐层应用合并后的矩阵乘法
        for layer_idx in range(1, self.num_layers + 1):
            layer_info = self.merged_layers[layer_idx]

            # 调用带有大步小步法的矩阵向量乘法来更新密文
            # 这一步会消耗 1 层极度宝贵的电路深度
            current_ctxt = self.bsgs_matrix_vector_multiply(layer_info, current_ctxt)

        return current_ctxt


# ==========================================
# 第三部分：执行与控制台输出代码
# ==========================================

if __name__ == "__main__":
    # 步骤 1：设置系统的基础参数
    # 为了方便我们在控制台查看完整的打印结果，这里我们不去跑 32768 这种巨大的数字
    # 我们将输入向量的长度设置得短一点，这里设置为 16
    vector_length = 16

    # 旋钮参数：设置基数为 2
    # 这代表我们使用最基础的拆解方式，不进行多层矩阵的明文合并
    radix = 2

    print("==================================================")
    print(f"系统启动：正在初始化基数可调的同态 DFT 引擎")
    print(f"当前配置 -> 向量长度: {vector_length}, 基数: {radix}")
    print("==================================================\n")

    # 步骤 2：实例化我们之前编写的核心引擎类
    dft_engine = TunableRadixHomomorphicDFT(vector_length, radix)

    # 步骤 3：构造模拟的密文数据
    # 在真实的同态加密环境中，这里应该传入由公钥加密生成的一大团乱码数据
    # 为了演示底层的数学流转过程，我们这里用纯明文的复数 1+0j 来模拟密文向量
    dummy_ciphertext = [complex(1, 0) for _ in range(vector_length)]

    print("【初始状态】原始密文向量的前 4 个元素如下：")
    for i in range(4):
        print(f"索引 {i}: {dummy_ciphertext[i]}")

    print("\n【执行运算】系统正在底层执行大步小步法矩阵乘法...")

    # 步骤 4：调用引擎的启动方法，开始一连串的稀疏矩阵连乘
    result_ciphertext = dft_engine.run_tunable_dft(dummy_ciphertext)

    # 步骤 5：在控制台展示最终的计算结果
    print("\n【运算完成】经过同态 DFT 变换后的密文向量前 4 个元素如下：")
    for i in range(4):
        # 提取实部和虚部进行格式化打印，方便查看
        real_part = result_ciphertext[i].real
        imag_part = result_ciphertext[i].imag
        print(f"索引 {i}: 包含实部 {real_part:.2f} 以及虚部 {imag_part:.2f}j")

    print("\n==================================================")
    print("所有同态计算流转测试完毕，进程安全退出。")
    print("==================================================")