import math


class OpenFHE_Negacyclic_NTT:
    def __init__(self, modulus, root_of_unity_2n, ring_dim):
        """
        :param root_of_unity_2n: 2N 次单位根 (psi), 用于 X^N+1
        """
        self.modulus = modulus
        self.n = ring_dim

        # 1. 保存 2N 次单位根 (psi)
        self.psi = root_of_unity_2n

        # 2. 推导 N 次单位根 (omega = psi^2)
        # 蝴蝶运算用 N 次根
        self.root = (self.psi * self.psi) % self.modulus

        # 3. 预计算表 (核心 NTT 用)
        self.root_table = [0] * self.n
        self.root_table_inverse = [0] * self.n
        self._precompute_ntt_tables()

        # 4. 预计算 Negacyclic 扭转因子表 (psi^i)
        self.psi_powers = [0] * self.n
        self.psi_inv_powers = [0] * self.n
        self._precompute_psi_tables()

    def _reverse_bits(self, index, bit_length):
        res = 0
        for i in range(bit_length):
            if (index >> i) & 1:
                res |= (1 << (bit_length - 1 - i))
        return res

    def _mod_inv(self, a):
        return pow(a, -1, self.modulus)

    def _precompute_ntt_tables(self):
        """造核心 NTT 的乱序表 (omega)"""
        root_inv = self._mod_inv(self.root)
        msb = (self.n - 1).bit_length()
        x, x_inv = 1, 1
        for i in range(self.n):
            i_inv = self._reverse_bits(i, msb)
            self.root_table[i_inv] = x
            self.root_table_inverse[i_inv] = x_inv
            x = (x * self.root) % self.modulus
            x_inv = (x_inv * root_inv) % self.modulus

    def _precompute_psi_tables(self):
        # 这个表是自然序的，不需要位反转
        psi_inv = self._mod_inv(self.psi)
        p, p_inv = 1, 1
        for i in range(self.n):
            self.psi_powers[i] = p
            self.psi_inv_powers[i] = p_inv
            p = (p * self.psi) % self.modulus
            p_inv = (p_inv * psi_inv) % self.modulus

    # =============================================================
    # 核心蝴蝶运算
    # =============================================================
    def _core_forward_ntt(self, element):
        a = list(element)
        n = len(a)
        t = n // 2
        m = 1
        while m < n:
            for i in range(m):
                omega = self.root_table[i + m]
                start = i * (2 * t)
                for j in range(start, start + t):
                    lo = a[j]
                    omega_factor = (a[j + t] * omega) % self.modulus
                    a[j] = (lo + omega_factor) % self.modulus
                    a[j + t] = (lo - omega_factor + self.modulus) % self.modulus
            m <<= 1
            t >>= 1
        return a

    def _core_inverse_ntt(self, element):
        a = list(element)
        n = len(a)
        m = n // 2
        t = 1
        while m >= 1:
            for i in range(m):
                omega = self.root_table_inverse[i + m]
                start = i * (2 * t)
                for j in range(start, start + t):
                    lo = a[j]
                    hi = a[j + t]
                    a[j] = (lo + hi) % self.modulus
                    diff = (lo - hi + self.modulus) % self.modulus
                    a[j + t] = (diff * omega) % self.modulus
            m >>= 1
            t <<= 1
        # INTT 里的缩放 1/N
        n_inv = self._mod_inv(n)
        for i in range(n):
            a[i] = (a[i] * n_inv) % self.modulus
        return a

    # =============================================================
    # X^N + 1计算
    # =============================================================
    def forward_transform_negacyclic(self, element):
        """
        完整流程:
        1. 预乘 psi^i (Twist)
        2. 核心 NTT (Forward)
        """
        # a'[i] = a[i] * psi^i
        temp = [(x * self.psi_powers[i]) % self.modulus
                for i, x in enumerate(element)]

        return self._core_forward_ntt(temp)

    def inverse_transform_negacyclic(self, element):
        """
        完整流程:
        1. 核心 INTT (Inverse)
        2. 后乘 psi^-i (Untwist)
        """
        # 这里出来的结果已经是自然序的了
        temp = self._core_inverse_ntt(element)

        # a[i] = a'[i] * psi^-i
        return [(x * self.psi_inv_powers[i]) % self.modulus
                for i, x in enumerate(temp)]


# ==================== 验证环节 ===================
if __name__ == "__main__":
    # 参数调整
    N = 8
    # 约束: 必须是 2 的幂 (e.g., 8, 1024, 4096, ...)。
    # 含义: 多项式环的度数，如 Z_q[X] / (X^N + 1)。
    q = 17
    # q = k * 2N + 1 (其中 k 是正整数)
    psi = 3
    # 定义: 模 q 下的 2N 次原根。

    ntt = OpenFHE_Negacyclic_NTT(q, psi, N)

    input_vec = [0, 4, 3, 5, 5, 9, 7, 8]
    print(f"输入数据: {input_vec}")

    # 此时输出的不仅是频域，还是在 Negacyclic 域下的频域
    res_fwd = ntt.forward_transform_negacyclic(input_vec)
    print(f"Negacyclic NTT 结果: {res_fwd}")

    res_inv = ntt.inverse_transform_negacyclic(res_fwd)
    print(f"还原结果: {res_inv}")

    assert res_inv == input_vec, "验证失败"
    print("验证成功")