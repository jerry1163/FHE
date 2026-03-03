import numpy as np
import matplotlib.pyplot as plt
import math
from numpy.polynomial.chebyshev import chebfit, chebval


# =========================
# Parameters
# =========================
q = 2**32     # modulus
iteration = 1     # number of tests
taylor_order = 12  # Taylor expansion terms (higher = better)
cheb_order = 52   # Chebyshev polynomial degree
scale = 1
sin_r = 7
cos_r = 5
sin_e = 0
cos_e = 0


# =========================
# Main test loop
# =========================
for it in range(iteration):
    m = np.linspace(0, 1, 100)
    I0 = 0
    while (I0 == 0):
        I0 = np.random.randint(-12, 12)
    t = m + q * I0

    # --------------------------
    # Method 1: Taylor Approximation
    # --------------------------
    x = 2 * np.pi * t / q
    x_r = x / (2**sin_r)  # reduce range by 2^r
    
    # Use Euler's formula and Taylor expansion to approximate sin(x)
    # Step 1: Compute e^(ix) using Taylor series
    taylor_x = np.zeros_like(x_r, dtype=np.complex128)
    for n in range(taylor_order):
        term = (1j * x_r) ** n / math.factorial(n)
        taylor_x += term
    # Rescale back
    for _ in range(sin_r):
        taylor_x = taylor_x * taylor_x
        
    # Step 2: Compute the conjugate e^(-ix)
    taylor_neg_x = np.conj(taylor_x)
    
    # Step 3: Compute sin(x) = (e^(ix) - e^(-ix)) / (2i)
    sin_x = (taylor_x - taylor_neg_x) / (2j)
    sin_x = np.real(sin_x)
    
    # Step 5: Mult the coeff
    sin_x = q / (2 * np.pi) * sin_x
    
    for i in range(5):
        print(f"Message {i}: m={m[i]:.6f}, Taylor sin={sin_x[i]:.6f}, error={abs(sin_x[i]-m[i]):.6f}")
    sin_e += np.mean(np.abs(sin_x - m))
    print()

     # --------------------------
    # Method 2: Chebyshev Approximation
    # --------------------------
    
    # sin(2*pi/q *t) = cos(2*pi/q * (t - q/4))
    #cos(2*pi (t/q - 1/4))
    
    x = scale* ( t /q-1/4) * 1/(2**cos_r)
 
    
    # Use Chebyshev polynomial to approximate cos(2πx) on [-0.5, 0.5] step by step
    # 先自己拟合出来一个多项式,不要用现有的数据
    cheb_x = np.linspace(-1, 1, 1000)
    cheb_y = np.cos(2*np.pi**scale*cheb_x)
    cheb_coeffs = chebfit(cheb_x, cheb_y, deg=cheb_order)
  
 
    cheb_cos = chebval(x, cheb_coeffs)
    # Rescale back with double-angle formula
    for _ in range(cos_r):
        cheb_cos = 2 * cheb_cos * cheb_cos - 1
    cheb_cos = q / ( 2  *scale * np.pi) * cheb_cos
        
    for i in range(4):
        print(f"Message {i}: m={m[i]:.6f}, Cheb cos={cheb_cos[i]:.6f}, error={abs(cheb_cos[i]-m[i]):.6f}")
    cos_e += np.mean(np.abs(cheb_cos - m))
    print(f"I = {I0}, sin_e={sin_e}, cos_e={cos_e}")
    
    # # Draw cos(2πx) within [-π, π]
    # plt.figure(figsize=(10, 6))
    # plt.plot(cheb_x, cheb_y, label='cos(x)', color='blue')
    # plt.plot(cheb_x, chebval(cheb_x, cheb_coeffs), label='Chebyshev Approximation', linestyle='--', color='orange')
    # plt.title('Chebyshev Approximation of cos(x)')
    # plt.xlabel('x')
    # plt.ylabel('cos(x)')
    # plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
    # plt.axvline(0, color='black', linewidth=0.5, linestyle='--')
    # plt.legend()
    # plt.grid()
    # plt.show()
    