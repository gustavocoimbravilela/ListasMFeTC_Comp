import numpy as np

# ---------------- dados do problema ----------------
Tb, Tinf = 373.0, 293.0
k, D, L, h, alpha = 10.0, 0.01, 0.05, 5.0, 1e-6
rhoc = k / alpha          # 1e7 J/m3K
P = np.pi * D #\pi D
PA = P / (np.pi) / D**2 * 4 # P/A = 400 1/m


def tdma(aW, aP, aE, b):
    N = len(aP)
    P = np.zeros(N)
    Q = np.zeros(N)
    T = np.zeros(N)
    den = aP[0]
    P[0] = aE[0] / den
    Q[0] = b[0] / den
    for i in range(1, N):
        den = aP[i] - aW[i] * P[i-1]
        P[i] = aE[i] / den
        Q[i] = (b[i] + aW[i] * Q[i-1]) / den
    T[N-1] = Q[N-1]
    for i in range(N-2, -1, -1):
        T[i] = P[i] * T[i+1] + Q[i]
    return T


def coeficientes(N, dt):
    dx = L / N
    aW = np.full(N, k/dx)
    aW[0] = 0.0
    aE = np.full(N, k/dx)
    aE[-1] = 0.0
    aP0 = rhoc * dx / dt if dt is not None else 0.0
    Sp = PA * h * dx                      # hP/A * dx
    aP = aW + aE + aP0 + Sp
    aP[0] += 2*k/dx                       # base: meia celula
    return aW, aP, aE, aP0, Sp, dx


def resolver(N=10, dt=1.0, eps=1e-6, tmax=1e6):
    aW, aP, aE, aP0, Sp, dx = coeficientes(N, dt)
    T = np.full(N, Tinf)
    t, passos = 0.0, 0
    while t < tmax:
        b = aP0*T + Sp*Tinf
        b[0] += (2*k/dx) * Tb
        Tn = tdma(aW, aP, aE, b)
        dmax = np.max(np.abs(Tn - T))
        T = Tn
        t += dt
        passos += 1
        if dmax < eps:
            break
    x = (np.arange(N) + 0.5) * dx
    return x, T, t, passos


def analitica(x):
    m = np.sqrt(4*h/(k*D))
    return Tinf + (Tb - Tinf) * np.cosh(m*(L-x)) / np.cosh(m*L)


if __name__ == "__main__":

    # Letra b) Implemente a solução utilizando o TDMA
    N, dt = 10, 1.0
    x, T, t, n = resolver(N, dt, eps=1e-6)
    print(f"N={N}  dt={dt}s  passos={n}  t={t:.0f} s\n")
    print("  m     x(m)    T_num(K)")
    for i in range(N):
        print(f"{i+1:3d}  {x[i]:7.4f}  {T[i]:9.3f}")

    # Letra c) Resolva o problema variando o número de volumes elementares

    N_it = [3, 5, 10, 20, 40, 80]
    emax = []
    for i in N_it:
        x, T, t, n = resolver(i, dt, eps=1e-6)
        Ta = analitica(x)
        emax.append(np.max(abs(T-Ta)))
    print("  N     \delta x_p(m)    e_max(K)")
    for i in range(len(N_it)):
        print(f"{N_it[i]:3d}  {L/N_it[i]:7.4f}  {emax[i]:9.3f}")

    # Letra d) Resolva o problema variando o critério de convergência
    
    eps_v = [1e-1, 1e-2, 1e-4, 1e-6, 1e-8]
    emax = []
    n_it = []
    for i in eps_v:
        N, dt = 20, 1.0
        x, T, t, n = resolver(N, dt, i)
        Ta = analitica(x)
        n_it.append(n)
        emax.append(np.max(abs(T-Ta)))
    print("\varespsilon     passos    e_max(K)")
    for i in range(len(eps_v)):
        print(f"{eps_v[i]}  {n_it[i]:3d}  {emax[i]:9.3f}")

    # Letra e) Compare a solução de regime permanente com a solução analítica

    N, dt = 10, 1.0
    x, T, t, n = resolver(N, dt, 1e-6)
    Ta = analitica(x)
    print(f"N={N}  dt={dt}s  passos={n}  t={t:.0f} s\n")
    print("  m     x(m)    T_num(K) T_teórico(K) erro(K)")
    for i in range(N):
        print(f"{i+1:3d}  {x[i]:7.4f}  {T[i]:9.3f}  {Ta[i]:9.3f}  {T[i]-Ta[i]:9.3f}")
