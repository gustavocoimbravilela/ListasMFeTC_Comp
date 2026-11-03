# Exemplo 6.2 página 200 livro An Introduction to Computational Fluid Dynamics – the finite volume method, 2ª ed. HK Versteeg, W Malalasekera

import numpy as np
import matplotlib.pyplot as plt
# Dados
rho = 1 # Kg/m³
L = 2 # m
N = 500
dx = L / N # m
A_A = 0.5 # m²
A_E = 0.1 # m²
# A área é uma função linear de x, sendo A(x) = -0,2 x + 0,5
P_0 = 10 # Pa
P_E = 0 # Pa
# A pressão de chute é assumida como uma função linear entre 10 e 0 no comprimento de 2 m.
# P(x) = -5x + 10
fr = 0.001 # Fator de relaxação
# Malha de pressão
As_P = [] # Áreas da malha de pressão
P_cht = [] # Pressões de chute inicial
for i in range (N + 1):
    As_P.append(-0.2 * (i*dx) + 0.5)
    P_cht.append(-5 * (i*dx) + 10)
# Malha de momento
As_M = [] # Áreas da malha de momento
m_dot = 1 # kg/s Vazão mássica de chute inicial
us_cht = [] # Velocidades de chute inicial
for i in range (N):
    As_M.append(-0.2 * (dx/2 + i*dx) + 0.5)
    us_cht.append(m_dot/rho/As_M[i])
err = 1e-5
res = 1
t = []

#TDMA
def tdma(aP, aW, aE, Su):
    n = len(aP)
    P = [0.0] * n
    Q = [0.0] * n

    # Varredura para frente
    P[0] = aE[0] / aP [0]
    Q[0] = Su[0] / aP[0]
    for i in range(1, n):
        den = aP[i] - aW[i] * P[i-1]
        P[i] = aE[i] / den
        Q[i] = (Su[i] + aW[i] * Q[i-1]) / den

    # Substituição pra trás
    u = [0.0] * n
    u[-1] = Q[-1]
    for i in range(n-2, -1, -1):
        u[i] = P[i] * u[i+1] + Q[i]

    return u

cont = 0

while res > err:
    if i != 0 and i % 100 == 0:
        print (f"Resíduos {res}\nIteração: {i}")
    # Nós internos
    Fw = [] # aW
    Fe = []
    aW = []
    aE = []
    aP = []
    Su = []
    d = []
    for i in range(1,N-1):
        Fw.append(rho * (us_cht[i] + us_cht[i-1])/2 * As_P[i])
        Fe.append(rho * (us_cht[i] + us_cht[i+1])/2 * As_P[i+1])
        aW.append(Fw[i-1])
        aE.append(0)
        aP.append(aW[i-1] + aE[i-1] + (Fe[i-1] - Fw[i-1]))
        Su.append((P_cht[i] - P_cht[i+1]) * As_M[i])
        d.append(As_M[i]/aP[i-1])

    # Condições de contorno
    # Pressão de estagnação (Ponto A)
    uA = us_cht[0] * As_M[0] / As_P[0]
    F_A = rho * uA * As_P[0]
    F_B = rho * (us_cht[0] + us_cht[1]) / 2 * As_P[1]
    aW = [0] + aW
    aE = [0] + aE
    aP = [F_B + F_A / 2 * (As_M[0] / As_P[0])**2] + aP
    Su = [(P_0 - P_cht[1]) * As_M[0] + F_A * (As_M[0] / As_P[0]) * us_cht[0]] + Su
    d = [As_M[0] / aP[0]] + d

    # Pressão ambiente P_E = 0
    F_D = rho * (us_cht[-1] + us_cht[-2]) / 2 * As_P[-2]
    F_E = rho * us_cht[-1] * As_M[-1]
    aW = aW + [F_D]
    aE = aE + [0]
    aP = aP + [aW[-1] + aE[-1] + F_E - F_D]
    Su = Su + [(P_cht[-2] - P_cht[-1]) * As_M[-1]]
    d = d + [As_M[-1] / aP[-1]]

    res = []

    for i in range(N):
        if i == 0:
            res.append(abs(aP[i] * us_cht[i] - Su[i]))
        else:
            res.append(abs(aP[i] * us_cht[i] - aW[i] * us_cht[i-1] - Su[i]))

    res = sum(res)
    t.append(res)

    u = tdma(aP, aW, aE, Su)

    # Equações de correção da pressão
    # Nós internos
    aW_P = []
    aE_P = []
    Fw_P = []
    Fe_P = []
    aP_P = []
    b_l = []

    for i in range(1, N):
        aW_P.append(rho * d[i-1]* As_M[i-1])
        aE_P.append(rho * d[i] * As_M[i])
        Fw_P.append(rho * u[i-1] * As_M[i-1])
        Fe_P.append(rho * u[i] * As_M[i])
        aP_P.append(aW_P[i-1] + aE_P[i-1])
        b_l.append(Fw_P[i-1] - Fe_P[i-1])

    P_l = tdma(aP_P, aW_P, aE_P, b_l)

    # Considerando p_A = p_E = 0
    P_l = [0] + P_l + [0]

    # Correção de P
    P = [a + b for a, b in zip(P_cht, P_l)]

    for i in range(N):
        u[i] = (u[i] + d[i] * (P_l[i] - P_l[i+1]))

    # Aplicando o fator de relaxação.
    u = [(1-fr) * a + fr * b for a,b in zip(us_cht, u)]
    P = [(1-fr) * a + fr * b for a,b in zip(P_cht, P)]

    # Correção da pressão em A
    P[0] = P_0 - rho / 2 * u[0]**2 * (As_M[0]/As_P[0])**2

    us_cht = u
    P_cht = P
    cont += 1