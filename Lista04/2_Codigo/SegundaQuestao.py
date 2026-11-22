# Lista 04 - Questão 02
#
# Problema de condução/advecção bidimensional em uma placa retangular
# Função de interpolação: WUDS
# Solução do sistema: TDMA linha a linha
#
# As velocidades u e v podem ser positivas ou negativas.
#
# Equação (já dividida por cp):
# d(rho u T)/dx + d(rho v T)/dy = d/dx(gamma dT/dx) + d/dy(gamma dT/dy)
# com gamma = k / cp
#
# Condições de contorno:
# y = 0 (sul):   T = sen(pi x / a)
# y = b (norte): T = 0
# x = 0 (oeste): T = 0
# x = a (leste): T = 0

import numpy as np
import matplotlib.pyplot as plt

# Dados que não mudam em nenhum caso
rho = 1    # kg/m³
k = 1      # W/(m K)
cp = 1     # J/(kg K)
gamma = k / cp
a = 1      # m (largura da placa)


# ======================================================================
# Função que calcula alfa e beta do WUDS
# ======================================================================
def wuds(Pe):
    # alfa tem o sinal de Pe: assim o peso maior vai sempre para o nó
    # a montante, qualquer que seja o sentido do escoamento
    alpha = np.sign(Pe) * Pe**2 / (10 + 2 * Pe**2)
    beta = (1 + 0.005 * Pe**2) / (1 + 0.05 * Pe**2)
    return alpha, beta


# ======================================================================
# TDMA para uma linha
# Resolve A_P T_P = A_E T_E + A_W T_W + d
# ======================================================================
def tdma(ap, ae, aw, d):
    N = len(ap)
    P = np.zeros(N)
    Q = np.zeros(N)
    T = np.zeros(N)
    # Varredura para frente
    for i in range(N):
        if i == 0:
            P[i] = ae[0] / ap[0]
            Q[i] = d[0] / ap[0]
        else:
            P[i] = ae[i] / (ap[i] - aw[i] * P[i-1])
            Q[i] = (d[i] + aw[i] * Q[i-1]) / (ap[i] - aw[i] * P[i-1])
    # Varredura para trás
    T[-1] = Q[N-1]
    for i in range(N-2, -1, -1):
        T[i] = P[i] * T[i+1] + Q[i]
    return T


# ======================================================================
# Função que monta os coeficientes e resolve o problema
#
# Nx, Ny: número de volumes em x e em y
# b: altura da placa
# u, v: velocidades
# M: parâmetro do termo fonte, M = 2 h a² / (k e)  (M = 0: sem fonte)
# paredes_isoladas: True para isolar as paredes oeste e leste
# T_base_constante: True para usar T = 1 na base (em vez do seno)
# eps: critério de convergência
# ======================================================================
def resolver(Nx, Ny, b, u, v, M=0, paredes_isoladas=False,
             T_base_constante=False, eps=1e-8):

    # Malha
    dx = a / Nx
    dy = b / Ny

    # Fluxos advectivos e condutâncias difusivas (com o tamanho da face)
    Fe = rho * u * dy
    Fn = rho * v * dx
    De = gamma * dy / dx
    Dn = gamma * dx / dy

    # Péclet de malha
    Pe_x = rho * u * dx / gamma
    Pe_y = rho * v * dy / gamma

    # Coeficientes do WUDS no interior e na meia distância até o contorno
    alpha_x, beta_x = wuds(Pe_x)
    alpha_y, beta_y = wuds(Pe_y)
    alpha_xb, beta_xb = wuds(Pe_x / 2)
    alpha_yb, beta_yb = wuds(Pe_y / 2)

    # Termo fonte (troca de calor pelas faces da placa), integrado no volume
    # S = -M * gamma / a² * (T - T_inf), com T_inf = 0
    Sp = M * gamma / a**2 * dx * dy
    T_inf = 0

    # Temperaturas das fronteiras
    Tbw = 0  # oeste
    Tbe = 0  # leste
    Tbn = 0  # norte

    # Matrizes dos coeficientes
    A_P = np.zeros((Nx, Ny))
    A_W = np.zeros((Nx, Ny))
    A_E = np.zeros((Nx, Ny))
    A_S = np.zeros((Nx, Ny))
    A_N = np.zeros((Nx, Ny))
    B = np.zeros((Nx, Ny))

    for i in range(Nx):
        for j in range(Ny):
            xP = (i + 0.5) * dx

            # Temperatura da base (sul) embaixo deste volume
            if T_base_constante:
                Tbs = 1
            else:
                Tbs = np.sin(np.pi * xP / a)

            # ---------------- Direção x ----------------
            if i == 0:  # fronteira oeste
                A_W[i, j] = 0
                A_E[i, j] = Fe * (alpha_x - 1/2) + beta_x * De
                if paredes_isoladas:
                    A_P[i, j] += Fe * (alpha_x - 1/2) + beta_x * De
                else:
                    # valor na face: WUDS entre o contorno e o nó (meia distância)
                    A_P[i, j] += Fe * (alpha_x + alpha_xb) + beta_x * De + 2 * beta_xb * De
                    B[i, j] += (2 * beta_xb * De + Fe * (1/2 + alpha_xb)) * Tbw
            elif i == Nx - 1:  # fronteira leste
                A_E[i, j] = 0
                A_W[i, j] = Fe * (alpha_x + 1/2) + beta_x * De
                if paredes_isoladas:
                    A_P[i, j] += Fe * (alpha_x + 1/2) + beta_x * De
                else:
                    # valor na face: WUDS entre o nó e o contorno (meia distância)
                    A_P[i, j] += Fe * (alpha_x + alpha_xb) + 2 * beta_xb * De + beta_x * De
                    B[i, j] += (2 * beta_xb * De - Fe * (1/2 - alpha_xb)) * Tbe
            else:  # interior
                A_P[i, j] += 2 * (Fe * alpha_x + beta_x * De)
                A_E[i, j] = Fe * (alpha_x - 1/2) + beta_x * De
                A_W[i, j] = Fe * (alpha_x + 1/2) + beta_x * De

            # ---------------- Direção y ----------------
            if j == 0:  # fronteira sul
                A_S[i, j] = 0
                A_N[i, j] = Fn * (alpha_y - 1/2) + beta_y * Dn
                # valor na face: WUDS entre o contorno e o nó (meia distância)
                A_P[i, j] += Fn * (alpha_y + alpha_yb) + beta_y * Dn + 2 * beta_yb * Dn
                B[i, j] += (2 * beta_yb * Dn + Fn * (1/2 + alpha_yb)) * Tbs
            elif j == Ny - 1:  # fronteira norte
                A_N[i, j] = 0
                A_S[i, j] = Fn * (alpha_y + 1/2) + beta_y * Dn
                # valor na face: WUDS entre o nó e o contorno (meia distância)
                A_P[i, j] += Fn * (alpha_y + alpha_yb) + 2 * beta_yb * Dn + beta_y * Dn
                B[i, j] += (2 * beta_yb * Dn - Fn * (1/2 - alpha_yb)) * Tbn
            else:  # interior
                A_P[i, j] += 2 * (Fn * alpha_y + beta_y * Dn)
                A_N[i, j] = Fn * (alpha_y - 1/2) + beta_y * Dn
                A_S[i, j] = Fn * (alpha_y + 1/2) + beta_y * Dn

            # ---------------- Termo fonte ----------------
            A_P[i, j] += Sp
            B[i, j] += Sp * T_inf

    # ---------------- TDMA linha a linha ----------------
    T = np.zeros((Nx, Ny))
    erro = 1
    iteracoes = 0
    while erro > eps:
        T_old = T.copy()
        for j in range(Ny):  # cada linha horizontal, de sul para norte
            d = B[:, j].copy()
            if j > 0:          # existe vizinho ao sul
                d += A_S[:, j] * T[:, j-1]
            if j < Ny - 1:     # existe vizinho ao norte
                d += A_N[:, j] * T[:, j+1]
            T[:, j] = tdma(A_P[:, j], A_E[:, j], A_W[:, j], d)
        erro = np.max(np.abs(T - T_old))
        iteracoes += 1

    # Posições dos centros dos volumes
    x = (np.arange(Nx) + 0.5) * dx
    y = (np.arange(Ny) + 0.5) * dy

    return x, y, T, iteracoes


# ======================================================================
# Soluções analíticas
# ======================================================================

# u = v = 0 (com ou sem termo fonte)
def analitica_laplace(X, Y, b, M=0):
    lam = np.sqrt((np.pi / a)**2 + M / a**2)
    return np.sin(np.pi * X / a) * np.sinh(lam * (b - Y)) / np.sinh(lam * b)

# Paredes isoladas, T = 1 na base, u = 0, v = cte (igual à Questão 1)
def analitica_1D_v(y, b, v):
    Pe_b = rho * v * b / gamma
    if Pe_b == 0:
        return 1 - y / b
    return (1 - np.exp(Pe_b * (y / b - 1))) / (1 - np.exp(-Pe_b))


# Temperatura no centro da placa: média dos 4 nós em volta do centro
# (vale para Nx e Ny pares)
def temperatura_centro(T):
    Nx = T.shape[0]
    Ny = T.shape[1]
    i = Nx // 2
    j = Ny // 2
    return (T[i-1, j-1] + T[i, j-1] + T[i-1, j] + T[i, j]) / 4


# Desenho do campo de temperatura
def desenhar_campo(x, y, T, titulo):
    X, Y = np.meshgrid(x, y, indexing='ij')
    plt.contourf(X, Y, T, levels=np.linspace(0, 1, 21), cmap='inferno')
    barra = plt.colorbar(ticks=[0, 0.5, 1])
    barra.ax.tick_params(labelsize=7)
    plt.contour(X, Y, T, levels=[0.05, 0.1, 0.2, 0.4, 0.6, 0.8],
                colors='white', linewidths=0.6)
    plt.gca().set_aspect('equal')
    plt.title(titulo, fontsize=9)
    plt.xlabel('x (m)', fontsize=8)
    plt.ylabel('y (m)', fontsize=8)
    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7)


# ======================================================================
# Caso limite 1: u = v = 0 -- comparação com a analítica
# ======================================================================
print('Caso 1: u = v = 0')
erro_anterior = 0
for N in [10, 20, 40, 80]:
    x, y, T, it = resolver(N, N, b=1, u=0, v=0, eps=1e-10)
    X, Y = np.meshgrid(x, y, indexing='ij')
    T_analitica = analitica_laplace(X, Y, b=1)
    erro_max = np.max(np.abs(T - T_analitica))
    if erro_anterior == 0:
        print(f'N = {N}, iterações = {it}, erro = {erro_max:.3e}')
    else:
        print(f'N = {N}, iterações = {it}, erro = {erro_max:.3e}, '
              f'razão = {erro_anterior / erro_max:.2f}')
    erro_anterior = erro_max

# Figura: isotermas numéricas e analíticas (malha 40 x 40)
x, y, T, it = resolver(40, 40, b=1, u=0, v=0, eps=1e-10)
X, Y = np.meshgrid(x, y, indexing='ij')
T_analitica = analitica_laplace(X, Y, b=1)
niveis = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
plt.figure(figsize=(5.2, 4.4))
plt.contourf(X, Y, T, levels=np.linspace(0, 1, 21), cmap='inferno')
plt.colorbar(label='T')
linhas = plt.contour(X, Y, T, levels=niveis, colors='white', linewidths=1.0)
plt.contour(X, Y, T_analitica, levels=niveis, colors='cyan', linewidths=1.0,
            linestyles='--')
plt.clabel(linhas, fontsize=7, fmt='%.2f')
plt.plot([], [], 'w-', label='Numérica')
plt.plot([], [], 'c--', label='Analítica')
plt.legend(loc='upper right', fontsize=8, facecolor='0.3', labelcolor='white')
plt.gca().set_aspect('equal')
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.title('u = v = 0, malha 40×40')
plt.tight_layout()
plt.savefig('L04_Q2_laplace.png', dpi=200)
plt.show()


# ======================================================================
# Caso limite 2: paredes isoladas, T = 1 na base, u = cte, v = 0
# Solução exata: T = 1 - y/b
# ======================================================================
print('Caso 2: paredes isoladas, u = cte, v = 0')
for u in [0, 10, 50, 100]:
    x, y, T, it = resolver(20, 20, b=1, u=u, v=0, paredes_isoladas=True,
                           T_base_constante=True, eps=1e-12)
    erro_max = 0
    for i in range(20):
        for j in range(20):
            erro_ij = abs(T[i, j] - (1 - y[j] / 1))
            if erro_ij > erro_max:
                erro_max = erro_ij
    print(f'u = {u}, Pe_dx = {u * (1/20)}, iterações = {it}, erro = {erro_max:.2e}')


# ======================================================================
# Extensão: paredes isoladas, T = 1 na base, u = 0, v = cte
# Mesmo problema da Questão 1, agora na direção y
# ======================================================================
print('Caso 2 (extensão): paredes isoladas, u = 0, v = cte')
y_fino = np.linspace(0, 1, 400)
plt.figure(figsize=(10, 4.2))
grafico = 1
for lista_v in [[0, 10, 50, 100], [-10, -50, -100]]:
    plt.subplot(1, 2, grafico)
    for v in lista_v:
        x, y, T, it = resolver(20, 20, b=1, u=0, v=v, paredes_isoladas=True,
                               T_base_constante=True, eps=1e-12)
        perfil = T[10, :]  # perfil vertical em uma coluna qualquer
        erro_max = np.max(np.abs(perfil - analitica_1D_v(y, 1, v)))
        print(f'v = {v}, Pe_dy = {v * (1/20)}, iterações = {it}, erro = {erro_max:.4f}')
        linha, = plt.plot(y, perfil, 'o-', markersize=3, label=f'v = {v} m/s')
        plt.plot(y_fino, analitica_1D_v(y_fino, 1, v), '--', color=linha.get_color())
    plt.plot([], [], 'k--', label='Analítica')
    plt.xlabel('y (m)')
    plt.ylabel('T')
    if grafico == 1:
        plt.title('v ≥ 0: escoamento para cima')
    else:
        plt.title('v < 0: escoamento para baixo')
    plt.legend()
    plt.grid(alpha=0.3)
    grafico += 1
plt.tight_layout()
plt.savefig('L04_Q2_1D.png', dpi=200)
plt.show()


# ======================================================================
# Estudo de Pe_dx e Pe_dy (malha 20 x 20, b/a = 1)
# Como dx = dy = 1/20, u = 20 * Pe_dx e v = 20 * Pe_dy
# ======================================================================
print('Estudo de Pe_dx e Pe_dy')
N = 20
plt.figure(figsize=(8.4, 8.4))
posicao = 1
for Pe_dy in [5, 1, 0]:          # linhas da figura (de cima para baixo)
    for Pe_dx in [0, 1, 5]:      # colunas da figura
        u = Pe_dx * N
        v = Pe_dy * N
        x, y, T, it = resolver(N, N, b=1, u=u, v=v)
        Tc = temperatura_centro(T)
        print(f'Pe_dx = {Pe_dx}, Pe_dy = {Pe_dy}, iterações = {it}, '
              f'T_centro = {Tc:.4f}, T_min = {T.min():.2e}, T_max = {T.max():.4f}')
        plt.subplot(3, 3, posicao)
        desenhar_campo(x, y, T, f'Pe_dx = {Pe_dx}, Pe_dy = {Pe_dy}')
        posicao += 1
plt.tight_layout()
plt.savefig('L04_Q2_mapaPe.png', dpi=200)
plt.show()


# ======================================================================
# Independência de malha com advecção (u = v = 50)
# ======================================================================
print('Independência de malha, u = v = 50')
for N in [10, 20, 40, 80, 160]:
    x, y, T, it = resolver(N, N, b=1, u=50, v=50, eps=1e-9)
    print(f'N = {N}, Pe_d = {50 / N:.3f}, iterações = {it}, '
          f'T_centro = {temperatura_centro(T):.5f}')


# ======================================================================
# Estudo da relação b/a (volumes quadrados, Nx = 40)
# ======================================================================
print('Estudo de b/a')
Nx = 40
plt.figure(figsize=(8.4, 7.4))
coluna = 1
for b in [0.5, 1.0, 2.0]:
    Ny = int(Nx * b / a)

    # sem escoamento: compara com a analítica
    x, y, T, it = resolver(Nx, Ny, b=b, u=0, v=0, eps=1e-10)
    X, Y = np.meshgrid(x, y, indexing='ij')
    erro_max = np.max(np.abs(T - analitica_laplace(X, Y, b)))
    Tc_analitica = np.sinh(np.pi * (b / 2) / a) / np.sinh(np.pi * b / a)
    print(f'b/a = {b}, u = v = 0, iterações = {it}, erro = {erro_max:.2e}, '
          f'T_centro = {temperatura_centro(T):.4f}, analítica = {Tc_analitica:.4f}')
    plt.subplot(2, 3, coluna)
    desenhar_campo(x, y, T, f'b/a = {b}, u = v = 0')

    # com escoamento
    x, y, T, it = resolver(Nx, Ny, b=b, u=20, v=20, eps=1e-10)
    print(f'b/a = {b}, u = v = 20, iterações = {it}, '
          f'T_centro = {temperatura_centro(T):.4f}')
    plt.subplot(2, 3, coluna + 3)
    desenhar_campo(x, y, T, f'b/a = {b}, u = v = 20 m/s')

    coluna += 1
plt.tight_layout()
plt.savefig('L04_Q2_ba.png', dpi=200)
plt.show()


# ======================================================================
# Velocidades negativas
# ======================================================================
print('Velocidades negativas')

# Teste de simetria: a base sen(pi x / a) é simétrica em relação a x = a/2.
# Então o campo com -u tem de ser o espelho (em x) do campo com +u.
x, y, T_mais, it_mais = resolver(20, 20, b=1, u=50, v=20)
x, y, T_menos, it_menos = resolver(20, 20, b=1, u=-50, v=20)
T_espelhado = T_mais[::-1, :]   # inverte a ordem das colunas em x
desvio = np.max(np.abs(T_menos - T_espelhado))
print(f'u = +50: iterações = {it_mais}; u = -50: iterações = {it_menos}; '
      f'desvio de simetria = {desvio:.1e}')

plt.figure(figsize=(8.4, 3.2))
posicao = 1
for u, v in [[50, 20], [-50, 20], [20, -20]]:
    x, y, T, it = resolver(20, 20, b=1, u=u, v=v)
    print(f'u = {u}, v = {v}, iterações = {it}, T_centro = {temperatura_centro(T):.4f}, '
          f'T_min = {T.min():.2e}, T_max = {T.max():.4f}')
    plt.subplot(1, 3, posicao)
    desenhar_campo(x, y, T, f'u = {u}, v = {v} m/s')
    posicao += 1
plt.tight_layout()
plt.savefig('L04_Q2_negativo.png', dpi=200)
plt.show()


# ======================================================================
# Termo fonte: placa fina trocando calor pelas faces
# ======================================================================
print('Termo fonte: verificação com u = v = 0')
for M in [0, 10, 50]:
    x, y, T, it = resolver(40, 40, b=1, u=0, v=0, M=M, eps=1e-10)
    X, Y = np.meshgrid(x, y, indexing='ij')
    erro_max = np.max(np.abs(T - analitica_laplace(X, Y, 1, M)))
    print(f'M = {M}, lambda*a = {np.sqrt(np.pi**2 + M):.3f}, iterações = {it}, '
          f'erro = {erro_max:.2e}, T_centro = {temperatura_centro(T):.4f}')

print('Chapa em movimento: M = 20, v = 0')
plt.figure(figsize=(8.4, 3.2))
posicao = 1
for u in [0, 10, 50]:
    x, y, T, it = resolver(40, 40, b=1, u=u, v=0, M=20, eps=1e-10)
    print(f'u = {u}, iterações = {it}, T_centro = {temperatura_centro(T):.4f}')
    plt.subplot(1, 3, posicao)
    desenhar_campo(x, y, T, f'M = 20, u = {u} m/s, v = 0')
    posicao += 1
plt.tight_layout()
plt.savefig('L04_Q2_fonte.png', dpi=200)
plt.show()