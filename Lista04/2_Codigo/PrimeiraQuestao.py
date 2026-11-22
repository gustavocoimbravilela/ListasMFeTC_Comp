import matplotlib.pyplot as plt # Gráficos de resultados
import numpy as np

# Primeira Questão

# Seja o problema advectivo dominante

# $\dfrac{\partial}{\partial x} (\rho\,u\,\varphi) = \dfrac{\partial}{\partial x}\left(\Gamma^{\varphi}\,\dfrac{\partial \varphi}{\partial x}\right)$

# Condições de contorno
# $x=0 : \phi = 0$
# $x=1 : \phi = 1$

# Dados do problema

rho = 1 # Densidade
gamma = 1 # Coeficiente de difusividade
N = 20 # Número de volumes
L = 1 # Tamanho do objeto
dx = L / N # Tamanho de malha
phi_1 = 1 # phi(1) = 1

# Vetor de resultados
vels = []

# Solução analítica

def analitica(x, u):
    Pe_L = rho * u * L / gamma
    if Pe_L == 0:
        return x
    return (np.exp(Pe_L*(x/L - 1)) - np.exp(-Pe_L)) / (1 - np.exp(-Pe_L))

# Letra a)

# Velocidades consideradas
us = [0, 10, 50, 100] # m/s
for u in us:
    F = rho * u
    D = gamma / dx

    P = [] # Vetor de Ps do TDMA
    Q = [] # Vetor de Qs do TDMA
    for i in range (N):
        if i == 0: # Fronteira oeste
            Ap = 3*D + F / 2
            Ae = D - F / 2
            Aw = 0
            B = 0
            P.append(Ae/Ap)
            Q.append(B/Ap)
            continue
        if i == N-1: # Fronteira leste
            Ap = 3*D - F / 2
            Ae = 0
            Aw = D + F / 2
            B = (2 * D - F) * phi_1
        else: # Nós centrais
            Ap = 2*D
            Ae = D - F / 2
            Aw = D + F / 2
            B = 0
        P.append(Ae / (Ap-Aw*P[i-1]))
        Q.append((B + Aw*Q[i-1])/(Ap - Aw*P[i-1]))
    phi = [] # Vetor de propriedades
    phi.append(Q[-1])
    for i in range(N-2, -1, -1):
        phi.append(P[i]*phi[N-2-i] + Q[i])

    phi.reverse() # Invertendo a ordem das propriedades phi para que comecem em x = dx/2 e terminem em x = N - dx/2

    vels.append(phi)

pos = []
for i in range(N):
    pos.append((i + 1/2) * dx)


x_fino = np.linspace(0, L, 300)  # malha fina só para desenhar a analítica

for i, result in enumerate(vels):
    linha, = plt.plot(pos, result,'o-', markersize=3, label=f'u = {us[i]} m/s')
    plt.plot(x_fino, analitica(x_fino, us[i]), '--', color=linha.get_color())
plt.plot([], [], 'k--', label='Analítica')
plt.xlabel("Posição (m)")
plt.ylabel(r"$\varphi$")
plt.title("CDS - Problema Advectivo-Difusivo")
plt.legend()
plt.show()


# Letra b)

# Vetor de soluções
vels = []

# Velocidades consideradas
us = [0, 10, 50, 100] # m/s
for u in us:
    F = rho * u
    D = gamma / dx

    P = [] # Vetor de Ps do TDMA
    Q = [] # Vetor de Qs do TDMA
    for i in range (N):
        if i == 0: # Fronteira oeste
            Ap = 3*D + F
            Ae = D
            Aw = 0
            B = 0
            P.append(Ae/Ap)
            Q.append(B/Ap)
            continue
        if i == N-1: # Fronteira leste
            Ap = 3*D + F
            Ae = 0
            Aw = D + F
            B = 2 * D * phi_1
        else: # Nós centrais
            Ap = F + 2 * D
            Ae = D 
            Aw = D + F
            B = 0
        P.append(Ae / (Ap-Aw*P[i-1]))
        Q.append((B + Aw*Q[i-1])/(Ap - Aw*P[i-1]))
    phi = [] # Vetor de propriedades
    phi.append(Q[-1])
    for i in range(N-2, -1, -1):
        phi.append(P[i]*phi[N-2-i] + Q[i])

    phi.reverse() # Invertendo a ordem das propriedades phi para que comecem em x = dx/2 e terminem em x = N - dx/2

    vels.append(phi)

pos = []
for i in range(N):
    pos.append((i + 1/2) * dx)

x_fino = np.linspace(0, L, 300)  # malha fina só para desenhar a analítica

for i, result in enumerate(vels):
    linha, = plt.plot(pos, result,'o-', markersize=3, label=f'u = {us[i]} m/s')
    plt.plot(x_fino, analitica(x_fino, us[i]), '--', color=linha.get_color())
plt.plot([], [], 'k--', label='Analítica')
plt.xlabel("Posição (m)")
plt.ylabel(r"$\varphi$")
plt.title("UDS - Problema Advectivo-Difusivo")
plt.legend()
plt.show()


# Letra c) i)

# Vetor de soluções
vels = []

# Velocidades consideradas
us = [0, 10, 50, 100] # m/s

for u in us:
    F = rho * u
    D = gamma / dx
    Pe = rho * u * dx / gamma # Número de Péclet
    alpha = (Pe**2) / (10 + 2 * Pe**2)  
    beta = (1 + 0.005 * Pe**2) / (1 + 0.05 * Pe**2)
    Pe_b = rho * u * dx / (gamma*2)
    alpha_b = (Pe_b**2) / (10 + 2 * Pe_b**2)
    beta_b = (1 + 0.005 * Pe_b**2) / (1 + 0.05 * Pe_b**2)  
    P = [] # Vetor de Ps do TDMA
    Q = [] # Vetor de Qs do TDMA
    for i in range (N):
        if i == 0: # Fronteira oeste
            Ap = F * (1/2 + alpha) + beta * D + 2 * beta_b * D
            Ae = beta * D - F * (1/2 - alpha)
            Aw = 0
            B = 0
            P.append(Ae/Ap)
            Q.append(B/Ap)
            continue
        if i == N-1: # Fronteira leste
            Ap = F * (alpha_b + alpha) + 2 * beta_b * D + beta * D
            Ae = 0
            Aw = F * (1/2 + alpha) + beta * D
            B = (2 * beta_b * D - F * (1/2 - alpha_b)) * phi_1
        else: # Nós centrais
            Ap = 2 * (F * alpha + beta * D)
            Ae = - F * (1/2 - alpha) + beta * D
            Aw = F * (1/2 + alpha) + beta * D
            B = 0
        P.append(Ae / (Ap-Aw*P[i-1]))
        Q.append((B + Aw*Q[i-1])/(Ap - Aw*P[i-1]))
    phi = [] # Vetor de propriedades
    phi.append(Q[-1])
    for i in range(N-2, -1, -1):
        phi.append(P[i]*phi[N-2-i] + Q[i])

    phi.reverse() # Invertendo a ordem das propriedades phi para que comecem em x = dx/2 e terminem em x = N - dx/2

    vels.append(phi)

pos = []
for i in range(N):
    pos.append((i + 1/2) * dx)


x_fino = np.linspace(0, L, 300)  # malha fina só para desenhar a analítica

for i, result in enumerate(vels):
    linha, = plt.plot(pos, result,'o-', markersize=3, label=f'u = {us[i]} m/s')
    plt.plot(x_fino, analitica(x_fino, us[i]), '--', color=linha.get_color())
plt.plot([], [], 'k--', label='Analítica')
plt.xlabel("Posição (m)")
plt.ylabel(r"$\varphi$")
plt.title("WUDS - Problema Advectivo-Difusivo")
plt.legend()
plt.show()

# Letra c) ii)

# Vetor de soluções
vels = []

# Velocidade considerada
u = 60 # m/s

# Números de volumes da malha
Ns = [10, 20, 50, 100]
pos = []
for N in Ns:
    dx = L / N
    F = rho * u
    D = gamma / dx
    Pe = rho * u * dx / gamma # Número de Péclet
    alpha = (Pe**2) / (10 + 2 * Pe**2)  
    beta = (1 + 0.005 * Pe**2) / (1 + 0.05 * Pe**2)
    Pe_b = rho * u * dx / (gamma*2)
    alpha_b = (Pe_b**2) / (10 + 2 * Pe_b**2)
    beta_b = (1 + 0.005 * Pe_b**2) / (1 + 0.05 * Pe_b**2)  
    P = [] # Vetor de Ps do TDMA
    Q = [] # Vetor de Qs do TDMA
    for i in range (N):
        if i == 0: # Fronteira oeste
            Ap = F * (1/2 + alpha) + beta * D + 2 * beta_b * D
            Ae = beta * D - F * (1/2 - alpha)
            Aw = 0
            B = 0
            P.append(Ae/Ap)
            Q.append(B/Ap)
            continue
        if i == N-1: # Fronteira leste
            Ap = F * (alpha_b + alpha) + 2 * beta_b * D + beta * D
            Ae = 0
            Aw = F * (1/2 + alpha) + beta * D
            B = (2 * beta_b * D - F * (1/2 - alpha_b)) * phi_1
        else: # Nós centrais
            Ap = 2 * (F * alpha + beta * D)
            Ae = - F * (1/2 - alpha) + beta * D
            Aw = F * (1/2 + alpha) + beta * D
            B = 0
        P.append(Ae / (Ap-Aw*P[i-1]))
        Q.append((B + Aw*Q[i-1])/(Ap - Aw*P[i-1]))
    phi = [] # Vetor de propriedades
    phi.append(Q[-1])
    for i in range(N-2, -1, -1):
        phi.append(P[i]*phi[N-2-i] + Q[i])

    phi.reverse() # Invertendo a ordem das propriedades phi para que comecem em x = dx/2 e terminem em x = N - dx/2

    temp = []
    for i in range(N):
        temp.append((i + 1/2) * dx)
    pos.append(temp)

    vels.append(phi)

x_fino = np.linspace(0, L, 300)

for i, result in enumerate(vels):
    plt.plot(pos[i], result, 'o-', markersize=3, markevery=3, label=f'N = {Ns[i]} volumes')
plt.plot(x_fino, analitica(x_fino, u), 'k--', label='Analítica')
plt.xlabel("Posição (m)")
plt.xlim(0.7,1)
plt.ylabel(r"$\varphi$")
plt.title("WUDS - Problema Advectivo-Difusivo (u = 60 m/s)")
plt.legend()
plt.show()