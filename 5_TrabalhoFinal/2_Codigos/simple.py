# Exemplo 6.2 página 200 livro An Introduction to Computational Fluid Dynamics –
# the finite volume method, 2ª ed. HK Versteeg, W Malalasekera

import math
import time
import matplotlib.pyplot as plt

# Dados fixos do problema (não mudam entre as simulações)
rho = 1 # kg/m³
L = 2 # m
A_A = 0.5 # m²
A_E = 0.1 # m²
P_0 = 10 # Pa (pressão de estagnação na entrada)
P_E = 0 # Pa (pressão estática na saída)
m_dot_exato = rho * A_E * math.sqrt(2 * P_0 / rho) # kg/s (Bernoulli)
b = (A_A - A_E) / L # m (A(x) = A_A - b x)

# Meio poroso: leito de esferas compactadas com ar
mu = 1.8e-5 # Pa.s (ar)
D_esf = 0.01 # m (diâmetro das esferas)
eps = 0.36 # porosidade
K = eps**3 * D_esf**2 / (150 * (1 - eps)**2) # m² (permeabilidade, Ergun)
CE = 1.75 / math.sqrt(150 * eps**3) # constante de Ergun


def vazao_exata(poroso):
    # Raiz positiva da equação do 2º grau a*m² + c*m - P_0 = 0 (solução analítica)
    a = 1 / (2 * rho * A_E**2)
    c = 0
    if poroso:
        a = a + CE / (rho * b * math.sqrt(K)) * (1/A_E - 1/A_A)
        c = mu / (rho * b * K) * math.log(A_A / A_E)
    return (-c + math.sqrt(c**2 + 4 * a * P_0)) / (2 * a)


#TDMA
def tdma(aP, aW, aE, Su):
    n = len(aP)
    P = [0.0] * n
    Q = [0.0] * n

    # Varredura para frente
    P[0] = aE[0] / aP[0]
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


def simple(N, fr, err=1e-5, max_it=20000, mostrar=False, poroso=False):
    # N: número de nós de velocidade
    # fr: fator de relaxação
    # err: critério de parada (soma dos resíduos do momento)
    # max_it: número máximo de iterações (proteção contra divergência)
    # mostrar: se True, imprime o resíduo a cada 100 iterações
    # poroso: se True, inclui o arrasto de Darcy-Forchheimer do leito de esferas

    dx = L / N # m

    # Malha de pressão
    As_P = [] # Áreas da malha de pressão
    P_cht = [] # Pressões de chute inicial
    for i in range(N + 1):
        As_P.append(0.5 - 0.2 * (i*dx))
        P_cht.append(10 - 5 * (i*dx))

    # Malha de momento
    As_M = [] # Áreas da malha de momento
    m_dot = 1 # kg/s Vazão mássica de chute inicial
    us_cht = [] # Velocidades de chute inicial
    for i in range(N):
        As_M.append(0.5 - 0.2 * (dx/2 + i*dx))
        us_cht.append(m_dot/rho/As_M[i])

    res = 1
    t = [] # histórico dos resíduos
    cont = 0
    convergiu = False

    inicio = time.perf_counter() # marca o tempo de início

    while cont < max_it:
        # Arrasto do meio poroso em cada ponto de velocidade (entra no aP)
        # D = (mu/K + CE*rho*|u_old|/sqrt(K)) * A * dx
        D = []
        for i in range(N):
            if poroso:
                D.append((mu/K + CE * rho * abs(us_cht[i]) / math.sqrt(K)) * As_M[i] * dx)
            else:
                D.append(0)

        # Nós internos da malha de velocidades
        Fw = []
        Fe = []
        aW = []
        aE = []
        aP = []
        Su = []
        d = []
        for i in range(1, N-1):
            Fw.append(rho * (us_cht[i] + us_cht[i-1])/2 * As_P[i])
            Fe.append(rho * (us_cht[i] + us_cht[i+1])/2 * As_P[i+1])
            aW.append(Fw[i-1])
            aE.append(0)
            aP.append(aW[i-1] + aE[i-1] + (Fe[i-1] - Fw[i-1]) + D[i])
            Su.append((P_cht[i] - P_cht[i+1]) * As_M[i])
            d.append(As_M[i]/aP[i-1])

        # Condições de contorno
        # Pressão de estagnação (Ponto A)
        uA = us_cht[0] * As_M[0] / As_P[0]
        F_A = rho * uA * As_P[0]
        F_B = rho * (us_cht[0] + us_cht[1]) / 2 * As_P[1]
        aW = [0] + aW
        aE = [0] + aE
        aP = [F_B + F_A / 2 * (As_M[0] / As_P[0])**2 + D[0]] + aP
        Su = [(P_0 - P_cht[1]) * As_M[0] + F_A * (As_M[0] / As_P[0]) * us_cht[0]] + Su
        d = [As_M[0] / aP[0]] + d

        # Pressão ambiente P_E = 0
        F_D = rho * (us_cht[-1] + us_cht[-2]) / 2 * As_P[-2]
        F_E = rho * us_cht[-1] * As_M[-1]
        aW = aW + [F_D]
        aE = aE + [0]
        aP = aP + [aW[-1] + aE[-1] + F_E - F_D + D[-1]]
        Su = Su + [(P_cht[-2] - P_cht[-1]) * As_M[-1]]
        d = d + [As_M[-1] / aP[-1]]

        # Resíduo do momento (com os valores da iteração anterior)
        res = 0
        for i in range(N):
            if i == 0:
                res = res + abs(aP[i] * us_cht[i] - Su[i])
            else:
                res = res + abs(aP[i] * us_cht[i] - aW[i] * us_cht[i-1] - Su[i])
        t.append(res)

        # Checagem de divergência: se o resíduo virou NaN ou infinito, para
        if not math.isfinite(res):
            break

        # Checagem de convergência
        if res < err:
            convergiu = True
            break

        if mostrar and cont != 0 and cont % 100 == 0:
            print(f"Iteração: {cont}   Resíduo: {res}")

        u = tdma(aP, aW, aE, Su)

        # Equações de correção da pressão (nós internos)
        aW_P = []
        aE_P = []
        aP_P = []
        b_l = []
        for i in range(1, N):
            aW_P.append(rho * d[i-1] * As_M[i-1])
            aE_P.append(rho * d[i] * As_M[i])
            aP_P.append(aW_P[i-1] + aE_P[i-1])
            b_l.append(rho * u[i-1] * As_M[i-1] - rho * u[i] * As_M[i])

        P_l = tdma(aP_P, aW_P, aE_P, b_l)

        # Considerando p'_A = p'_E = 0
        P_l = [0] + P_l + [0]

        # Correção de P e de u
        P = [a + b for a, b in zip(P_cht, P_l)]
        for i in range(N):
            u[i] = u[i] + d[i] * (P_l[i] - P_l[i+1])

        # Aplicando o fator de relaxação
        u = [(1-fr) * a + fr * b for a, b in zip(us_cht, u)]
        P = [(1-fr) * a + fr * b for a, b in zip(P_cht, P)]

        # Checagem de divergência: se alguma velocidade explodiu, para
        divergiu = False
        for valor in u:
            if not math.isfinite(valor) or abs(valor) > 1e6:
                divergiu = True
        if divergiu:
            break

        # Pressão em A recalculada por Bernoulli (não é relaxada)
        P[0] = P_0 - rho / 2 * u[0]**2 * (As_M[0]/As_P[0])**2

        us_cht = u
        P_cht = P
        cont += 1

    fim = time.perf_counter() # marca o tempo de fim

    # Vazão mássica (igual em todos os pontos se a continuidade foi satisfeita)
    m_dot = rho * us_cht[-1] * As_M[-1]

    # Junta tudo num dicionário para devolver
    resultado = {
        "N": N,
        "fr": fr,
        "convergiu": convergiu,
        "iteracoes": cont,
        "tempo": fim - inicio,
        "residuos": t,
        "u": us_cht,
        "P": P_cht,
        "m_dot": m_dot,
        "erro_%": (m_dot - vazao_exata(poroso)) / vazao_exata(poroso) * 100,
        "As_M": As_M,
        "As_P": As_P,
    }
    return resultado

# Teste: problema do livro (N = 4, fr = 0.8)
r = simple(4, 0.8)
print("Convergiu:", r["convergiu"])
print("Iterações:", r["iteracoes"])
print("Tempo (s):", r["tempo"])
print("Vazão (kg/s):", r["m_dot"], "  Exata:", m_dot_exato)
print("Erro (%):", r["erro_%"])
print("u:", r["u"])
print("P:", r["P"])

# Estudos do relatório: sub-relaxação (Figura 1) e malha (Tabela 3 e Figura 2)

# Fonte das figuras do mesmo tamanho do texto do relatório
plt.rcParams.update({"font.size": 9, "legend.fontsize": 8})

# ======================================================================
# Estudo 1: efeito do fator de sub-relaxação
# ======================================================================
Ns_relax = [2, 4, 8, 16, 32, 64]
fs_relax = [0.8, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01]

resultados = []
for N in Ns_relax:
    for f in fs_relax:
        resultados.append(simple(N, f))

for r in resultados:
    print(f"N:{r['N']}, fr: {r['fr']}, convergiu?: {r['convergiu']}, "
          f"nº it: {r['iteracoes']}, tempo: {r['tempo']:.3f}")

plt.figure(figsize=(5.0, 3.3))
for N in Ns_relax:
    # Separa só os resultados desta malha que convergiram
    lista_fr = []
    lista_it = []
    for r in resultados:
        if r["N"] == N and r["convergiu"]:
            lista_fr.append(r["fr"])
            lista_it.append(r["iteracoes"])
    plt.plot(lista_fr, lista_it, marker="o", ms=4, label=f"N = {N}")

plt.xscale("log")
plt.yscale("log")
plt.xlabel(r"Fator de sub-relaxação $\alpha$")
plt.ylabel("Iterações até convergir")
plt.grid(True, which="both", alpha=0.3)
plt.legend(ncol=2)
plt.tight_layout()
plt.savefig("fig_relaxacao.png", dpi=300)
plt.show()

# ======================================================================
# Estudo 2: malha (cada N com o alpha de menor número de iterações)
# ======================================================================
Ns_malha = [4, 8, 16, 32, 64, 128, 256]
fs_malha = [0.8, 0.3, 0.15, 0.05, 0.03, 0.015, 0.007]

erros = []
tempos = []
print()
print("Estudo de malha (N, alpha, iterações, tempo [s], vazão [kg/s], erro [%]):")
for i in range(len(Ns_malha)):
    r = simple(Ns_malha[i], fs_malha[i], max_it=60000)
    erro = abs(r["erro_%"])
    erros.append(erro)
    tempos.append(r["tempo"])
    print(f"{r['N']}  {r['fr']}  {r['iteracoes']}  {r['tempo']:.4f}  "
          f"{r['m_dot']:.5f}  {erro:.2f}")

# Retas de referência (passam pelo primeiro ponto do erro e pelo último do tempo)
ref_erro = []
ref_tempo = []
for N in Ns_malha:
    ref_erro.append(erros[0] * Ns_malha[0] / N)               # inclinação -1
    ref_tempo.append(tempos[-1] * (N / Ns_malha[-1])**2)      # inclinação 2

fig, ax = plt.subplots(1, 2, figsize=(6.3, 2.6))

ax[0].loglog(Ns_malha, erros, "o-", ms=4, label="SIMPLE (UDS)")
ax[0].loglog(Ns_malha, ref_erro, "k--", lw=1, label="inclinação = -1")
ax[0].set_xlabel("N (volumes)")
ax[0].set_ylabel(r"Erro em $\dot m$ (%)")
ax[0].grid(True, which="both", alpha=0.3)
ax[0].legend()

ax[1].loglog(Ns_malha, tempos, "s-", ms=4, color="C1", label="SIMPLE")
ax[1].loglog(Ns_malha, ref_tempo, "k--", lw=1, label="inclinação = 2")
ax[1].set_xlabel("N (volumes)")
ax[1].set_ylabel("Tempo (s)")
ax[1].grid(True, which="both", alpha=0.3)
ax[1].legend()

plt.tight_layout()
plt.savefig("fig_malha.png", dpi=300)
plt.show()


# Estudo do critério de convergência (Tabela 4 do relatório)
# Malha fixa: N = 64 com alpha = 0.03.

N_crit = 64
fr_crit = 0.03

# Solução de referência: critério muito apertado, para medir o erro de iteração
ref = simple(N_crit, fr_crit, err=1e-10, max_it=100000)
m_ref = ref["m_dot"]

criterios = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]

print()
print("Critério de convergência (critério, iterações, tempo [s], vazão [kg/s], diferença [%]):")
for e in criterios:
    r = simple(N_crit, fr_crit, err=e, max_it=100000)
    dif = abs(r["m_dot"] - m_ref) / m_ref * 100   # diferença em relação à referência (%)
    print(f"{e:.0e}  {r['iteracoes']}  {r['tempo']:.4f}  {r['m_dot']:.7f}  {dif:.1e}")


# Estudo do bocal com meio poroso (Tabela 5 e Figura 3 do relatório)

plt.rcParams.update({"font.size": 9, "legend.fontsize": 8})

# ======================================================================
# Estudo de malha com meio poroso (sem sub-relaxação: alpha = 1)
# Critério 1e-10: aqui o erro de discretização é tão pequeno que o erro de
# iteração com 1e-5 já seria visível nas malhas finas (custo: ~18 iterações)
# ======================================================================
Ns_por = [4, 8, 16, 32, 64, 128, 256]
erros_por = []

print()
print("Meio poroso (N, iterações, tempo [s], vazão [g/s], erro [%]):")
for N in Ns_por:
    r = simple(N, 1.0, err=1e-10, poroso=True)
    erro = abs(r["erro_%"])
    erros_por.append(erro)
    print(f"{N}  {r['iteracoes']}  {r['tempo']:.4f}  {r['m_dot']*1000:.5f}  {erro:.4f}")

# Erros sem meio poroso, para comparar (alpha de menor número de iterações)
fs_sem = [0.8, 0.3, 0.15, 0.05, 0.03, 0.015, 0.007]
erros_sem = []
for i in range(len(Ns_por)):
    r = simple(Ns_por[i], fs_sem[i], max_it=60000)
    erros_sem.append(abs(r["erro_%"]))

# ======================================================================
# Perfis de pressão: numérico (N = 16) e analítico, com e sem meio poroso
# ======================================================================
def pressao_exata(x, poroso):
    m = vazao_exata(poroso)
    A = A_A - b * x
    p = P_0 - m**2 / (2 * rho * A**2)
    if poroso:
        p = p - mu * m / (rho * b * K) * math.log(A_A / A)
        p = p - CE * m**2 / (rho * b * math.sqrt(K)) * (1/A - 1/A_A)
    return p

x_ex = []
for i in range(201):
    x_ex.append(i * L / 200)

N_perfil = 16
dx = L / N_perfil
x_p = []
for i in range(N_perfil + 1):
    x_p.append(i * dx)          # posições dos nós de pressão

r_sem = simple(N_perfil, 0.15)
r_com = simple(N_perfil, 1.0, err=1e-10, poroso=True)

p_ex_sem = []
p_ex_com = []
for x in x_ex:
    p_ex_sem.append(pressao_exata(x, False))
    p_ex_com.append(pressao_exata(x, True))

# Retas de referência para o gráfico de erro
ref1 = []
ref2 = []
for N in Ns_por:
    ref1.append(erros_sem[0] * Ns_por[0] / N)          # inclinação -1
    ref2.append(erros_por[0] * (Ns_por[0] / N)**2)     # inclinação -2

fig, ax = plt.subplots(1, 2, figsize=(6.3, 2.7))

ax[0].plot(x_ex, p_ex_sem, "k-", lw=1, label="exata")
ax[0].plot(x_ex, p_ex_com, "k-", lw=1)
ax[0].plot(x_p, r_sem["P"], "o", ms=4, color="C0", label="sem meio poroso")
ax[0].plot(x_p, r_com["P"], "s", ms=4, color="C3", label="com meio poroso")
ax[0].set_xlabel("x (m)")
ax[0].set_ylabel("p (Pa)")
ax[0].grid(True, alpha=0.3)
ax[0].legend()

ax[1].loglog(Ns_por, erros_sem, "o-", ms=4, color="C0", label="sem meio poroso")
ax[1].loglog(Ns_por, erros_por, "s-", ms=4, color="C3", label="com meio poroso")
ax[1].loglog(Ns_por, ref1, "k--", lw=1, label="inclinação = -1")
ax[1].loglog(Ns_por, ref2, "k:", lw=1, label="inclinação = -2")
ax[1].set_xlabel("N (volumes)")
ax[1].set_ylabel(r"Erro em $\dot m$ (%)")
ax[1].grid(True, which="both", alpha=0.3)
ax[1].legend(loc="lower left", fontsize=7)

plt.tight_layout()
plt.savefig("fig_poroso.png", dpi=300)
plt.show()