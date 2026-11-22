# Exemplo 6.2 página 200 livro An Introduction to Computational Fluid Dynamics – the finite volume method, 2ª ed. HK Versteeg, W Malalasekera
# Bloco 1: o SIMPLE virou uma função, com limite de iterações, checagem de NaN e medição de tempo.

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


def simple(N, fr, err=1e-5, max_it=20000, mostrar=False):
    # N: número de nós de velocidade
    # fr: fator de relaxação
    # err: critério de parada (soma dos resíduos do momento)
    # max_it: número máximo de iterações (proteção contra divergência)
    # mostrar: se True, imprime o resíduo a cada 100 iterações

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
            aP.append(aW[i-1] + aE[i-1] + (Fe[i-1] - Fw[i-1]))
            Su.append((P_cht[i] - P_cht[i+1]) * As_M[i])
            d.append(As_M[i]/aP[i-1])
            # (Bloco 3: o arrasto do meio poroso será somado ao aP aqui, antes do d)

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

        # Equações de correção da pressão (nós internos B, C, D...)
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
        "erro_%": (m_dot - m_dot_exato) / m_dot_exato * 100,
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

# Gráfico do resíduo em escala log
plt.semilogy(r["residuos"])
plt.xlabel("Iteração")
plt.ylabel("Soma dos resíduos do momento")
plt.title(f"N = {r['N']}, fr = {r['fr']}")
plt.grid(True)
plt.show()


# Estudo de convergência de malha e dos coeficientes de realaxação
Ns = [2, 4, 8, 16, 32, 64]
fs = [0.8, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01]

resultados = []

for N in Ns:
    for f in fs:
        resultados.append(simple(N, f))

for r in (resultados):
    print(f"N:{r['N']}, fr: {r['fr']}, convergiu?: {r['convergiu']}, nº it: {r['iteracoes']}, tempo: {r['tempo']:.3f}")

plt.figure(figsize=(7, 5))

for N in Ns:
    # Separa só os resultados desta malha que convergiram
    frs = []
    its = []
    for r in resultados:
        if r["N"] == N and r["convergiu"]:
            frs.append(r["fr"])
            its.append(r["iteracoes"])
    plt.plot(frs, its, marker="o", label=f"N = {N}")

plt.xscale("log")
plt.yscale("log")
plt.xlabel("Fator de relaxação (fr)")
plt.ylabel("Iterações até convergir")
plt.title("Estudo B: efeito da relaxação (só os casos que convergiram)")
plt.grid(True, which="both", alpha=0.3)
plt.legend()
plt.show()




