# Questao 4 - Conducao de calor 1D transiente
# Formulacao explicita (MVF, malha uniforme)
# Malha com 5 pontos: dois nas fronteiras (T = 1) e tres internos

import matplotlib.pyplot as plt

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "axes.labelsize": 14,
    "axes.titlesize": 16,
    "legend.fontsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
})

# ---------------- DADOS DO PROBLEMA ----------------
alfa = 1.0          # difusividade termica
dx = 1.0            # espacamento da malha
n_pontos = 5        # numero de pontos da malha
T_fronteira = 1.0   # temperatura prescrita nas duas fronteiras
T_inicial = 0.0     # condicao inicial em todo o dominio
n_passos = 6        # numero de intervalos de tempo

lista_dt = [0.25, 0.50, 0.75]   # valores de dt a testar


# ---------------- FUNCAO QUE RESOLVE ----------------
def resolver(dt):

    # numero de Fourier da malha
    r = alfa * dt / (dx * dx)

    # monta o vetor de temperaturas no tempo inicial
    T = []
    for i in range(n_pontos):
        T.append(T_inicial)

    # aplica a condicao de contorno nas duas pontas
    T[0] = T_fronteira
    T[n_pontos - 1] = T_fronteira

    # guarda a solucao de cada instante de tempo
    historico = [list(T)]

    # avanca no tempo
    for passo in range(n_passos):

        T_novo = list(T)   # copia; as fronteiras ja ficam com o valor certo

        # calcula somente os pontos internos
        for i in range(1, n_pontos - 1):
            T_novo[i] = r * T[i - 1] + (1 - 2 * r) * T[i] + r * T[i + 1]

        T = T_novo
        historico.append(list(T))

    return r, historico


# ---------------- IMPRIME OS RESULTADOS ----------------
for dt in lista_dt:

    r, historico = resolver(dt)

    print()
    print("=" * 55)
    print("dt =", dt, "   r = alfa*dt/dx^2 =", r, "   (1-2r) =", 1 - 2 * r)
    print("=" * 55)

    # cabecalho da tabela
    linha = "passo    t   "
    for i in range(n_pontos):
        linha = linha + "     x=" + str(i * dx)
    print(linha)

    # uma linha por instante de tempo
    for passo in range(len(historico)):
        t = passo * dt
        linha = "  " + str(passo) + "   " + format(t, "5.2f") + "  "
        for i in range(n_pontos):
            linha = linha + format(historico[passo][i], "10.4f")
        print(linha)


# Questao 4 - Conducao de calor 1D transiente
# Formulacao explicita (MVF, malha uniforme)
# Malha com 5 pontos: dois nas fronteiras (T = 1) e tres internos

# ---------------- DADOS DO PROBLEMA ----------------
alfa = 1.0          # difusividade termica
dx = 1.0            # espacamento da malha
n_pontos = 5        # numero de pontos da malha
T_fronteira = 1.0   # temperatura prescrita nas duas fronteiras
T_inicial = 0.0     # condicao inicial em todo o dominio
n_passos = 6        # numero de intervalos de tempo

lista_dt = [0.25, 0.50, 0.75]   # valores de dt a testar


# ---------------- FUNCAO QUE RESOLVE ----------------
def resolver(dt):

    # numero de Fourier da malha
    r = alfa * dt / (dx * dx)

    # monta o vetor de temperaturas no tempo inicial
    T = []
    for i in range(n_pontos):
        T.append(T_inicial)

    # aplica a condicao de contorno nas duas pontas
    T[0] = T_fronteira
    T[n_pontos - 1] = T_fronteira

    # guarda a solucao de cada instante de tempo
    historico = [list(T)]

    # avanca no tempo
    for passo in range(n_passos):

        T_novo = list(T)   # copia; as fronteiras ja ficam com o valor certo

        # calcula somente os pontos internos
        for i in range(1, n_pontos - 1):
            T_novo[i] = r * T[i - 1] + (1 - 2 * r) * T[i] + r * T[i + 1]

        T = T_novo
        historico.append(list(T))

    return r, historico


# ---------------- POSICOES DA MALHA ----------------
x = []
for i in range(n_pontos):
    x.append(i * dx)


# ---------------- IMPRIME E GERA OS GRAFICOS ----------------
nomes = {0.25: "q4_dt025.png", 0.50: "q4_dt050.png", 0.75: "q4_dt075.png"}

for dt in lista_dt:

    r, historico = resolver(dt)

    # ---- impressao no console ----
    print()
    print("=" * 55)
    print("dt =", dt, "   r =", r, "   (1-2r) =", 1 - 2 * r)
    print("=" * 55)
    for passo in range(len(historico)):
        linha = "passo " + str(passo) + "  "
        for i in range(n_pontos):
            linha = linha + format(historico[passo][i], "10.4f")
        print(linha)

    # ---- grafico: perfil de temperatura em cada passo ----
    plt.figure(figsize=(7, 4.5))

    for passo in range(len(historico)):
        plt.plot(x, historico[passo], marker="o",
                 label="passo " + str(passo) + "  (t = " + format(passo * dt, ".2f") + ")")

    plt.xlabel(r"Posição $x$ (m)")
    plt.ylabel(r"Temperatura $T$ ($^\circ$C)")
    plt.grid(True, linestyle=":")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(nomes[dt], dpi=200)
    plt.close()

    print("grafico salvo:", nomes[dt])


# ---------------- GRAFICO COMPARATIVO ----------------
# evolucao da temperatura do no central (x = 2) nos tres casos

plt.figure(figsize=(7, 4.5))

for dt in lista_dt:
    r, historico = resolver(dt)

    passos = []
    T_central = []
    for passo in range(len(historico)):
        passos.append(passo)
        T_central.append(historico[passo][2])

    plt.plot(passos, T_central, marker="o", label="dt = " + str(dt) + "  (r = " + str(r) + ")")

plt.xlabel(r"Número da iteração")
plt.ylabel(r"Temperatura $T$ ($^\circ$C) no ponto central $x=2$")
plt.grid(True, linestyle=":")
plt.legend()
plt.tight_layout()
plt.savefig("q4_central.png", dpi=200)
plt.close()

print("grafico salvo: q4_central.png")