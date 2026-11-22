"""
PLACA PLANA COM GERACAO DE CALOR
=================================
Equacao:  d2T/dx2 + q'''/k = 0

Dados:
  L    = 3 m        (espessura da placa)
  k    = 1 W/mK     (condutividade)
  q''' = 7 W/m3     (calor gerado dentro da placa)
  q0   = 10 W/m2    (fluxo ENTRANDO pela face esquerda)

A face direita pode ser de dois tipos:
  - fluxo prescrito  (itens a e b do exercicio)
  - temperatura prescrita (item c)

Resolvemos por 4 metodos: Jacobi, Gauss-Seidel, SOR e TDMA.
"""

# =====================================================================
# 1) DADOS DO PROBLEMA
# =====================================================================

L = 3.0        # espessura da placa, em metros
k = 1.0        # condutividade termica, em W/(m.K)
q_ger = 7.0    # geracao de calor por metro cubico, em W/m3
q_esq = 10.0   # fluxo de calor entrando pela esquerda, em W/m2


# =====================================================================
# 2) MONTAGEM DOS COEFICIENTES
# =====================================================================
# Cada volume de controle tem uma equacao no formato:
#
#     AP * T_m  =  AW * T_(m-1)  +  AE * T_(m+1)  +  b
#
# onde:
#   AW = Coeficientes oeste
#   AE = Coeficientes leste
#   AP = Coeficiente do proprio volume
#   B  = Termo fonte (geracao, fluxos, temperaturas dadas)


def montar_coeficientes(N, tipo_direita, valor_direita):
    """
    N             = quantos volumes queremos
    tipo_direita  = "fluxo" ou "temperatura"
    valor_direita = o numero correspondente (W/m2 ou graus C)
    """

    dx = L / N   # largura de cada volume

    # Cria 4 listas com N posicoes, todas comecando em zero.
    aW = []
    aE = []
    aP = []
    b = []

    for i in range(N):
        aW.append(0.0)
        aE.append(0.0)
        aP.append(0.0)
        b.append(0.0)

    # ---- Preenche os volumes um por um ----
    for i in range(N):

        # Ligacao com o vizinho da esquerda.
        # O volume 0 (o primeiro) nao tem vizinho a esquerda: fica zero.
        if i == 0:
            aW[i] = 0.0
        else:
            aW[i] = k / dx

        # Ligacao com o vizinho da direita.
        # O volume N-1 (o ultimo) nao tem vizinho a direita: fica zero.
        if i == N - 1:
            aE[i] = 0.0
        else:
            aE[i] = k / dx

        # O coeficiente do proprio volume e a soma das ligacoes.
        aP[i] = aW[i] + aE[i]

        # Todo volume recebe calor da geracao interna.
        b[i] = q_ger * dx

    # ---- Fronteira ESQUERDA: fluxo entrando ----
    # O calor que entra e um valor fixo, entao soma direto no b.
    b[0] = b[0] + q_esq

    # ---- Fronteira DIREITA: depende do tipo ----
    if tipo_direita == "fluxo":
        # Calor saindo: diminui do b. Nao mexe no aP.
        b[N - 1] = b[N - 1] - valor_direita

    elif tipo_direita == "temperatura":
        # A temperatura do contorno e conhecida e fica a meia largura
        # de volume do ultimo no. Por isso a distancia e dx/2, e a
        # ligacao vale k/(dx/2) = 2k/dx.
        ligacao = 2.0 * k / dx
        aP[N - 1] = aP[N - 1] + ligacao
        b[N - 1] = b[N - 1] + ligacao * valor_direita

    else:
        print("ERRO: tipo_direita deve ser 'fluxo' ou 'temperatura'")

    return aW, aP, aE, b, dx


# =====================================================================
# 3) METODO DE JACOBI
# =====================================================================
# Ideia: isolamos T_i na equacao do volume i:
#
#     T_i = (aW * T_(i-1) + aE * T_(i+1) + b) / aP
#
# Chutamos todas as temperaturas como zero e vamos recalculando.
# No Jacobi, TODOS os valores novos sao calculados usando SOMENTE os
# valores da rodada anterior. So no fim da rodada trocamos tudo de uma vez.

def jacobi(aW, aP, aE, b, tolerancia, max_rodadas):

    N = len(aP)

    # Chute inicial: tudo zero.
    T = []
    for i in range(N):
        T.append(0.0)

    for rodada in range(1, max_rodadas + 1):

        # Lista onde guardamos os valores NOVOS, sem mexer nos velhos.
        T_novo = []
        for i in range(N):
            T_novo.append(0.0)

        for i in range(N):

            soma = 0.0

            # Contribuicao do vizinho da esquerda (se existir).
            if i > 0:
                soma = soma + aW[i] * T[i - 1]

            # Contribuicao do vizinho da direita (se existir).
            if i < N - 1:
                soma = soma + aE[i] * T[i + 1]

            T_novo[i] = (soma + b[i]) / aP[i]

        # Quanto mudou a temperatura que mais mudou?
        maior_mudanca = 0.0
        for i in range(N):
            mudanca = abs(T_novo[i] - T[i])
            if mudanca > maior_mudanca:
                maior_mudanca = mudanca

        # Agora sim, os novos viram os atuais.
        T = T_novo

        # Se quase nada mudou, convergiu.
        if maior_mudanca < tolerancia:
            return T, rodada

        # Se os numeros explodiram, divergiu.
        if maior_mudanca > 1e12:
            return T, -rodada

    # Estourou o limite de rodadas sem convergir.
    return T, -max_rodadas


# =====================================================================
# 4) METODO DE GAUSS-SEIDEL
# =====================================================================
# Quase igual ao Jacobi. A UNICA diferenca: aqui usamos o valor novo
# assim que ele fica pronto, em vez de esperar o fim da rodada.
# Como a informacao se propaga mais rapido, costuma convergir antes.

def gauss_seidel(aW, aP, aE, b, tolerancia, max_rodadas):

    N = len(aP)

    T = []
    for i in range(N):
        T.append(0.0)

    for rodada in range(1, max_rodadas + 1):

        # Guardamos uma copia so para medir o quanto mudou.
        T_anterior = []
        for i in range(N):
            T_anterior.append(T[i])

        for i in range(N):

            soma = 0.0

            # T[i-1] JA foi atualizado nesta mesma rodada.
            if i > 0:
                soma = soma + aW[i] * T[i - 1]

            # T[i+1] ainda e o da rodada passada.
            if i < N - 1:
                soma = soma + aE[i] * T[i + 1]

            # Escreve direto na lista T (nao em uma copia).
            T[i] = (soma + b[i]) / aP[i]

        maior_mudanca = 0.0
        for i in range(N):
            mudanca = abs(T[i] - T_anterior[i])
            if mudanca > maior_mudanca:
                maior_mudanca = mudanca

        if maior_mudanca < tolerancia:
            return T, rodada

        if maior_mudanca > 1e12:
            return T, -rodada

    return T, -max_rodadas


# =====================================================================
# 5) METODO SOR (Sobre-Relaxacao Sucessiva)
# =====================================================================
# O Gauss-Seidel caminha sempre na direcao certa, mas devagar.
# O SOR aposta que, se o passo esta indo para o lado certo, vale a pena
# dar um passo MAIOR. O tamanho do exagero e o omega:
#
#     omega = 1.0  -> identico ao Gauss-Seidel
#     omega > 1.0  -> acelera (sobre-relaxacao)
#     omega < 1.0  -> freia (sub-relaxacao, para casos instaveis)
#
# Valores usuais ficam entre 1.2 e 1.9. Acima de 2.0 o metodo diverge.

def sor(aW, aP, aE, b, omega, tolerancia, max_rodadas):

    N = len(aP)

    T = []
    for i in range(N):
        T.append(0.0)

    for rodada in range(1, max_rodadas + 1):

        T_anterior = []
        for i in range(N):
            T_anterior.append(T[i])

        for i in range(N):

            soma = 0.0

            if i > 0:
                soma = soma + aW[i] * T[i - 1]

            if i < N - 1:
                soma = soma + aE[i] * T[i + 1]

            # Primeiro calculamos o que o Gauss-Seidel daria.
            valor_gs = (soma + b[i]) / aP[i]

            # Depois exageramos o passo na mesma direcao.
            T[i] = T[i] + omega * (valor_gs - T[i])

        maior_mudanca = 0.0
        for i in range(N):
            mudanca = abs(T[i] - T_anterior[i])
            if mudanca > maior_mudanca:
                maior_mudanca = mudanca

        if maior_mudanca < tolerancia:
            return T, rodada

        if maior_mudanca > 1e12:
            return T, -rodada

    return T, -max_rodadas


# =====================================================================
# 6) METODO TDMA (algoritmo de Thomas)
# =====================================================================
# Este nao e iterativo: resolve o sistema de uma vez so, exato.
#
# Ideia: supomos que cada temperatura depende so da seguinte,
#
#     T_i = P_i * T_(i+1) + Q_i
#
# Na VARREDURA PARA FRENTE calculamos todos os P e Q.
# Na VARREDURA PARA TRAS descobrimos as temperaturas de fato,
# comecando pela ultima (que nao depende de ninguem, pois aE = 0).

def tdma(aW, aP, aE, b):

    N = len(aP)

    P = []
    Q = []
    T = []
    for i in range(N):
        P.append(0.0)
        Q.append(0.0)
        T.append(0.0)

    # ---- VARREDURA PARA FRENTE ----
    for i in range(N):

        if i == 0:
            denominador = aP[i]
        else:
            denominador = aP[i] - aW[i] * P[i - 1]

        # Se o denominador for zero, nao da para dividir:
        # a matriz e singular e o metodo nao funciona.
        if abs(denominador) < 1e-14:
            print("   TDMA FALHOU: denominador zerou no volume", i + 1)
            if i == 0:
                numerador = b[i]
            else:
                numerador = b[i] + aW[i] * Q[i - 1]
            print("   numerador nesse ponto =", round(numerador, 6))
            return None

        P[i] = aE[i] / denominador

        if i == 0:
            Q[i] = b[i] / denominador
        else:
            Q[i] = (b[i] + aW[i] * Q[i - 1]) / denominador

    # ---- VARREDURA PARA TRAS ----
    # O ultimo volume tem aE = 0, logo P = 0, logo T = Q.
    T[N - 1] = Q[N - 1]

    # Agora descemos: N-2, N-3, ..., 1, 0.
    for i in range(N - 2, -1, -1):
        T[i] = P[i] * T[i + 1] + Q[i]

    return T


# =====================================================================
# 7) SOLUCAO ANALITICA (so vale quando a direita tem T prescrita)
# =====================================================================
# Integrando a equacao duas vezes:
#     T(x) = -(q'''/2k) x^2 + C1 x + C2
# C1 vem do fluxo na esquerda, C2 vem da temperatura na direita.

def solucao_analitica(x, T_direita):

    C1 = -q_esq / k
    C2 = T_direita + (q_ger / (2.0 * k)) * L * L - C1 * L

    return -(q_ger / (2.0 * k)) * x * x + C1 * x + C2


# =====================================================================
# 8) RODANDO UM CASO
# =====================================================================

def rodar_caso(N, tipo_direita, valor_direita, omega, tolerancia, titulo):

    print("")
    print("=" * 62)
    print(titulo, "   (N =", N, "volumes)")
    print("=" * 62)

    aW, aP, aE, b, dx = montar_coeficientes(N, tipo_direita, valor_direita)

    max_rodadas = 100000

    # --- Jacobi ---
    T_jacobi, n_jacobi = jacobi(aW, aP, aE, b, tolerancia, max_rodadas)
    if n_jacobi > 0:
        print("   Jacobi       : convergiu em", n_jacobi, "rodadas")
    else:
        print("   Jacobi       : NAO convergiu em", -n_jacobi, "rodadas")

    # --- Gauss-Seidel ---
    T_gs, n_gs = gauss_seidel(aW, aP, aE, b, tolerancia, max_rodadas)
    if n_gs > 0:
        print("   Gauss-Seidel : convergiu em", n_gs, "rodadas")
    else:
        print("   Gauss-Seidel : NAO convergiu em", -n_gs, "rodadas")

    # --- SOR ---
    T_sor, n_sor = sor(aW, aP, aE, b, omega, tolerancia, max_rodadas)
    if n_sor > 0:
        print("   SOR (w =", omega, "): convergiu em", n_sor, "rodadas")
    else:
        print("   SOR (w =", omega, "): NAO convergiu em", -n_sor, "rodadas")

    # --- TDMA ---
    T_tdma = tdma(aW, aP, aE, b)
    if T_tdma is not None:
        print("   TDMA         : resolveu direto (sem iteracao)")

    # --- Tabela final, so faz sentido se o TDMA funcionou ---
    if T_tdma is None:
        return

    if tipo_direita != "temperatura":
        return

    print("")
    print("   no    x(m)     Jacobi       G-S          SOR          TDMA        teorico")

    for i in range(N):

        x = (i + 0.5) * dx
        T_teorico = solucao_analitica(x, valor_direita)

        print("   %2d  %6.3f  %11.5f  %11.5f  %11.5f  %11.5f  %11.5f"
              % (i + 1, x, T_jacobi[i], T_gs[i], T_sor[i], T_tdma[i], T_teorico))


# =====================================================================
# 9) PROGRAMA PRINCIPAL
# =====================================================================

N = 5                # numero de volumes
omega = 1.5          # fator do SOR
tolerancia = 1e-8    # criterio de parada dos metodos iterativos

rodar_caso(N, "fluxo", 21.0, omega, tolerancia,
           "ITEM (a): fluxo de saida = 21 W/m2")

rodar_caso(N, "fluxo", 31.0, omega, tolerancia,
           "ITEM (b): fluxo de saida = 31 W/m2")

rodar_caso(N, "temperatura", 10.0, omega, tolerancia,
           "ITEM (c): temperatura na direita = 10 C")