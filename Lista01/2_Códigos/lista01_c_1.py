# 1) c) Análise com $\varepsilon = 0,01\,\circ\mathrm{C}$

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import time
plotvec = [[],[]] # Vetor com as temperaturas $T_7$ e $T_8$, que serão usadas na análise de convergência de malha.
vn_el = [] # Vetor com os números de elementos das malhas
tit = [] # Vetor com os tempos de iteração
n_int = [0]
# Valores para dividir o comprimento incial da malha por múltiplos de 5 até 50
for i in range(5):
    n_int.append(5*(i+1) - 1)

for n in n_int:
    # Dados
    h = 10 # W/(m² * K)
    k = 1 # W/(m * K)
    dx = 0.25/(n+1) # m = dy
    Tinf = 300 # K
    Tsup = 500 # K
    Tchute = 480 # K
    episilon = 0.01 # °C
    
    # Número de elementos da malha
    n_el = int((1/dx)**2)
    print(f"Número de elementos: {n_el} - malha de {dx:.5f} mm")
    # Número de linhas e colunas da matriz N x M (considerando a simetria do problema)
    n_l = int((n_el)**(0.5) + 1)
    n_col = int((n_l+1)/2)

    # Vetor de temperaturas
    temps = []
    for i in range((n_l-1)*(n_col-1)):
        temps.append([Tchute])

    # Matriz inicial
    m = np.zeros((n_l,n_col))

    # Preenche os valores das bordas
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            if i==0 or j == 0:
                m[i,j] = Tsup

    # Preenche os valores iniciais das temperaturas
    c = 0
    for i in range(m.shape[0]-1):
        for j in range(m.shape[1]-1):
                m[i+1,j+1] = temps[c][-1]
                c += 1
    cut = True

    def chk_erro(vec, e_cut):
        chk = []
        for i in range (len(vec)):
            chk.append(abs(vec[i][-1] - vec[i][-2]) < e_cut)
        if all(chk):
            return False
        else:
            return True
    inicio = time.perf_counter() # Exercício 1) b)
    it = 0
    while cut:
        c = 0
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                    # Cálculo das temperaturas internas simétricas
                    if i != 0 and i != m.shape[0]-1 and j != 0 and j != m.shape[1] - 1: 
                        m[i,j] = (m[i,j-1] + m[i-1,j] + m[i,j+1] + m[i+1,j])/(4)
                        c += 1
                        temps[c-1].append(float(m[i,j]))
                    # Cálculo das temperaturas sobre a adiábata submetidas à condução
                    if i != 0 and i != m.shape[0]-1 and j == m.shape[1] - 1: 
                        m[i,j] = (2 * m[i,j-1] + m[i-1,j] + m[i+1,j])/(4)
                        c += 1
                        temps[c-1].append(float(m[i,j]))
                    # Cálculo das temperaturas simétricas submetidas à condução + convecção
                    if j != 0 and j != m.shape[1] - 1 and i == m.shape[0]-1:
                        m[i,j] = (m[i,j-1] + m[i,j+1] + 2 * m[i-1,j] + 2 * h * dx / k * Tinf)/(4 + 2 * h * dx / k)
                        c += 1
                        temps[c-1].append(float(m[i,j]))
                    # Cálculo da temperatura sobre a adiábata submetida à condução + convecção
                    if j == m.shape[1] - 1 and i == m.shape[0] - 1:
                        m[i,j] = (m[i-1,j] + m[i,j-1] + h * dx / k * Tinf)/(2 + h * dx / k)
                        c += 1
                        temps[c-1].append(float(m[i,j]))
        it += 1
        cut = chk_erro(temps, episilon)
    fim = time.perf_counter()
    texec = round(fim - inicio,6)
    tit.append(texec) # Guarda os tempos de execução
    # Espelhando a matriz simétrica
    # Matriz de Zeros com as dimensões completas
    mf = np.zeros((n_l,n_l))
    # Preenche as colunas da matriz completa com os valores calculados
    mf[:,:n_col] = m
    # Para preencher as colunas faltantes (simetria)
    mf[:,(n_col):] = m[:,:-1][:,-1::-1]

    # Matriz com as temperaturas obtidas
    print("Matriz com as temperaturas obtidas")
    print("\n")
    print(mf)
    print("\n")

    print("\n")


    for i in range(8):
        if i % 2 == 0:
            t = m[int((i+2)/2*(n+1)),(n+1)]
            print(f"T{(i+1)}: {t:.2f}")
            # Guardando a temperatura T7 para estudo de convergência de malha.
            if i == 6:
                plotvec[0].append(float(t))
        if i % 2 != 0:
            t = m[int((i+1)/2*(n+1)),(n+1)*2]
            print(f"T{(i+1)}: {t:.2f}")
            #Guardando a temperatura T8 para estudo de convergência de malha.
            if i == 7:
                plotvec[1].append(float(t))
            
    print("\n")

    print(f"Tempo total de execução do método: {texec} segundos")

    print("\n")

    print(f"Número de iterações necessárias: {it}\n\n")

    # Guardando o número de elementos desse estudo
    vn_el.append(n_el)

# Gráficos

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "axes.labelsize": 14,
    "axes.titlesize": 16,
    "legend.fontsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
})

# Refino de malha - Temperaturas $T_7$ e $T_8$
plt.figure(figsize=(10,6))
plt.xlabel("Número de elementos da malha")
plt.ylabel("Temperatura " + r"($^\circ$C)")
lb = ["T7", "T8"]
for i in range(len(plotvec)):
    plt.plot(vn_el, plotvec[i], label = lb[i], marker = "o")
plt.xticks(vn_el, rotation=45)
plt.grid(True)
plt.legend()
plt.show()

# Refino de malha - Diferenças das temperaturas na nova malha e na imediatamente anterior
vn_el_t = vn_el[1:]
k = [[],[]]
for i in range(len(plotvec)):
    for j in range(1,len(plotvec[i])):
        k[i].append(abs(plotvec[i][j]-plotvec[i][j-1]))
plt.figure(figsize=(10,6))
plt.xlabel("Número de elementos da malha")
plt.ylabel("Diferença entre as temperaturas"+ r"($^\circ$C)")
for i in range(len(k)):
    plt.plot(vn_el_t, k[i], label = lb[i], marker = "o")
plt.xticks(vn_el_t)
plt.grid(True)
plt.legend()
plt.show()

# Refino de malha - Tempo das iterações
plt.figure(figsize=(10,6))
plt.xlabel("Número de elementos da malha")
plt.ylabel("Tempo de execução (s)")
plt.plot(vn_el, tit, marker = "o")
plt.xticks(vn_el, rotation=45)
plt.grid(True)
plt.show()