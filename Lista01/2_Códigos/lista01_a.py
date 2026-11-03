#Lista de exercícios 1 - Mecânica dos Fluidos e Transferência de Calor computacional

# 1) a)

import numpy as np
import matplotlib.pyplot as plt
import time

for n in range(1):
    # Dados
    h = 10 # W/(m² * K)
    k = 1 # W/(m * K)
    dx = 0.25/(n+1) # m = dy
    Tinf = 300 # K
    Tsup = 500 # K
    Tchute = [480, 470, 440, 430, 400, 390, 370, 350]
    episilon = 0.1 # °C

    # Número de elementos da malha
    n_el = int((1/dx)**2)

    # Número de linhas e colunas da matriz N x M (considerando a simetria do problema)
    n_l = int((n_el)**(0.5) + 1)
    n_col = int((n_l+1)/2)

    # Vetor de temperaturas
    temps = []
    for i in range((n_l-1)*(n_col-1)):
        temps.append([Tchute[i]])

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
    inicio = time.perf_counter()
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

    for i in range(8):
        if i % 2 == 0:
            print(f"T{(i+1)}: {m[int(((i+1)+1)/2*(n+1)),(n+1)]:.2f}")
        if i % 2 != 0:
            print(f"T{(i+1)}: {m[int((i+1)/2*(n+1)),(n+1)*2]:.2f}")

    print("\n")

    print(f"Tempo total de execução do método: {fim - inicio:.6f} segundos")

    print("\n")

    print(f"Número de iterações necessárias: {it-1}\n\n")

    