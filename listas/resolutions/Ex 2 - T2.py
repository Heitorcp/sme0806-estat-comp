import numpy as np
import random
from scipy.optimize import curve_fit
from scipy.stats import bootstrap


x_list = np.array([84.30,97.30,10.73,80.65,34.49,77.40,52.94,22.92,76.30,99.65,92.08,1.49,16.88,72.23,28.23,55.22,12.15,49.12,64.32,40.68,68.76,44.34,6.89,18.78,68.15], dtype=np.float64)
y_list = np.array([0.28,0.74,4.52,0.60,1.17,0.31,0.70,3.53,0.40,0.08,0.48,1.73,3.77,0.06,3.03,0.67,4.01,0.63,0.21,1.72,0.31,1.01,3.60,3.95,0.20], dtype=np.float64)

def base_func(x, beta_1, beta_2):
  return beta_1 * x * np.exp((-1)*beta_2*x)

iteracoes = 1000

# Inicializando os arrays para guardar as estimativas bootstrap
beta1_bootstrap = np.zeros(iteracoes)
beta2_bootstrap = np.zeros(iteracoes)

#Bootstrap
n = len(x_list) 
for i in range(iteracoes):
    # Gerando a amostra bootstrap
    indices = np.random.choice(n, size=n, replace=True)
    X_bootstrap = x_list[indices]
    y_bootstrap = y_list[indices]
    
    # Ajustando ao modelo
    params, _ = curve_fit(base_func, X_bootstrap, y_bootstrap)
    beta1_bootstrap[i] = params[0]
    beta2_bootstrap[i] = params[1]


# Erro Padrão Bootstrap
se_beta1 = np.std(beta1_bootstrap)
se_beta2 = np.std(beta2_bootstrap)

print("Erro Padrão Bootstrap:")
print("β1:", se_beta1)
print("β2:", se_beta2)



# ### Exercicio 2.c

# #n<25

# beta1 = 1.222225486364666
# beta2 = 0.1003063875804432

# simulacoes = 1000
# n = 10
# erro_padrao = 0.1  # Desvio padrão do erro aleatório

# cobertura_beta1 = 0
# cobertura_beta2 = 0

# vies_beta1 = 0
# vies_beta2 = 0


# # Simulação de Monte Carlo
# for i in range(simulacoes):
#     # Gerar amostra com erro aleatório
#     x_simulado = np.random.choice(x_list, size=n, replace=False)
#     y_simulado = base_func(x_simulado, beta1, beta2) + np.random.normal(0, erro_padrao, size=n)

#     # Ajustar o modelo
#     params, cov_matrix = curve_fit(base_func, x_simulado, y_simulado)
#     estimativa_beta1 = params[0]
#     estimativa_beta2 = params[1]

#     # Acumular o viés
#     vies_beta1 += estimativa_beta1 - beta1
#     vies_beta2 += estimativa_beta2 - beta2

#     #intervalo de confiança
#     std_errors = np.sqrt(np.diag(cov_matrix))
#     limite_inferior_beta1 = params[0] - 1.96 * std_errors[0]
#     limite_superior_beta1 = params[0] + 1.96 * std_errors[0]
#     limite_inferior_beta2 = params[1] - 1.96 * std_errors[1]
#     limite_superior_beta2 = params[1] + 1.96 * std_errors[1]

#     # Verificando se o parametro verdadeiro está no intervalo de confiança
#     if limite_inferior_beta1 <= beta1 <= limite_superior_beta1:cobertura_beta1 += 1
#     if limite_inferior_beta2 <= beta2 <= limite_superior_beta2:cobertura_beta2 += 1

# # Viés médio
# vies_beta1 /= simulacoes
# vies_beta2 /= simulacoes

# print("Viés médio do estimador de Beta1:", vies_beta1)
# print("Viés médio do estimador de Beta2:", vies_beta2)

# # Probabilidade de cobertura
# coverage_prob_beta1 = cobertura_beta1 / simulacoes
# coverage_prob_beta2 = cobertura_beta2 / simulacoes
# print("Coverage probability of Beta1:", coverage_prob_beta1)
# print("Coverage probability of Beta2:", coverage_prob_beta2)


# #n=25

# n = 25

# cobertura_beta1 = 0
# cobertura_beta2 = 0

# vies_beta1 = 0
# vies_beta2 = 0


# # Simulação de Monte Carlo
# for i in range(simulacoes):
#     # Gerar amostra com erro aleatório
#     x_simulado = np.random.choice(x_list, size=n, replace=False)
#     y_simulado = base_func(x_simulado, beta1, beta2) + np.random.normal(0, erro_padrao, size=n)

#     # Ajustar o modelo
#     params, cov_matrix = curve_fit(base_func, x_simulado, y_simulado)
#     estimativa_beta1 = params[0]
#     estimativa_beta2 = params[1]

#     # Acumular o viés
#     vies_beta1 += estimativa_beta1 - beta1
#     vies_beta2 += estimativa_beta2 - beta2

#     #intervalo de confiança
#     std_errors = np.sqrt(np.diag(cov_matrix))
#     limite_inferior_beta1 = params[0] - 1.96 * std_errors[0]
#     limite_superior_beta1 = params[0] + 1.96 * std_errors[0]
#     limite_inferior_beta2 = params[1] - 1.96 * std_errors[1]
#     limite_superior_beta2 = params[1] + 1.96 * std_errors[1]

#     # Verificar se o parametro verdadeiro está no intervalo de confiança
#     if limite_inferior_beta1 <= beta1 <= limite_superior_beta1:cobertura_beta1 += 1
#     if limite_inferior_beta2 <= beta2 <= limite_superior_beta2:cobertura_beta2 += 1

# # Calcular o viés médio
# vies_beta1 /= simulacoes
# vies_beta2 /= simulacoes

# print("Viés médio do estimador de Beta1:", vies_beta1)
# print("Viés médio do estimador de Beta2:", vies_beta2)

# # Probabilidade de cobertura
# coverage_prob_beta1 = cobertura_beta1 / simulacoes
# coverage_prob_beta2 = cobertura_beta2 / simulacoes
# print("Coverage probability of Beta1:", coverage_prob_beta1)
# print("Coverage probability of Beta2:", coverage_prob_beta2)

# #n>25

# n = 50

# cobertura_beta1 = 0
# cobertura_beta2 = 0

# vies_beta1 = 0
# vies_beta2 = 0


# # Simulação de Monte Carlo
# for i in range(simulacoes):
#     # Gerar amostra com erro aleatório
#     x_simulado = np.random.choice(x_list, size=n, replace=True)
#     y_simulado = base_func(x_simulado, beta1, beta2) + np.random.normal(0, erro_padrao, size=n)

#     # Ajustar o modelo
#     params, cov_matrix = curve_fit(base_func, x_simulado, y_simulado)
#     estimativa_beta1 = params[0]
#     estimativa_beta2 = params[1]

#     # Acumular o viés
#     vies_beta1 += estimativa_beta1 - beta1
#     vies_beta2 += estimativa_beta2 - beta2

#     #intervalo de confiança
#     std_errors = np.sqrt(np.diag(cov_matrix))
#     limite_inferior_beta1 = params[0] - 1.96 * std_errors[0]
#     limite_superior_beta1 = params[0] + 1.96 * std_errors[0]
#     limite_inferior_beta2 = params[1] - 1.96 * std_errors[1]
#     limite_superior_beta2 = params[1] + 1.96 * std_errors[1]

#     # Verificar se o parametro verdadeiro está no intervalo de confiança
#     if limite_inferior_beta1 <= beta1 <= limite_superior_beta1:cobertura_beta1 += 1
#     if limite_inferior_beta2 <= beta2 <= limite_superior_beta2:cobertura_beta2 += 1

# # Calcular o viés médio
# vies_beta1 /= simulacoes
# vies_beta2 /= simulacoes

# print("Viés médio do estimador de Beta1:", vies_beta1)
# print("Viés médio do estimador de Beta2:", vies_beta2)

# # Probabilidade de cobertura
# coverage_prob_beta1 = cobertura_beta1 / simulacoes
# coverage_prob_beta2 = cobertura_beta2 / simulacoes
# print("Coverage probability of Beta1:", coverage_prob_beta1)
# print("Coverage probability of Beta2:", coverage_prob_beta2)