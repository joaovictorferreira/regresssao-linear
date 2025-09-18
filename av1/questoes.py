import numpy as np
import pandas as pd

def calcular_r_score(X, beta, y):
    y_predito = X @ beta
    ss_total = np.sum((y - y.mean())**2)
    ss_residuos = np.sum((y-y_predito)**2)
    return 1 - ss_residuos / ss_total

def calcular_r_ajustado(n, p, r_score):
    return 1 - (1 - r_score) * (n -1) / (n - p - 1)

def calcular_rmse(y, X, beta):
    y_predito = X @ beta
    return np.sqrt((1/len(y)) * np.sum((y - y_predito) ** 2))

def calcular_mae(y, X, beta):
    y_predito = X @ beta
    return np.sum(np.abs(y - y_predito)) / len(y)

## Questão A ##
dados = np.loadtxt("C:\\Users\\jvf_h\\Documents\\IA\\av1\\arsenio_dataset.csv", delimiter=",", skiprows=1)

#x1 = idade
#x2 = uso do para beber
#x3 = uso para cozinhar 
#x4 = arsênio na agua
#y = arsênio nas unhas
x1 = dados[:,0]
x2 = dados[:,2]
x3 = dados[:,3]
x4 = dados[:,4]
y = dados[:,5]

X = np.column_stack((np.ones(len(x1)), x1, x2, x3, x4))
X_T = X.T
beta = np.linalg.inv((X_T @ X)) @ (X_T @ y)

modelo = f"y = {beta[0]:.3f} + {beta[1]:.3f} * x1 + {beta[2]:.3f} * x2 + {beta[3]:.3f} * x3 + {beta[4]:.3f} * x4"
print(f"O modelo encontrado foi: {modelo}")

## Questão B ##
# x1 = 30
# x2 = 5
# x3 = 5
# x4 = 0.135

arsenio_previsto = 0.488 + -0.001 * 30 + -0.023 * 5 + -0.042 * 5 + 13.240 * 0.135
print(f"O arsênio nas unhas previsto é de: {arsenio_previsto}ppm")

## Questão C ##
r_score = calcular_r_score(X, beta, y)
print(f"O R2 score para esse modelo foi de: {r_score}")

## Questão D ##
r_ajustado = calcular_r_ajustado(len(y), 4, r_score) 
print(f"O valo do r_ajustado foi de: {r_ajustado} ja o r_score foi de: {r_score}")



## Questâo E ##
x_alt = np.column_stack((np.ones(len(x4)), x4))
x_alt_t = x_alt.T
beta_alt = np.linalg.inv((x_alt_t @ x_alt)) @ (x_alt_t @ y)
modelo_alt = f"y = {beta_alt[0]:.3f} + {beta[1]:.3f} * x4"
y_alt = 0.155 + - 0.001 * 0.135

r_score_alt = calcular_r_score(x_alt, beta_alt, y)
print(f"O R2 score para esse modelo alternativo foi de: {r_score_alt}")
# O primeiro modelo apresentou desempenho superior ao segundo, 
# com R2= 0.81 em comparação ao do segundo modelo R2=0.80. 
# Essa variação se deve provavelmente pelo fato de que 
# o primeiro modelo conta com mais informações dos preditores para poder prever 
# a concentração de arsênio na água.

## Questão f(Do inicio ate a montagem da tabela) ##

linhas = []
for dado in dados:
    x1 = dado[0]
    x2 = dado[2]
    x3 = dado[3]
    x4 = dado[4]
    valor_observado = dado[5]
    valor_ajustado = beta[0] + beta[1] * x1 + beta[2] * x2 + beta[3] * x3 + beta[4] * x4
    residuo = valor_observado - valor_ajustado
    linhas.append({
        "Valor observado": valor_observado,
        "Valor ajustado": valor_ajustado,
        "Residuo": residuo 
    })

tabela = pd.DataFrame(linhas)
print(tabela)

## Questão J ##
#Não feita

## Questão K ##
rmse_modelo_completo = calcular_rmse(y, X, beta)
rmse_modelo_alternativo = calcular_rmse(y, x_alt, beta_alt)
print(f"O valor do rmse para o modelo completo é de: {rmse_modelo_completo} e para o modelo alternativo é: {rmse_modelo_alternativo}")
mae_modelo_completo = calcular_mae(y, X, beta)
mae_modelo_alternativo = calcular_mae(y, x_alt, beta_alt)
print(f"O valor do mae para o modelo completo é de: {mae_modelo_completo} e para o modelo alternativo é: {mae_modelo_alternativo}")