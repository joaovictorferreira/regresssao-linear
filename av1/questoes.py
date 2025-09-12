import numpy as np

def calcular_r_ajustado(n, p, r_score):
    return 1 - (1 - r_score) * (n -1) / (n - p - 1)

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

#y = b0 + b1 * x1 + b2 * x2 + b3 * x3 + b4 * x4 + e

X = np.column_stack((np.ones(len(x1)), x1, x2, x3, x4))
X_T = X.T
beta = np.linalg.inv((X_T @ X)) @ (X_T @ y)

modelo = f"y = {beta[0]:.3f} + {beta[1]:.3f} * x1 + {beta[2]:.3f} * x2 + {beta[3]:.3f} * x3 + {beta[4]:.3f} * x4"
print(f"O modelo encontrado foi: {modelo}")

## Questão B ##
#Substituir valores da questão B no modelo que foi encontrado

# x1 = 30
# x2 = 5
# x3 = 5
# x4 = 0.135

arsenio_previsto = 0.488 + -0.001 * 30 + -0.023 * 5 + -0.042 * 5 + 13.240 * 0.135
print(f"O arsênio nas unhas previsto é de: {arsenio_previsto}ppm")

## Questão C ##
#Aplicar o calculo para encontrar o r_score
y_predito = X @ beta
ss_total = np.sum((y - y.mean())**2)
ss_residuos = np.sum((y-y_predito)**2)
r_score = 1 - ss_residuos / ss_total
print(f"O R2 score para esse modelo foi de: {r_score}")

## Questão D ##
r_ajustado = calcular_r_ajustado(22, 4, r_score)
print(f"O valo do r_ajustado foi de: {r_ajustado} ja o r_score foi de: {r_score}")



