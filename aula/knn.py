import numpy as np
import math
import urllib.request

# Busca os dados
def coletar_dados():
    #Substituir pelo arquivo baixado em sí
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"
    data = np.genfromtxt(urllib.request.urlopen(url), delimiter=",")
    y = data[:,0]
    x = data[:, 1:]
    print(x, y)

# Divide o que é dado de treino e o que é dado de teste
def train_test_split(X, y, test_size=0.2, random_state=42):
    if random_state is not None:
        np.random.seed(random_state)

    if len(X) != len(y):
        raise ValueError("X e y devem ter o mesmo tamanho")

    n_samples = len(X)
    indices = np.random.permutation(n_samples)
    print("Indices aleatórios:", indices)

    n_test = math.ceil(n_samples * test_size)  # arredonda pra baixo
    print("Tamanho do teste:", n_test)
    #n_test = int(n_samples * test_size)

    test_indices = indices[:n_test]
    train_indices = indices[n_test:]

    if X.ndim == 1:
        X_train, X_test = X[train_indices], X[test_indices]
    else:
        X_train, X_test = X[train_indices, :], X[test_indices, :]

    y_train, y_test = y[train_indices], y[test_indices]

    return X_train, X_test, y_train, y_test


#Aplica o alg em cima dos dados obtidos de teste e de treino da função do hold_out 
class KNN:

    def __init__(self, k=5, task='classification'):
        self.k = k
        self.task = task

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    #x1 são os dados de teste
    #x2 são os dados de treinamento
    # Talvez tenha que transformar o x1 e x2 em arra antes de dar return
    def euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1-x2)**2))
    
    # x recebido são os dados de teste
    # x_train são os dados de treinamento que ele vai utilizar para fazer a diferença na formula
    # Para cada X_train(dados de treinamento) ele vai fazer o calculo da distancia
    # Depois vai pegar os indices das menores distancias dependendo do número k de diatancias fornecido
    # Após isso ele vai ir buscar o rotulo(y) dos elementos de menor distancia que foram obtidos no k_indices
    # Depois de coletado essas labels vai ser feita a contagem para classificar ou a media para regressão
    def calculate_prediction(self, x):
        distances = [self.euclidean_distance(x, x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_neares_labels = [self.y_train[i] for i in k_indices]

        if self.task == 'classification':
            unique, counts = np.unique(k_neares_labels, return_counts=True)
            return unique[np.argmax(counts)]
        elif self.task == 'regression':
            return np.mean(k_neares_labels)
        else:
            raise ValueError("Tarefa não definida")
        


# Copiar o hold out para separar os dados de teste e de treino
# Pegar o dataset KNN
# Integrar com o código do KNN  