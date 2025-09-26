import numpy as np

class KNN:

    def __init__(self, k=5, task='classification'):
        self.k = k
        self.task = task

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def euclidean_distance(self, x1, x2):
        #x1 são os dados de teste
        #x2 são os dados de treinamento
        # Talvez tenha que transformar o x1 e x2 em arra antes de dar return
        return np.sqrt(np.sum((x1-x2)**2))
    
    def calculate_prediction(self, x):
        # x recebido são os dados de teste
        # x_train são os dados de treinamento que ele vai utilizar para fazer a diferença na formula
        # Para cada X_train(dados de treinamento) ele vai fazer o calculo da distancia
        # Depois vai pegar os indices das menores distancias dependendo do número k de diatancias fornecido
        distances = [self.euclidean_distance(x, x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        pass