import numpy as np


class KMeans():
    def __init__(self, dadosObservados, qntCluster):
        self.__dadosObservados = dadosObservados
        self.__qntCluster = qntCluster
        self.__qntObservacoes = self.__dadosObservados.shape[0]
        self.__qntVariaveis = self.__dadosObservados.shape[1]

    def getObservacoes(self):
        return self.__observacoescomClasse
    
    def getCentroid(self):
        return self.__centroid
    
    def criarCluster(self):
        self.__selecionarPontosCentroid()
        while True:
            centroidAntigo = self.__centroid.copy()
            self.__calcularDistanciaPontoCentroid()
            self.__encontrarClasseObservacao()
            self.__calcularCentroid()
            if np.linalg.norm(centroidAntigo - self.__centroid) < 1**(-4):
                break

    def __calcularCentroid(self):
        for i in range(self.__qntCluster):      
            selecionarPontos = self.__observacoescomClasse[:,-1] == i
            self.__centroid[i,:] = np.mean(self.__dadosObservados[selecionarPontos],axis=0)

    def __selecionarPontosCentroid(self):
        posicaoObservacoes = np.random.choice(range(self.__qntObservacoes),
                                                    self.__qntCluster, replace=False)
        self.__centroid = self.__dadosObservados[posicaoObservacoes]
    
    def __calcularDistanciaPontoCentroid(self):
        distanciaPontoCentroid = np.ones([self.__qntObservacoes,self.__qntCluster])
        for i in range(self.__qntObservacoes):
            for j in range(self.__qntCluster):
                observacao = self.__dadosObservados[i,:]
                centroidSelecionada = self.__centroid[j,:]
                distanciaPontoCentroid[i,j] = np.linalg.norm(observacao - centroidSelecionada)
        return distanciaPontoCentroid

    def __encontrarClasseObservacao(self):
        matrizDistancias = self.__calcularDistanciaPontoCentroid()
        classes = np.argmin(matrizDistancias, axis=1).reshape(self.__qntObservacoes,1)
        self.__observacoescomClasse = np.concatenate([self.__dadosObservados, classes], axis=1)


