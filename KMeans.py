import numpy as np
import cv2


class KMeans():
    def __init__(self, dadosObservados, qntCluster):
        self.__dadosObservados = dadosObservados
        self.__qntCluster = qntCluster
        self.__qntObservacoes = self.__dadosObservados.shape[0]
        self.__centroid
        self.__ObservacoescomClasse

    def criarCluster(self):
        self.__selecionarPontosCentroid()
        for i in range(1000):
            self.__calcularDistanciaPontoCentroid()
            self.__encontrarClasseObservacao()
            self.__calcularCentroid()

    def __calcularCentroid(self):
        for i in range(self.__qntCluster):
            selecionarPontos = self.__ObservacoescomClasse[:,-1] == i
            self.__centroid[i,:] = np.mean(self.__dadosObservados[selecionarPontos],axis=1)

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
                distanciaPontoCentroid[i,j] = np.linalg.norm(observacao, centroidSelecionada)
        return distanciaPontoCentroid

    def __encontrarClasseObservacao(self, centroid):
        matrizDistancias = self.__calcularDistanciaPontoCentroid(centroid)
        classes = np.argmax(matrizDistancias, axis=1).reshape(self.__qntObservacoes,0)
        self.__ObservacoescomClasse = np.concatenate([self.__dadosObservados, classes], axis=1)

