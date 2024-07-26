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
        self.__diferencaCentroidAntigo = [np.inf]
        altaConsecutiva=[]
        while True:
            self.__centroidAntigo = self.__centroid.copy()
            self.__encontrarClasseObservacao()
            self.__calcularCentroid()
            self.__calcularDiferencaCentroid()
            if self.__verificarConvergencia(altaConsecutiva):
                break

    def __calcularDiferencaCentroid(self):
        diferenca = np.linalg.norm([self.__centroidAntigo - self.__centroid])
        self.__diferencaCentroidAntigo.append(diferenca)
        
    def __verificarConvergencia(self, altaConsecutiva):
        evolucaoNovoCentroid = self.__diferencaCentroidAntigo[-2] - self.__diferencaCentroidAntigo[-1]
        if evolucaoNovoCentroid<=0:
            altaConsecutiva.append(1)
            if len(altaConsecutiva) == 2:
                self.__centroid = self.__melhorCentroid
                return True
            if len(altaConsecutiva) == 1:
                self.__melhorCentroid = self.__centroidAntigo
        else:
            altaConsecutiva = []
        return False

    def __calcularCentroid(self):
        centroidAntigo = self.__centroid.copy()
        for i in range(self.__qntCluster):      
            selecionarPontos = self.__observacoescomClasse[:,-1] == i
            if np.count_nonzero(selecionarPontos) == 0:
                self.__centroid[i,:] = centroidAntigo[i,:]
            else:
                self.__centroid[i,:] = np.mean(self.__dadosObservados[selecionarPontos],axis=0)

    def __selecionarPontosCentroid(self):
        posicaoObservacoes = np.random.choice(range(self.__qntObservacoes),
                                                    self.__qntCluster, replace=False)
        self.__centroid = self.__dadosObservados[posicaoObservacoes]
    
    def __calcularDistanciaPontoCentroid(self):
        distanciaPontoCentroid = np.ones([self.__qntObservacoes,self.__qntCluster])
        for j in range(self.__qntCluster):
            relacaoCentroidObservacoes = self.__dadosObservados - self.__centroid[j,:]
            distanciaMatrizCentroid = np.linalg.norm(relacaoCentroidObservacoes, axis=1)
            distanciaPontoCentroid[:,j:j+1] = distanciaMatrizCentroid.reshape(self.__qntObservacoes,1)
        return distanciaPontoCentroid

    def __encontrarClasseObservacao(self):
        matrizDistancias = self.__calcularDistanciaPontoCentroid()
        classes = np.argmin(matrizDistancias, axis=1).reshape(self.__qntObservacoes,1)
        self.__observacoescomClasse = np.concatenate([self.__dadosObservados, classes], axis=1)

