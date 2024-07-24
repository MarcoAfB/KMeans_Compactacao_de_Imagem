import numpy as np
import matplotlib.pyplot as plt
import KMeans as km

class Imagem():
    def __init__(self, imagem, larguraConjuntoPixel, comprimentoConjuntoPixel, qntCluster):
        self.__imagem = imagem
        self.__qntComprimentoPixel = imagem.shape[0]
        self.__qntLarguraPixel = imagem.shape[1]
        try:
            self.__qntLayer = imagem.shape[2]
        except:
            self.__qntLayer = 1
        self.__qntVariaveis = larguraConjuntoPixel*comprimentoConjuntoPixel*self.__qntLayer
        self.__qntObservacoes = int((self.__qntLarguraPixel/larguraConjuntoPixel)*(self.__qntComprimentoPixel/comprimentoConjuntoPixel))
        self.__conjuntoDados = np.ones([self.__qntObservacoes,self.__qntVariaveis])
        self.__qntCluster = qntCluster
        self.__larguraConjuntoPixel = larguraConjuntoPixel
        self.__comprimentoConjuntoPixel = comprimentoConjuntoPixel
        if self.__verificarDimensao(larguraConjuntoPixel, comprimentoConjuntoPixel):
            self.__sugerirTamanhoPixel()
            raise ValoresParaConjuntoPixelErro()
        
    def __verificarDimensao(self, x, y):
        if self.__qntLarguraPixel % x != 0 or self.__qntComprimentoPixel % y != 0:
            return True
    
    def __sugerirTamanhoPixel(self):
        comprimento=[]
        largura=[]
        for i in range(2, 100):
            if self.__qntComprimentoPixel % i == 0:
                comprimento.append(i)
            if self.__qntLarguraPixel % i == 0:
                largura.append(i)    
        print("Possíveis valores como largura:\n",largura)
        print("Possíveis valores como comprimento:\n",comprimento)

    def getConjuntoImagemCompactada(self):
        return self.__imagemReduzida
    
    def mostrarImagemCompactada(self):
        plt.imshow(self.__imagemReduzida)

    def compactarImagem(self):
        self.__criarConjuntoDadosparaImagem()
        self.__clusterizacaoConjuntoDados()
        self.__substituirPixelIndividualporCentroid()
        self.__transformarConjuntoDadosemImagem()

    def __criarConjuntoDadosparaImagem(self):
        m=0
        for i in range(0, self.__qntComprimentoPixel, self.__larguraConjuntoPixel):
            for j in range(0, self.__qntLarguraPixel, self.__comprimentoConjuntoPixel):
                obs = self.__imagem[i:i+self.__larguraConjuntoPixel,j:j+self.__comprimentoConjuntoPixel].reshape(1,self.__qntVariaveis)
                self.__conjuntoDados[m,:] = obs
                m+=1
        return self.__conjuntoDados
    
    def __clusterizacaoConjuntoDados(self):
        algoritmoCluster = km.KMeans(self.__conjuntoDados, self.__qntCluster)
        algoritmoCluster.criarCluster()
        self.__conjuntoDadosReduzidos = algoritmoCluster.getObservacoes()
        self.__conjuntoCentroids = algoritmoCluster.getCentroid()

    def __substituirPixelIndividualporCentroid(self):
        self.__conjuntoPixelCentroid = np.ones([self.__conjuntoDadosReduzidos.shape[0], 
                                                self.__conjuntoDadosReduzidos.shape[1]-1])
        for i in range(self.__conjuntoDadosReduzidos.shape[0]):
            classe = int(self.__conjuntoDadosReduzidos[i,-1])
            self.__conjuntoPixelCentroid[i,:] = self.__conjuntoCentroids[classe,:]

    def __transformarConjuntoDadosemImagem(self):
        self.__imagemReduzida = np.ones([self.__qntComprimentoPixel, self.__qntLarguraPixel, self.__qntLayer])
        m=0
        for i in range(0, self.__qntComprimentoPixel, self.__comprimentoConjuntoPixel):
            for j in range(0, self.__qntLarguraPixel, self.__larguraConjuntoPixel):
                formatoImg = self.__conjuntoPixelCentroid[m].reshape(self.__comprimentoConjuntoPixel,self.__larguraConjuntoPixel,self.__qntLayer)
                self.__imagemReduzida[i:i+self.__comprimentoConjuntoPixel,j:j+self.__larguraConjuntoPixel] = formatoImg
                m+=1
        self.__imagemReduzida = self.__imagemReduzida.astype('uint8')
        return self.__imagemReduzida
    
class ValoresParaConjuntoPixelErro(Exception):
    def __init__(self, message="A combinação do conjunto está incorreta. Verifique uma que possa ser utilizada"):
        self.message = message
        super().__init__(self.message)

