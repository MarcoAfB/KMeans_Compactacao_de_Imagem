import numpy as np
import matplotlib.pyplot as plt

class Imagem():

    def __init__(self, imagem):
        self.__imagem = imagem
        self.__qntComprimentoPixel = imagem.shape[0]
        self.__qntLarguraPixel = imagem.shape[1]
        try:
            self.__qntLayer = imagem.shape[2]
        except:
            self.__qntLayer = 1
        self.__sugerirTamanhoPixel()

    def __verificarDimensao(self, x, y):
        if self.__qntLarguraPixel % x != 0 or self.__qntComprimentoPixel % y != 0:
            raise ValoresParaConjuntoPixelErro()
    
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

    def __criarConjuntoDados(self, larguraConjuntoPixel, comprimentoConjuntoPixel):
        self.__verificarDimensao(larguraConjuntoPixel, comprimentoConjuntoPixel)
        qntVariaveis = larguraConjuntoPixel*comprimentoConjuntoPixel*self.__qntLayer
        qntObservacoes = int((self.__qntLarguraPixel/larguraConjuntoPixel)*(self.__qntComprimentoPixel/comprimentoConjuntoPixel))
        self.__conjuntoDados = np.ones([qntObservacoes,qntVariaveis])
        self.__larguraConjuntoPixel = larguraConjuntoPixel
        self.__comprimentoConjuntoPixel = comprimentoConjuntoPixel
        m=0
        for i in range(0, self.__qntComprimentoPixel, larguraConjuntoPixel):
            for j in range(0, self.__qntLarguraPixel, comprimentoConjuntoPixel):
                obs = self.__imagem[i:i+larguraConjuntoPixel,j:j+comprimentoConjuntoPixel].reshape(1,qntVariaveis)
                self.__conjuntoDados[m,:] = obs
                m+=1
        return self.__conjuntoDados

    def __criarImagemReduzida(self):
        self.__imagemReduzida = np.ones([self.__qntComprimentoPixel, self.__qntLarguraPixel, self.__qntLayer])
        m=0
        for i in range(0, self.__qntComprimentoPixel, self.__comprimentoConjuntoPixel):
            for j in range(0, self.__qntLarguraPixel, self.__larguraConjuntoPixel):
                formatoImg = self.__conjuntoDados[m].reshape(self.__comprimentoConjuntoPixel,self.__larguraConjuntoPixel,self.__qntLayer)
                self.__imagemReduzida[i:i+self.__comprimentoConjuntoPixel,j:j+self.__larguraConjuntoPixel] = formatoImg
                m+=1
        self.__imagemReduzida = self.__imagemReduzida.astype('uint8')
        return self.__imagemReduzida

class ValoresParaConjuntoPixelErro(Exception):
    def __init__(self, message="A combinação do conjunto está incorreta. Verifique uma que possa ser utilizada"):
        self.message = message
        super().__init__(self.message)

