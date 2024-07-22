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

class ValoresParaConjuntoPixelErro(Exception):
    def __init__(self, message="A combinação do conjunto está incorreta. Verifique uma que possa ser utilizada"):
        self.message = message
        super().__init__(self.message)

