import Imagem as im
import matplotlib.pyplot as plt

imagemOriginal = plt.imread('Flor.jpg')
imagemFlor = im.Imagem(imagemOriginal, 2, 2, 100)
imagemFlor.compactarImagem()
imagemCompactada = imagemFlor.getConjuntoImagemCompactada()
imagemFlor.mostrarImagemCompactada()