#===============================================================================
# Exemplo: segmentação de uma imagem em escala de cinza.
#-------------------------------------------------------------------------------
# Autor: Bogdan T. Nassu
# Universidade Tecnológica Federal do Paraná
#===============================================================================

import sys
import timeit
import numpy as np
import cv2

#===============================================================================

INPUT_IMAGE =  'arroz.bmp'
#INPUT_IMAGE = 'documento-3mp.bmp'

# TODO: ajuste estes parâmetros!
NEGATIVO = False
THRESHOLD = 0.7
ALTURA_MIN = 10
LARGURA_MIN = 10
N_PIXELS_MIN = 400

#===============================================================================

def binariza (img, threshold):
    ''' Binarização simples por limiarização.

Parâmetros: img: imagem de entrada. Se tiver mais que 1 canal, binariza cada
              canal independentemente.
            threshold: limiar.
            
Valor de retorno: versão binarizada da img_in.'''

    # TODO: escreva o código desta função.
    # Dica/desafio: usando a função np.where, dá para fazer a binarização muito
    # rapidamente, e com apenas uma linha de código!

    img= np.where( img < threshold,0.0,1.0)
    return img
#-------------------------------------------------------------------------------

def rotula (img, largura_min, altura_min, n_pixels_min):
    '''Rotulagem usando flood fill. Marca os objetos da imagem com os valores
[0.1,0.2,etc].

Parâmetros: img: imagem de entrada E saída.
            largura_min: descarta componentes com largura menor que esta.
            altura_min: descarta componentes com altura menor que esta.
            n_pixels_min: descarta componentes com menos pixels que isso.

Valor de retorno: uma lista, onde cada item é um vetor associativo (dictionary)
com os seguintes campos:

'label': rótulo do componente.
'n_pixels': número de pixels do componente.
'T', 'L', 'B', 'R': coordenadas do retângulo envolvente de um componente conexo,
respectivamente: topo, esquerda, baixo e direita.'''

    # TODO: escreva esta função.
    # Use a abordagem com flood fill recursivo.
    rows = len(img)
    cols = len(img[0])
    label = 2
    n_pixel = 0
    componentes = []
    i=0
    # A matriz começa a explorar pela linha primeiro
    # Linha 0- 00 11 22 33 44 
    # Linha 1- 10 11 12 13 14
    # Linha 2- 20 21 22 23 24 
    # Linha 3- 30 31 32 33 34
    # Linha 4- 40 41 42 43 44
    #
    for col in range(cols):
        for row in range(rows):
            if(img [row][col] == 1):
                retangulo = {'L':col,'T':row,'R':col,'B':row}
                n_pixel =  rotula_arroz(img,row,col,label,retangulo,rows,cols)
                if(n_pixel > n_pixels_min and (retangulo['R'] -retangulo ['L']) > largura_min and (retangulo['B']-retangulo['T']) > altura_min):
                    componente = {'label' : label, "n_pixel" :n_pixel}
                    componente.update(retangulo)
                    componentes.append(componente)
                label +=1
        
    return componentes
#----------------------------------------------------------------------------------------
''' Parametros :    img : Imagem de entrada e saida
                    row : linha da matriz imagem
                    col : Coluna da matriz imagem
                    label : Label dado ao arroz
                    retangulo : Coordenada x do pixel mais a direita e mais a esquerda,e coordenada y mais a cima e a mais baixa

    Valores de retorno: N_pixel = numero de pixel do arroz
   '''
def rotula_arroz(img,row,col,label,retangulo,rows,cols):
    if(img[row,col]!= 1):
        return 0

    img[row,col] = label
    pixels = 1
    retangulo['T'] = min(retangulo['T'],row)
    retangulo['L'] = min(retangulo['L'],col)
    retangulo['R'] = max(retangulo['R'],col)
    retangulo['B'] = max(retangulo['B'],row)
    direcao = [(-1,0),(1,0),(0,-1),(0,1)]
    
    for dx,dy in direcao:
        if ((row + dx >= 0) and (col + dy >= 0) and (row + dx < rows)  and (col + dy < cols) ): 
            if(img[row + dx,col + dy] == 1):
                pixels += rotula_arroz(img,row + dx,col+ dy,label,retangulo,rows,cols)
   
  

    return pixels
#========================================================================================
def main ():

    # Abre a imagem em escala de cinza.
    img = cv2.imread (INPUT_IMAGE, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print ('Erro abrindo a imagem.\n')
        sys.exit ()

    # É uma boa prática manter o shape com 3 valores, independente da imagem ser
    # colorida ou não. Também já convertemos para float32.
    img = img.reshape ((img.shape [0], img.shape [1], 1))
    img = img.astype (np.float32) / 255

    # Mantém uma cópia colorida para desenhar a saída.
    img_out = cv2.cvtColor (img, cv2.COLOR_GRAY2BGR)

    # Segmenta a imagem.
    if NEGATIVO:
        img = 1 - img
    img = binariza (img, THRESHOLD)
    cv2.imshow ('01 - binarizada', img)
    cv2.imwrite ('01 - binarizada.png', img*255)

    start_time = timeit.default_timer ()
    componentes = rotula (img, LARGURA_MIN, ALTURA_MIN, N_PIXELS_MIN)
    n_componentes = len (componentes)
    print ('Tempo: %f' % (timeit.default_timer () - start_time))
    print ('%d componentes detectados.' % n_componentes)

    # Mostra os objetos encontrados.
    for c in componentes:
        cv2.rectangle (img_out, (c ['L'], c ['T']), (c ['R'], c ['B']), (0,0,1))

    cv2.imshow ('02 - out', img_out)
    cv2.imwrite ('02 - out.png', img_out*255)
    cv2.waitKey ()
    cv2.destroyAllWindows ()


if __name__ == '__main__':
    main ()

#===============================================================================
