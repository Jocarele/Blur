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

#==============================================================================
#INPUT_IMAGE = 'documento-3mp.bmp'

INPUT_IMAGE = 'arroz.bmp'

# TODO: ajuste estes parâmetros!
NEGATIVO = False
THRESHOLD = 0.7
ALTURA_MIN = 1
LARGURA_MIN = 1
N_PIXELS_MIN = 10

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
    caixas = []
    i=0
    for row in range(rows):
        for col in range(cols):
            if(img [row][col] == 1):
                arroz =  rotula_arroz(img,row,col,label,n_pixel,rows,cols)
                if(arroz[5] > n_pixels_min and arroz[2] - arroz[1] > largura_min and arroz[4] - arroz[3] > altura_min):
                    componente = {'label' : label, "n_pixel" :arroz[5],
                               'T' :arroz[3], 'L' :arroz[1],
                               'B' :arroz[4], 'R' :arroz[2]}
                    componentes.append(componente)
                    label +=1
                    img = arroz[0]
              

    
    
    return componentes
#----------------------------------------------------------------------------------------
''' Parametros :    img : Imagem de entrada e saida
                    row : linha da matriz imagem
                    col : Coluna da matriz imagem
                    label : Label dado ao arroz
                    n_pixel : Quantidade de pixels explorados na recursão

    Valores de retorno: Uma lista com os valores: 
                    img : a imagem atualizada com o Label
                    col_min : o pixel mais a baixo do arroz
                    col_max : o pixel mais a direita do arroz
                    row_min : o pixel mais a cima do arroz
                    row_max : o pixel mais a baixo do arroz
                    n_pixel : quantidade de pixels do arroz
   '''
def rotula_arroz(img,row,col,label,n_pixel,rows,cols):
    row_min = row
    row_max = row
    col_min = col
    col_max = col
    img[row][col] = label
    n_pixel += 1

    if (row - 1 > 0):
        if(img[row-1][col] == 1):
            arroz = rotula_arroz(img,row-1,col,label,n_pixel,rows,cols)
            if (arroz[5] > n_pixel):
                img = arroz[0]
                n_pixel = arroz[5]
            if (arroz[2] > col_max):
                col_max = arroz[2]
            if (arroz[4] > row_max):
                row_max = arroz[4] 
            if (arroz[1] < col_min):
                col_min = arroz[1]
            if (arroz[3] < row_min):
                row_min = arroz[3]
   
    if (col - 1 > 0):
        if(img[row][col-1] == 1):
            arroz = rotula_arroz(img,row,col-1,label,n_pixel,rows,cols)
            if (arroz[5] > n_pixel):
                img = arroz[0]
                n_pixel = arroz[5]
            if (arroz[2] > col_max):
                col_max = arroz[2]
            if (arroz[4] > row_max):
                row_max = arroz[4] 
            if (arroz[1] < col_min):
                col_min = arroz[1]
            if (arroz[3] < row_min):
                row_min = arroz[3]
 
    if (row + 1 < rows-1):
        if(img[row+1][col] == 1):
            arroz = rotula_arroz(img,row+1,col,label,n_pixel,rows,cols)
            if (arroz[5] > n_pixel):
                img = arroz[0]
                n_pixel = arroz[5]
            if (arroz[2] > col_max):
                col_max = arroz[2]
            if (arroz[4] > row_max):
                row_max = arroz[4] 
            if (arroz[1] < col_min):
                col_min = arroz[1]
            if (arroz[3] < row_min):
                row_min = arroz[3]

    if (col + 1 < cols-1):
        if(img[row][col+1] == 1):
            arroz = rotula_arroz(img,row,col+1,label,n_pixel,rows,cols)
            if (arroz[5] > n_pixel):
                img = arroz[0]
                n_pixel = arroz[5]
            if (arroz[2] > col_max):
                col_max = arroz[2]
            if (arroz[4] > row_max):
                row_max = arroz[4] 
            if (arroz[1] < col_min):
                col_min = arroz[1]
            if (arroz[3] < row_min):
                row_min = arroz[3]
'''
    if (row - 1 > 0 and col-1 > 0):
        if(img[row-1][col-1] == 1):
            arroz = rotula_arroz(img,row-1,col-1,label,n_pixel,rows,cols)
            if (arroz[5] > n_pixel):
                img = arroz[0]
                n_pixel = arroz[5]
            if (arroz[2] > col_max):
                col_max = arroz[2]
            if (arroz[4] > row_max):
                row_max = arroz[4] 
            if (arroz[1] < col_min):
                col_min = arroz[1]
            if (arroz[3] < row_min):
                row_min = arroz[3]
   
    if (row - 1 > 0 and col + 1 < cols-1):
        if(img[row-1][col+1] == 1):
            arroz = rotula_arroz(img,row-1,col+1,label,n_pixel,rows,cols)
            if (arroz[5] > n_pixel):
                img = arroz[0]
                n_pixel = arroz[5]
            if (arroz[2] > col_max):
                col_max = arroz[2]
            if (arroz[4] > row_max):
                row_max = arroz[4] 
            if (arroz[1] < col_min):
                col_min = arroz[1]
            if (arroz[3] < row_min):
                row_min = arroz[3]
 
    if (row + 1 < rows-1 and col -1 > 0):
        if(img[row+1][col-1] == 1):
            arroz = rotula_arroz(img,row+1,col-1,label,n_pixel,rows,cols)
            if (arroz[5] > n_pixel):
                img = arroz[0]
                n_pixel = arroz[5]
            if (arroz[2] > col_max):
                col_max = arroz[2]
            if (arroz[4] > row_max):
                row_max = arroz[4] 
            if (arroz[1] < col_min):
                col_min = arroz[1]
            if (arroz[3] < row_min):
                row_min = arroz[3]

    if (col + 1 < cols-1 and row + 1 < rows-1):
        if(img[row+1][col+1] == 1):
            arroz = rotula_arroz(img,row+1,col+1,label,n_pixel,rows,cols)
            if (arroz[5] > n_pixel):
                img = arroz[0]
                n_pixel = arroz[5]
            if (arroz[2] > col_max):
                col_max = arroz[2]
            if (arroz[4] > row_max):
                row_max = arroz[4] 
            if (arroz[1] < col_min):
                col_min = arroz[1]
            if (arroz[3] < row_min):
                row_min = arroz[3]
'''

    arroz = [img,col_min,col_max,row_min,row_max,n_pixel]
    return arroz
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
