#===============================================================================
# Exemplo: segmentação de uma imagem em escala de cinza.
#-------------------------------------------------------------------------------
# Autor: 
# Universidade Tecnológica Federal do Paraná
#===============================================================================

import sys
import timeit
import numpy as np
import cv2

#===============================================================================

INPUT_IMAGE =  '60.bmp'
#INPUT_IMAGE = 'documento-3mp.bmp'

# TODO: ajuste estes parâmetros!
NEGATIVO = False
THRESHOLD = 0.7
THRESHOLD2 = 0.3
ALTURA_MIN = 10
LARGURA_MIN = 10
N_PIXELS_MIN = 100

#===============================================================================

def binariza (img, threshold):
    ''' Binarização simples por limiarização.

Parâmetros: img: imagem de entrada. Se tiver mais que 1 canal, binariza cada
              canal independentemente.
            threshold: limiar.
            
Valor de retorno: versão binarizada da img_in.'''



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
    rows,cols = np.shape(img)
    pilha = []
    label = 2
    
    componentes = []
    
    
    
    for row in range(rows):
        for col in range(cols):
            
            if(img [row][col] == 1):
                coord = (row,col)
                pilha.append(coord)
                n_pixel =1
                retangulo = {'L':col,'T':row,'R':col,'B':row}
                while (len(pilha) != 0):
                    
                    y,x = pilha.pop()
                   
                    direcao = [(-1,0),(1,0),(0,-1),(0,1)]
                    for dx,dy in direcao:
                        if ((y + dy >= 0) and (x + dx >= 0) and (y + dy < rows)  and (x + dx < cols) ): 
                            if img[y+dy,x+dx] == 1:
                                img[y+dy,x+dx] = label
                                n_pixel += 1
                                retangulo['T'] = min(retangulo['T'],y+dy)
                                retangulo['L'] = min(retangulo['L'],x+dx)
                                retangulo['R'] = max(retangulo['R'],x+dx)
                                retangulo['B'] = max(retangulo['B'],y+dy)
                                pilha.append((y+dy,x+dx))
                
                if(n_pixel > n_pixels_min and (retangulo['R'] -retangulo ['L']) > largura_min and (retangulo['B']-retangulo['T']) > altura_min):
                    componente = {'label' : label, "n_pixel" :n_pixel}
                    componente.update(retangulo)
                    componentes.append(componente)
                   
                label +=1

                
        
    return componentes
#----------------------------------------------------------------------------------------

#========================================================================================
def main ():

    # Abre a imagem em escala de cinza.
    img = cv2.imread (INPUT_IMAGE, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print ('Erro abrindo a imagem.\n')
        sys.exit ()

    # É uma boa prática manter o shape com 3 valores, independente da imagem ser
    # colorida ou não. Também já convertemos para float32.
    #img = img.reshape ((img.shape [0], img.shape [1], 1))
    img = img.astype (np.float32) / 255
    
    # Mantém uma cópia colorida para desenhar a saída.
    img_out = cv2.cvtColor (img, cv2.COLOR_GRAY2BGR)

    # Segmenta a imagem.
    if NEGATIVO:
        img = 1 - img

    rows,cols = np.shape(img)
    resized = cv2.resize(img, (cols*30, rows*30), 0, 0, interpolation = cv2.INTER_NEAREST)
    resized = cv2.resize(resized, (cols, rows), 0, 0, interpolation = cv2.INTER_LINEAR)
    dx = cv2.Sobel(resized,cv2.CV_32F,1,0)
    dy = cv2.Sobel(resized,cv2.CV_32F,0,1)
    mag = cv2.magnitude(dx,dy)
    #minimo = np.min(mag)
    #maximo = np.max(mag)
    #print(minimo,maximo)
    mag =cv2.normalize(mag,None,alpha=0,beta=1,norm_type=cv2.NORM_MINMAX)
    mag = binariza(mag,THRESHOLD2)
    mag = cv2.dilate(mag,(3,3),iterations=1)

    
    img2 = img - mag
    
    img2 = binariza (img, THRESHOLD)
    
    #cv2.imshow ('01 - binarizada', mag)
    #cv2.imwrite ('01 - binarizada.png', mag*255)
    cv2.imshow ('ORIGINAL', img)
    cv2.imwrite ('ORIGINAL.png', img*255)
    cv2.imshow ('02 - binarizada2', img2)
    cv2.imwrite ('02 - binarizada2.png', img2*255)

    start_time = timeit.default_timer ()
    componentes = rotula (img2, LARGURA_MIN, ALTURA_MIN, N_PIXELS_MIN)
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
    main()