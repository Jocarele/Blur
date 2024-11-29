#===============================================================================
# Projeto 4 - Contagem de arroz 
#-------------------------------------------------------------------------------
# Autor: João Lucas M. Camilo
#        Viviane Ruotolo
# Universidade Tecnológica Federal do Paraná
#===============================================================================


import sys
import timeit
import numpy as np
import cv2

#===============================================================================

INPUT_IMAGE =  '205.bmp'

ALTURA_MIN = 1
LARGURA_MIN = 1
N_PIXELS_MIN = 80

#===============================================================================


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

#----------------------------


#========================================================================================
def main ():

    # Abre a imagem em escala de cinza.
    img = cv2.imread (INPUT_IMAGE, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print ('Erro abrindo a imagem.\n')
        sys.exit ()

    img = img.astype (np.float32) / 255
    rows,cols = np.shape(img)
    
    img= cv2.normalize(img,None,alpha=0,beta=1,norm_type=cv2.NORM_MINMAX)
    
    #Binarização Local
    kernel = 201
    img = (img * 255).astype(np.uint8)  
    img3 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, kernel, -30)
    img3 = img3.astype (np.float32) / 255

    #Limpa ruido
    kernel = np.ones((5,5), dtype=np.uint8)
    img3 = cv2.erode(img3,kernel)
    img3 = cv2.dilate(img3,(kernel))
    
    cv2.imshow ('ORIGINAL', img/255)
    cv2.imwrite ('ORIGINAL.png', img)
    cv2.imshow ('03 - binzarizada', img3)
    cv2.imwrite ('03 - binzarizada.png', img3*255)

    start_time = timeit.default_timer ()
    componentes = rotula (img3, LARGURA_MIN, ALTURA_MIN, N_PIXELS_MIN)
    n_componentes = len (componentes)
    print ('%d blobs detectados.' % n_componentes)

    pixel =np.zeros(n_componentes)
    for c in range(0,n_componentes):
        pixel[c] = componentes[c]["n_pixel"]

    pixel = np.sort(pixel)
    desvio = np.std(pixel)
    mediana = np.median(pixel)
    pixel_buffer = pixel.copy()
    
    while desvio/mediana >= 0.09: 

       
        distancias = np.abs(pixel_buffer - mediana)
        
        # Remover o valor mais distante da mediana
        indice_remover = np.argmax(distancias)
        pixel_buffer = np.delete(pixel_buffer, indice_remover)

        mediana = np.median(pixel_buffer)
        desvio = np.std(pixel_buffer)

    
    cont_arroz=0

    #Maior valor de blob encontrado que é considerado Arroz
    max_arroz = pixel_buffer[-1]
    #Proporção entre o blob mais pequeno e a mediana. 
    #Responsavel por determinar se as casas decimais são considerados arroz ou não.
    min_arroz = pixel[0]/mediana

    #Conta a quantidade de arroz nos blobs
    for i in range (0,n_componentes):
        if(pixel[i] > max_arroz):

            cont_arroz += int(pixel[i]/mediana) -1
            if(pixel[i]/mediana - int(pixel[i]/mediana)>=min_arroz):
                cont_arroz+=1
    
    print("Quantidade de arroz: ", cont_arroz+n_componentes)
    print ('Tempo: %f' % (timeit.default_timer () - start_time))

    cv2.waitKey ()
    cv2.destroyAllWindows ()


if __name__ == '__main__':
    main()