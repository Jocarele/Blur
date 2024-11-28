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
import pandas as pd
#===============================================================================

INPUT_IMAGE =  'arroz_V2/205.bmp'
#INPUT_IMAGE = 'documento-3mp.bmp'

# TODO: ajuste estes parâmetros!
NEGATIVO = False
THRESHOLD = 0.7
THRESHOLD2 = 0.3
ALTURA_MIN = 1
LARGURA_MIN = 1
N_PIXELS_MIN = 1

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
                    #print(n_pixel)
                   
                label +=1

                
        
    return componentes
#----------------------------------------------------------------------------------------

def normaliza_local(img):
    #TODO: determinar o tamanho do kernel ser proximo do tamanho de um arroz
    kernel = np.ones((101,101), dtype=np.uint8)
    #substitui o pixel pelo menor pixel encontrado no kernel
    mini = cv2.erode(img,kernel)
    sigma = 10
    mini= cv2.GaussianBlur(mini, (0,0), sigma)
    #Substitui o pixel pelo maior pixel encontrado no kerel
    maxi = cv2.dilate(img,kernel)
    maxi= cv2.GaussianBlur(maxi, (0,0), sigma)

    img2 = np.copy(img)

    img2 = (img -mini)/(maxi-mini)
    return img2

#----------------------------

#-------------------------------------------------------------------------------------------

def magnitude(img):
    dx = cv2.Sobel(img,cv2.CV_32F,1,0)
    dy = cv2.Sobel(img,cv2.CV_32F,0,1)
    mag = cv2.magnitude(dx,dy)
    #minimo = np.min(mag)
    #maximo = np.max(mag)
    #print(minimo,maximo)
    mag =cv2.normalize(mag,None,alpha=0,beta=1,norm_type=cv2.NORM_MINMAX)
    return mag
#========================================================================================
def main ():

    # Abre a imagem em escala de cinza.
    img = cv2.imread (INPUT_IMAGE, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print ('Erro abrindo a imagem.\n')
        sys.exit ()

    img = img.astype (np.float32) / 255
    
    # Mantém uma cópia colorida para desenhar a saída.
    img_out = cv2.cvtColor (img, cv2.COLOR_GRAY2BGR)

    # Segmenta a imagem.
    if NEGATIVO:
        img = 1 - img

    rows,cols = np.shape(img)
    
    img= cv2.normalize(img,None,alpha=0,beta=1,norm_type=cv2.NORM_MINMAX)
    

    #expalhar o brilho da imagem
    img2 = normaliza_local(img)

    #img3 = binariza (img2, THRESHOLD)
    kernel = 201
    img = (img * 255).astype(np.uint8)  # Normalize para 8 bits
    img3 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, kernel, -30)
    img3 = img3.astype (np.float32) / 255

    #Limpa ruido
    kernel = np.ones((5,5), dtype=np.uint8)
    
    #img3 = cv2.medianBlur(img3,9)

    img3 = cv2.erode(img3,kernel)
    img3 = cv2.dilate(img3,(kernel))
    
    cv2.imshow ('ORIGINAL', img/255)
    cv2.imwrite ('ORIGINAL.png', img)
    cv2.imshow ('01 - normalizado local', img2)
    cv2.imwrite ('01 - normalizado local.png', img2*255)
    cv2.imshow ('03 - binzarizada', img3)
    cv2.imwrite ('03 - binzarizada.png', img3*255)

    

    start_time = timeit.default_timer ()
    componentes = rotula (img3, LARGURA_MIN, ALTURA_MIN, N_PIXELS_MIN)
    n_componentes = len (componentes)
    print ('Tempo: %f' % (timeit.default_timer () - start_time))
    print ('%d componentes detectados.' % n_componentes)

    # Mostra os objetos encontrados.
    tam = len(componentes)
    pixel =np.zeros(tam)
    for c in range(0,tam):
        pixel[c] = componentes[c]["n_pixel"]
    desvio = 100000
    desvio = np.std(pixel)

    
    pixel = np.sort(pixel)
    mediana =0

    pixels_sem_outliers = extract_outliers(pixel)

    pixel_buffer = pixel.copy()
    #20
    while desvio > 20:

       
        mediana = np.median(pixel_buffer)
        distancias = np.abs(pixel_buffer - mediana)
        
        # Remover o valor mais distante da mediana
        indice_remover = np.argmax(distancias)
        pixel_buffer = np.delete(pixel_buffer, indice_remover)
        desvio = np.std(pixel_buffer)
        #print("desvio",desvio)

    print("desvio padrao",desvio)
    print(pixel_buffer)
    
    
    x=0


    max_arroz = pixel_buffer[-1]/mediana
    min_arroz = pixel[0]/mediana

    print("max arroz",max_arroz)
    print("min arroz",min_arroz)

   # print("Parametros",min_arroz,1/min_arroz)

    #Estou pegando arroz grande e somando +1. ex 340/240
    for i in range (int(tam/2),tam):
        if(pixel[i]/mediana > max_arroz):
            #print(pixel[i]/mediana, x,pixel[i]/mediana - int(pixel[i]/mediana))
            #print("pixel[",i,"] = ",pixel[i] )

            x += int(pixel[i]/mediana) -1
            if(pixel[i]/mediana - int(pixel[i]/mediana)>=min_arroz):
                x+=1
    
    print(x+tam)
    cv2.imshow ('02 - out', img_out)
    cv2.imwrite ('02 - out.png', img_out*255)

    #calcula mediana e desvio

    '''mediana_outl = np.median(pixels_sem_outliers)
    desvio_outl = np.std(pixels_sem_outliers)
    cont_arroz = 0
    pixels_out = pixel_buffer - pixels_sem_outliers
    for pixel in pixels_out:
        if pixel <= (mediana_outl + 2 * desvio_outl):
            cont_arroz += 1
    #Para cada pixel entre o intervalo de mediana +- 2desvios, aumenta o contador
    #printa contador
    print("printa sem outl")
    print(pixels_sem_outliers)
    print(cont_arroz)
    print(mediana_outl)
    print(desvio_outl)
    #cv2.waitKey ()
    #cv2.destroyAllWindows ()'''


def extract_outliers(pixels):

    # Dados de exemplo
    data = {'values': pixels}  # Lista com valores (incluindo outliers)
    df = pd.DataFrame(data)

    # Cálculo do IQR
    Q1 = df['values'].quantile(0.12)  # Primeiro quartil (25%)
    Q3 = df['values'].quantile(0.23)  # Terceiro quartil (75%)
    IQR = Q3 - Q1                     # Intervalo interquartil (IQR)

    # Definir limites para outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Filtrar dados dentro dos limites
    filtered_df = df[(df['values'] >= lower_bound) & (df['values'] <= upper_bound)]
    filtered = filtered_df['values'].to_numpy()
    
    mean = np.mean(filtered)
    
    big_blobs_df = df[df['values'] > upper_bound]
    big_blobs = big_blobs_df['values'].to_numpy()
    arroz_count = len(filtered)
    
    for blob in big_blobs:
        arroz_count += round(blob/mean)

    print("AQUIIIIIIII")
    print(filtered)
    print(big_blobs)
    print(arroz_count)
    
    return arroz_count

if __name__ == '__main__':
    main()