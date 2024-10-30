#===============================================================================
# Exemplo: Blur em imagens em escala de cinza e colorida
#-------------------------------------------------------------------------------
# Autores:  João Lucas Marques Camilo
#           Viviane Ruotolo        
# Universidade Tecnológica Federal do Paraná
#===============================================================================

import sys
import timeit
import numpy as np
import cv2

#===============================================================================

INPUT_IMAGE =  'bOriginal.bmp'
#INPUT_IMAGE = 'Original.bmp'

# TODO: ajuste estes parâmetros!
NEGATIVO = False
THRESHOLD = 0.7

TAMANHO_JANELAH = 50
TAMANHO_JANELAW = 50


#===============================================================================

def binariza (img, threshold):
    img= np.where( img < threshold,0.0,1.0)
    return img
#-------------------------------------------------------------------------------
'''
Parametros: img

Retorno: uma outra imagem que cada pixel tem o valor da média dos pixels, delimitado pela janela H X W,
da imagem original. Ou seja, uma imagem borrada  


'''
def blur(img,dimension):
    rows,cols,channel = dimension
    
    img_out = img.copy()
    hT = TAMANHO_JANELAH
    wT = TAMANHO_JANELAW
    i_ht = int(hT/2)
    i_wt = int(wT/2)
    
    for row in range (0,rows):
        for col in range (0,cols):
              
            media = [0.0,0.0,0.0]
            #crio uma margem preta
            if row < i_ht or col < i_wt or row > (rows - i_ht-1) or col > (cols -i_wt-1):
                img_out[row][col] = 0
                continue
            ##range é excludente, adicionar +1
            for h in range(row-i_ht ,row+i_ht+1):
                for w in range(col-i_wt ,col+i_wt+1):        
                    for c in range (0,channel):
                        media[c] += img[h][w][c] 
                    

                                  
            for i in range (0,channel):
                media[i] = media[i]/(hT*wT)
                img_out[row][col][i] = media[i]
            
                
    return img_out

def blur_separavel(img,dimension):
    rows, cols, channel = dimension

    img_out = img.copy()
    img_buffer = img.copy()
    hT = TAMANHO_JANELAH
    wT = TAMANHO_JANELAW
    meio_ht = int(hT/2)
    meio_wt = int(wT/2)

    #Borrar imagem na horizontal
    for row in range(0, rows):
        primeira_coluna = True
        soma = 0 
        for col in range(0, cols):
            #margem preta para colunas não utilizadas
            if col < meio_wt or col > (cols - meio_wt - 1):
                img_buffer[row, col] = 0
                continue
            #Janela deslizante 1xW
            if primeira_coluna:
                for x in range(col - meio_wt, col + meio_wt +1):#Fazer a soma completa na primeira janela 
                    soma += img[row, x]
                primeira_coluna = False
            else:
                soma = soma - img[row, col - meio_wt - 1] + img[row, col + meio_wt]
            img_buffer[row, col] = soma/wT
            
    #Borrar imagem na vertical
    for col in range(0, cols):
        primeira_linha = True
        soma = 0
        for row in range(0, rows):
            #margem preta para linhas e colunas não utilizadas
            if row < meio_ht or col < meio_wt or row > (rows - meio_ht - 1) or col > (cols - meio_wt - 1):
                img_out[row, col] = 0
                continue
            #Janela deslizante Hx1
            if primeira_linha:
                for y in range(row - meio_ht, row + meio_ht + 1):
                    soma += img_buffer[y, col]    
                primeira_linha = False
            else:
                soma = soma - img_buffer[row - meio_ht - 1, col] + img_buffer[row + meio_ht, col]
            img_out[row, col] = soma/hT
    return img_out

def integral(img,rows,cols,channel):
    
    
    img_out = img.copy()
    
    
    for row in range (0,rows):
        for col in range (0,cols):
            for c in range (0,channel):
            
                if row == 0 and col ==0:
                    continue
                elif row == 0:
                    img_out[row,col,c] += img_out[row,col-1,c]
                elif col == 0: 
                    img_out[row,col,c] += img_out[row-1,col,c]
                else:
                    img_out[row,col,c] += img_out[row,col-1,c] + img_out[row-1,col,c] - img_out[row-1,col-1,c]

    return img_out
            
            
def blur_integral(img,dimension): 
    rows,cols,channel = dimension
    img_buffer =integral(img,rows,cols,channel)
    img_out = img.copy()
    
    
   
    hT = TAMANHO_JANELAH
    wT = TAMANHO_JANELAW
    
    i_ht = int(hT/2)
    i_wt = int(wT/2)
    
    for row in range (0,rows):   
        tamh = 0
         #o pixel mais a baixo da janela
        row_baixo_dir = row +i_ht

        #verifica se o pixel mais a baixo da janela não esta na imagem
        #caso não esteja, arruma a posição do pixel mais a baixo para o limite da imagem (rows -1)
        if row > rows -i_ht -1:
            #Pega o valor de quantos pixels foi perdido na janela
            tamh = row_baixo_dir -rows+1
            row_baixo_dir = rows -1
            

        for col in range (0,cols):
            tamw = 0
            #o pixel mais a direita da janela
            col_baixo_dir = col +i_wt
            
            #verifica se o pixel mais a direita da janela não esta na imagem
            #caso não esteja, arruma a posição do pixel mais a baixo para o limite da imagem (cols -1)
            if col > cols - i_wt -1: 
                #Pega o valor de quantos pixels foi perdido na janela
                tamw = col_baixo_dir - cols+1
                col_baixo_dir = cols -1
           
            for c in range (0,channel):
                flag1 = False
                flag2 = False
                menos_h = tamh
                menos_w = tamw   
                    
                #coloca o pixel diagonal direita baixo na somatória da imagem
                img_out[row,col,c] = img_buffer[row_baixo_dir][col_baixo_dir][c]

                #verifica se o topo -1 do kernel esta na imagem
                #caso estiver,o inclui na somatória
                if row > i_ht :
                    #adiciona na somatória o pixel direita cima 
                    img_out[row,col,c] -= img_buffer[row-i_ht-1][col_baixo_dir][c]
                    flag1= True
                else :
                    #ajusta o tamanho da divisão da média
                    menos_h += i_ht - row
                #verifica se a esqueda  -1 do kernel esta na imagem
                #caso estiver,o inclui na somatória
                if col > i_wt :
                    #adiciona na somatória o pixel esquerda baixo
                    img_out[row,col,c] -= img_buffer[row_baixo_dir][col-i_wt-1][c]
                    flag2 = True
                else: 
                    menos_w += i_wt -col
                
                #Adiciona um pixel no somatório para arrumar a área 
                if(flag1 and flag2):
                    img_out[row,col,c] += img_buffer[row-i_ht-1][col-i_wt-1][c]
                
                
                img_out [row,col,c] /= ((hT-menos_h)*(wT-menos_w))
               
                
    return img_out
   

#--------------------------------------------------------------------------------------------

#========================================================================================
def main ():

    # Abre a imagem em escala de colorida.
    img = cv2.imread (INPUT_IMAGE)
    
    if img is None:
        print ('Erro abrindo a imagem.\n')
        sys.exit ()

    #Converte para float de 32 bits
    img = img.astype (np.float32) / 255
    
    # Negativo da Imagem
    if NEGATIVO:
        img = 1 - img

    # Imagem Original
    cv2.imshow ('01 - Original_cinza', img)
    cv2.imwrite ('01 - Original_cinza.png', img*255)

    start_time = timeit.default_timer ()
    dimension = np.shape(img)
    #Blur na imagem . 
    #img_blur = blur(img,dimension)
    img_blur = blur_integral(img,dimension)
    #img_blur = blur_separavel(img,dimension)
    img_cv = cv2.blur(img, ksize=(TAMANHO_JANELAW, TAMANHO_JANELAH))

    print ('Tempo: %f' % (timeit.default_timer () - start_time))
    
    #Subtrai a imagem blur criada com a do cv2
    img_blur_m_cv = img_blur.copy()
    img_blur_m_cv = img_blur - img_cv

    #Normaliza a imagem, assim verificando se realmente zerou
    img_norm = img_blur.copy()  
    cv2.normalize(img_blur_m_cv,dst=img_norm,alpha=0,beta=255,norm_type=cv2.NORM_MINMAX)
    
    # Mostra os objetos encontrados.
    cv2.imshow ('02 - out', img_blur)
    cv2.imwrite ('02 - out.png', img_blur*255)
    cv2.imshow ('03 - Img Subtraida', img_blur_m_cv)
    cv2.imwrite ('03 - Img Subtraida.png', img_blur_m_cv*255)
    cv2.imshow ('04 - normalizada', img_norm)
    cv2.imwrite ('04 - normalizada.png',img_norm)
    cv2.imshow ('05 - cv', img_cv)
    cv2.imwrite ('05 - cv.png', img_cv)
    cv2.waitKey ()
    cv2.destroyAllWindows ()


if __name__ == '__main__':
    main ()

#===============================================================================
