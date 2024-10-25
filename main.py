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
TAMANHO_JANELAH = 3
TAMANHO_JANELAW = 13

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
        for col in range (0,cols):
            for c in range (0,channel):
                if row < i_ht or col < i_wt or row > (rows - i_ht-1) or col > (cols -i_wt-1):
                    img_out[row][col][c] = 0
                    continue

                img_out [row,col,c] = img_buffer[row+i_ht][col+i_wt][c]  + img_buffer[row-i_ht-1][col-i_wt-1][c] - img_buffer[row+i_ht][col-i_wt-1][c] -img_buffer[row-i_ht-1][col+i_wt][c]
                img_out [row,col,c] /= (hT*wT)
            
            
               
            ##range é excludente, adicionar +1     
        
            
                
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
    img_blur = blur(img,dimension)
    #img_blur = blur_integral(img,dimension)

    img_cv = cv2.blur(img, ksize=(TAMANHO_JANELAW, TAMANHO_JANELAH))

    print ('Tempo: %f' % (timeit.default_timer () - start_time))
    
    #Subtrai a imagem blur criada com a do cv2
    img_blur_m_cv = img_blur.copy()
    img_blur_m_cv = img_blur - img_cv

    #Normaliza a imagem, assim verificando se realmente zerou
    img_norm = img_blur.copy()  
    cv2.normalize(img_blur,dst=img_norm,alpha=0,beta=255,norm_type=cv2.NORM_MINMAX)
    
    # Mostra os objetos encontrados.
    cv2.imshow ('02 - out', img_blur)
    cv2.imwrite ('02 - out.png', img_blur*255)
    cv2.imshow ('03 - Img Subtraida', img_blur_m_cv)
    cv2.imwrite ('03 - Img Subtraida.png', img_blur_m_cv*255)
    cv2.imshow ('04 - normalizada', img_norm)
    cv2.imwrite ('04 - normalizada.png',img_norm)
    cv2.imshow ('08 - cv', img_cv)
    cv2.imwrite ('08 - cv.png', img_cv)
    cv2.waitKey ()
    cv2.destroyAllWindows ()


if __name__ == '__main__':
    main ()

#===============================================================================
