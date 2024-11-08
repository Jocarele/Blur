#===============================================================================
# Exemplo: BLOOM 
#-------------------------------------------------------------------------------
# Autor: João Lucas M. Camilo
#        Viviane Ruotolo
# Universidade Tecnológica Federal do Paraná
#===============================================================================

import numpy as np
import cv2

INPUT_IMAGE = "Wind Waker GC.bmp"
#INPUT_IMAGE = "GT2.BMP"
LIMITE = 0.5
alpha = 1.09
betha = 0.08

img = cv2.imread(INPUT_IMAGE)
img = img.astype(np.float32)/255


img_limite = img.copy()

rows,cols,c = np.shape(img)
img_limite = cv2.cvtColor(img,cv2.COLOR_BGR2HLS)
img_limite[:,:,1] =np.where(img_limite[:,:,1] > LIMITE,img_limite[:,:,1],0)
img_limite = cv2.cvtColor(img_limite,cv2.COLOR_HLS2BGR)



img_blur = np.zeros((rows,cols,c))
sigma = 1
img_bloom_gaus = np.zeros((rows,cols,c))


while sigma <9:
    img_blur_gaus = cv2.GaussianBlur(img_limite, (0,0), sigma)
    flag =0
    if sigma == 1:
        img_blur_buffer = cv2.blur(img_limite,ksize=(3,3))

    elif sigma == 2:
        img_blur_buffer = cv2.blur(img_limite,ksize=(5,5))
        for i in range(0,1):
            img_blur_buffer = cv2.blur(img_blur_buffer,ksize=(5,5))
           
       
    elif sigma == 4:
        img_blur_buffer = cv2.blur(img_limite,ksize=(7,7))
        for i in range(0,2):
            img_blur_buffer = cv2.blur(img_blur_buffer,ksize=(7,7)) 
           
    else:
        img_blur_buffer = cv2.blur(img_limite,ksize=(13,13))
        for i in range(0,4):
            img_blur_buffer = cv2.blur(img_blur_buffer,ksize=(13,13))
            
        
    
    
    
    cv2.imshow("boxblur",img_blur_buffer)
    cv2.imwrite(f"BoxBLur_{sigma}.png",img_blur_buffer*255)    
    img_blur += img_blur_buffer 
    img_bloom_gaus += img_blur_gaus 
    cv2.imshow(f"02.1 - blur_{sigma}_gaus", img_blur_gaus)
    cv2.imwrite(f"02.1 - blur_{sigma}_gaus.png", img_blur_gaus*255)
    sigma*=2


img_bloom = alpha*img + betha*img_blur
img_bloom_gaus = alpha*img + betha * img_bloom_gaus

cv2.imshow("00 - original",img)
cv2.imwrite("00 - Orignial.png", img*255)
cv2.imshow("01 - blight pass", img_limite)
cv2.imwrite("01 - blight pass.png", img_limite*255)

cv2.imshow("03 - bloom", img_bloom)
cv2.imwrite("03 - bloom.png",img_bloom*255)
cv2.imshow("03 - bloom_gauss", img_bloom_gaus)
cv2.imwrite("03 - bloom_gauss.png",img_bloom_gaus*255)
cv2.waitKey()
