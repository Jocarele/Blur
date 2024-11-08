import sys
import timeit
import numpy as np
import cv2

# Parâmetros ajustáveis
INPUT_IMAGE = 'Original.bmp'
NEGATIVO = False
TAMANHO_JANELAH = 3
TAMANHO_JANELAW = 13

def integral(img, rows, cols, channel):
    img_out = np.zeros_like(img)
    for row in range(rows):
        for col in range(cols):
            for c in range(channel):
                if row == 0 and col == 0:
                    img_out[row, col, c] = img[row, col, c]
                elif row == 0:
                    img_out[row, col, c] = img_out[row, col - 1, c] + img[row, col, c]
                elif col == 0:
                    img_out[row, col, c] = img_out[row - 1, col, c] + img[row, col, c]
                else:
                    img_out[row, col, c] = (
                        img_out[row, col - 1, c]
                        + img_out[row - 1, col, c]
                        - img_out[row - 1, col - 1, c]
                        + img[row, col, c]
                    )
    return img_out

def blur_integral(img, dimension): 
    rows, cols, channel = dimension
    img_buffer = integral(img, rows, cols, channel)
    img_out = np.zeros_like(img)

    hT, wT = TAMANHO_JANELAH, TAMANHO_JANELAW
    i_ht, i_wt = int(hT / 2), int(wT / 2)

    for row in range(rows):
        for col in range(cols):
            for c in range(channel):
                # Definir limites da janela
                row_ini = max(0, row - i_ht)
                row_end = min(rows - 1, row + i_ht)
                col_ini = max(0, col - i_wt)
                col_end = min(cols - 1, col + i_wt)

                # Calcular a área da janela
                area = (row_end - row_ini + 1) * (col_end - col_ini + 1)

                # Somar os valores dentro da janela
                sum_value = img_buffer[row_end, col_end, c]
                if row_ini > 0:
                    sum_value -= img_buffer[row_ini - 1, col_end, c]
                if col_ini > 0:
                    sum_value -= img_buffer[row_end, col_ini - 1, c]
                if row_ini > 0 and col_ini > 0:
                    sum_value += img_buffer[row_ini - 1, col_ini - 1, c]

                # Calcular média e atribuir ao pixel de saída
                img_out[row, col, c] = sum_value / area
                # Diagnóstico: verificar somatório e média em algumas posições
                if (row == 0 and col == 0) or (row == rows - 1 and col == cols - 1):
                    print(f"Pixel ({row},{col},{c}): Suma={sum_value}, Área={area}, Média={img_out[row, col, c]}")

    return img_out

def main():
    img = cv2.imread(INPUT_IMAGE)
    if img is None:
        print('Erro abrindo a imagem.\n')
        sys.exit()

    img = img.astype(np.float32) / 255
    if NEGATIVO:
        img = 1 - img

    cv2.imshow('01 - Original', img)
    cv2.imwrite('01 - Original.png', img * 255)

    start_time = timeit.default_timer()
    dimension = np.shape(img)

    # Aplica blur integral
    img_blur = blur_integral(img, dimension)
    img_cv = cv2.blur(img, ksize=(TAMANHO_JANELAW, TAMANHO_JANELAH))

    print('Tempo: %f' % (timeit.default_timer() - start_time))

    # Subtrai e normaliza para ver diferenças absolutas
    img_diff = cv2.absdiff(img_blur, img_cv)
    img_norm = cv2.normalize(img_diff, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    # Exibe as imagens
    cv2.imshow('02 - Out', img_blur)
    cv2.imwrite('02 - Out.png', img_blur * 255)
    cv2.imshow('03 - Img Subtraida', img_diff)
    cv2.imwrite('03 - Img Subtraida.png', img_diff * 255)
    cv2.imshow('04 - Normalizada', img_norm / 255)
    cv2.imwrite('04 - Normalizada.png', img_norm)
    cv2.imshow('05 - CV Blur', img_cv)
    cv2.imwrite('05 - CV Blur.png', img_cv * 255)

    cv2.waitKey()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
