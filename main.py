import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from otsu import find_optimal_threshold

def plot_histogram(hist, title, filename):
    """Plota e salva o histograma"""
    plt.figure()
    plt.title(title)
    plt.xlabel("Nível de Cinza")
    plt.ylabel("Frequência Normalizada")
    plt.plot(hist)
    plt.xlim([0, 256])
    plt.savefig(filename)
    plt.close()

def main():
    if not os.path.exists("opencv_comparacao"):
        os.makedirs("opencv_comparacao")

    output_dir = "results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    hist_bom = np.zeros(256)
    hist_bom[50:100] = np.linspace(0, 1, 50)
    hist_bom[100:150] = np.linspace(1, 0, 50)
    hist_bom[180:210] = np.linspace(0, 0.8, 30)
    hist_bom[210:240] = np.linspace(0.8, 0, 30)
    hist_bom /= hist_bom.sum()  
    plot_histogram(hist_bom, "Histograma Fictício (Bom)", "results/hist_bom.png")
    
    limiar_bom = find_optimal_threshold(hist_bom)
    print(f"Limiar ótimo -- Histograma Fictício (Bom): {limiar_bom}")

    hist_ruim = np.zeros(256)
    hist_ruim[80:130] = np.linspace(0, 1, 50)
    hist_ruim[130:180] = np.linspace(1, 0, 50)
    hist_ruim /= hist_ruim.sum() 
    plot_histogram(hist_ruim, "Histograma Fictício (Ruim)", "results/hist_ruim.png")
    
    limiar_ruim = find_optimal_threshold(hist_ruim)
    print(f"Limiar ótimo -- Histograma Fictício (Ruim): {limiar_ruim}")

    img_boa = cv2.imread("results/imagem_boa_original.png")
    if img_boa is None:
        print("Erro: Imagem 'imagem_boa_original.png' não encontrada.")
        return  
    
    img_boa_gray = cv2.cvtColor(img_boa, cv2.COLOR_BGR2GRAY)  

    _, binarizada_boa_opencv = cv2.threshold(img_boa_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imwrite("opencv_comparacao/imagem_boa_binarizada_opencv.png", binarizada_boa_opencv)
    
    print(f"Limiar ótimo -- OpenCV (Bom): {limiar_bom}")

    hist_boa = cv2.calcHist([img_boa_gray], [0], None, [256], [0, 256])
    hist_boa = cv2.normalize(hist_boa, hist_boa).flatten()
    plot_histogram(hist_boa, "Histograma Imagem Boa", "results/hist_boa.png")
    limiar_boa = find_optimal_threshold(hist_boa)
    print(f"Limiar ótimo -- Imagem Sintética com Boa Separação: {limiar_boa}")

    _, binarizada_boa = cv2.threshold(img_boa_gray, limiar_boa, 255, cv2.THRESH_BINARY)
    cv2.imwrite("results/imagem_boa_binarizada.png", binarizada_boa)

    print(f"Limiar ótimo -- OpenCV (Boa Separação): {limiar_boa}")

    img_ruim = cv2.imread("results/imagem_ruim_original.bmp", cv2.IMREAD_GRAYSCALE) 
    if img_ruim is None:
        print("Erro: Imagem 'imagem_ruim_original.bmp' não encontrada.")
        return  
    
    hist_ruim = cv2.calcHist([img_ruim], [0], None, [256], [0, 256])
    hist_ruim = cv2.normalize(hist_ruim, hist_ruim).flatten()
    plot_histogram(hist_ruim, "Histograma Imagem Texto Escrito", "results/hist_ruim.png")
    limiar_ruim = find_optimal_threshold(hist_ruim)
    print(f"Limiar ótimo -- Imagem Real (Texto Escrito): {limiar_ruim}")

    _, binarizada_ruim = cv2.threshold(img_ruim, limiar_ruim, 255, cv2.THRESH_BINARY)
    cv2.imwrite("results/imagem_ruim_binarizada.png", binarizada_ruim)  

    _, binarizada_ruim_opencv = cv2.threshold(img_ruim, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imwrite("opencv_comparacao/imagem_ruim_binarizada_opencv.png", binarizada_ruim_opencv)

if __name__ == "__main__":
    main()
