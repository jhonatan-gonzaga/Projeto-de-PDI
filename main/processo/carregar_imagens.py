import os
import cv2
import numpy as np

def carregar_imagens(input_folder):
    lista_de_imagens_bgr = []
    nomes_imagens = []

    extensoes_validas = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')

    if not os.path.exists(input_folder):
        raise FileNotFoundError(f"Pasta '{input_folder}' não encontrada.")

    for nome_do_arquivo in sorted(os.listdir(input_folder)):
        if nome_do_arquivo.lower().endswith(extensoes_validas):
            caminho_completo = os.path.join(input_folder, nome_do_arquivo)
            imagem_bgr = cv2.imread(caminho_completo)
            if imagem_bgr is not None:
                lista_de_imagens_bgr.append(imagem_bgr)
                nomes_imagens.append(nome_do_arquivo)

    low_de_imagens = np.array(lista_de_imagens_bgr)
    return low_de_imagens, nomes_imagens
