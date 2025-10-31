import os
import cv2
import numpy as np
from carregar_imagens import carregar_imagens

def melhorar_imagens_pouca_luz(input_folder, output_folder, clip_limit=3.0, tile_size=(8, 8)):
    """
    Melhora imagens com pouca luz usando CLAHE combinado com correção de brilho.

    Args:
        input_folder (str): Pasta de entrada com as imagens originais.
        output_folder (str): Pasta de saída para salvar as imagens melhoradas.
        clip_limit (float): Controle de contraste local (maior = mais contraste).
        tile_size (tuple): Tamanho dos blocos do CLAHE.
    """
    os.makedirs(output_folder, exist_ok=True)

    # Carregar imagens e nomes
    low_de_imagens, nomes_imagens = carregar_imagens(input_folder)
    total = len(nomes_imagens)

    if total == 0:
        print("⚠️ Nenhuma imagem encontrada para processar.")
        return

    print(f"\n💡 Melhorando {total} imagens com pouca luz...")
    print(f"📁 Salvando resultados em: {output_folder}\n")

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)

    for idx, (img_bgr, nome) in enumerate(zip(low_de_imagens, nomes_imagens), start=1):
        try:
            # Converter para espaço de cor LAB (onde L = luminosidade)
            lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)

            # Aplicar CLAHE apenas no canal de luminosidade
            l_clahe = clahe.apply(l)

            # Recombinar e converter de volta para BGR
            lab_clahe = cv2.merge((l_clahe, a, b))
            img_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

            # Ajuste leve de brilho (melhora áreas muito escuras)
            img_final = cv2.convertScaleAbs(img_clahe, alpha=1.2, beta=15)

            # Salvar imagem resultante
            save_path = os.path.join(output_folder, nome)
            cv2.imwrite(save_path, img_final)
            print(f"[{idx}/{total}] ✅ {nome} melhorada e salva.")

        except Exception as e:
            print(f"[{idx}/{total}] ⚠️ Erro ao processar '{nome}': {e}")

    print("\n🎯 Processamento concluído com sucesso!")


if __name__ == "__main__":
    # Caminhos
    input_folder = r"C:\Users\jhona\Desktop\jhon\pdi\pipeline\main\LOLdataset\eval15\low"
    output_folder = r"C:\Users\jhona\Desktop\jhon\pdi\pipeline\main\imagem_filtradas"

    melhorar_imagens_pouca_luz(
        input_folder=input_folder,
        output_folder=output_folder,
        clip_limit=3.0,   # Aumentar para realçar mais o contraste
        tile_size=(8, 8)  # Tamanho padrão dos blocos CLAHE
    )
