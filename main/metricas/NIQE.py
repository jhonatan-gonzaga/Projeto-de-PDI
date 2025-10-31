import os
import cv2
import numpy as np
from skimage import color


# ======== Cálculo NIQE simplificado ========
def get_niqe(image):
    """
    Calcula um score NIQE simplificado baseado em contraste local.
    Quanto menor o valor, melhor a qualidade perceptual.
    """
    gray = color.rgb2gray(image)
    grad = np.abs(np.gradient(gray))
    local_contrast = np.abs(grad)

    mean_contrast = np.mean(local_contrast)
    std_contrast = np.std(local_contrast)

    # Fórmula empírica de qualidade perceptual
    niqe_score = 1.0 / (1 + 6.6 * mean_contrast + 0.228 * std_contrast)
    return niqe_score


# ======== Leitura de imagens (sem redimensionar) ========
def read_images_from_folder(folder_path):
    images = {}
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            path = os.path.join(folder_path, filename)
            img = cv2.imread(path)
            if img is not None:
                key = '_'.join(filename.split('_')[:2])
                images[key] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # para compatibilidade com skimage
    return images


# ======== Função principal ========
def main(input_folder, output_folder):
    input_images = read_images_from_folder(input_folder)
    output_images = read_images_from_folder(output_folder)

    if set(input_images.keys()) != set(output_images.keys()):
        raise ValueError("As pastas devem conter o mesmo conjunto de imagens com nomes correspondentes.")

    scores = {}
    niqe_values = []

    for key in sorted(input_images.keys()):
        img_out = output_images[key]
        niqe_score = get_niqe(img_out)
        scores[key] = niqe_score
        niqe_values.append(niqe_score)
        print(f"{key}: NIQE = {niqe_score:.4f}")

    niqe_values = np.array(niqe_values)
    print("\n===== RESULTADOS =====")
    print(f"Média NIQE: {niqe_values.mean():.4f}")
    print(f"Desvio padrão: {niqe_values.std():.4f}")
    print(f"Total de imagens: {len(niqe_values)}")


# ======== Execução ========
if __name__ == "__main__":
    input_folder = r"C:\Users\jhona\Desktop\jhon\pdi\pipeline\main\LOLdataset\eval15\low"
    output_folder = r"C:\Users\jhona\Desktop\jhon\pdi\pipeline\main\imagem_filtradas"
    main(input_folder, output_folder)
