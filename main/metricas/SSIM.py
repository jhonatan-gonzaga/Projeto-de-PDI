import os
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

# ======== Cálculo SSIM ========
def calculate_ssim(image1, image2, win_size=3):
    if image1 is None or image2 is None:
        raise ValueError("Uma ou ambas as imagens não puderam ser carregadas.")
    
    # Converter BGR para RGB
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
    
    # Verificar se as dimensões coincidem
    if image1.shape != image2.shape:
        raise ValueError("As imagens devem ter as mesmas dimensões.")
    
    # Calcular SSIM multicanal
    return ssim(image1, image2, multichannel=True, win_size=win_size)


# ======== Leitura de imagens sem redimensionamento ========
def read_images_from_folder(folder_path):
    images = {}
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            path = os.path.join(folder_path, filename)
            img = cv2.imread(path)
            if img is not None:
                key = '_'.join(filename.split('_')[:2])
                images[key] = img
    return images


# ======== Função principal ========
def main(input_folder, output_folder):
    input_images = read_images_from_folder(input_folder)
    output_images = read_images_from_folder(output_folder)

    # Checagem de correspondência de chaves
    if set(input_images.keys()) != set(output_images.keys()):
        raise ValueError("As pastas devem conter o mesmo conjunto de imagens com nomes correspondentes.")

    ssim_scores = []

    for key in sorted(input_images.keys()):
        img_in = input_images[key]
        img_out = output_images[key]
        score = calculate_ssim(img_in, img_out)
        ssim_scores.append(score)
        print(f"{key}: SSIM = {score:.4f}")

    ssim_array = np.array(ssim_scores)
    print("\n===== RESULTADOS =====")
    print(f"Média SSIM: {ssim_array.mean():.4f}")
    print(f"Desvio padrão: {ssim_array.std():.4f}")
    print(f"Total de imagens: {len(ssim_array)}")


# ======== Execução ========
if __name__ == "__main__":
    input_folder = r"C:\Users\jhona\Desktop\jhon\pdi\pipeline\main\LOLdataset\eval15\low"
    output_folders = [r"C:\Users\jhona\Desktop\jhon\pdi\pipeline\main\imagem_filtradas"]

    for folder in output_folders:
        print(f"==== Avaliando pasta: {folder} ====")
        main(input_folder, folder)