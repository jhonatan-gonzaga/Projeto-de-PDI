import os
import cv2
import numpy as np


# ======== Cálculo do PSNR ========
def calculate_psnr(imageA, imageB):
    """Calcula o PSNR (Peak Signal-to-Noise Ratio) entre duas imagens."""
    if imageA.shape != imageB.shape:
        raise ValueError("As imagens devem ter as mesmas dimensões para calcular o PSNR.")
    return cv2.PSNR(imageA, imageB)


# ======== Leitura de imagens (sem redimensionar) ========
def read_images_from_folder(folder_path):
    """Lê todas as imagens da pasta e as armazena em um dicionário."""
    images = {}
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            path = os.path.join(folder_path, filename)
            img = cv2.imread(path)
            if img is not None:
                key = '_'.join(filename.split('_')[:2])  # identifica o par pelo prefixo
                images[key] = img
    return images


# ======== Função principal ========
def main(input_folder, output_folder):
    print(f"\n📁 Lendo imagens de entrada: {input_folder}")
    print(f"📁 Lendo imagens de saída:  {output_folder}\n")

    input_images = read_images_from_folder(input_folder)
    output_images = read_images_from_folder(output_folder)

    if set(input_images.keys()) != set(output_images.keys()):
        raise ValueError("As pastas devem conter o mesmo conjunto de imagens com nomes correspondentes.")

    psnr_values = {}

    for key in sorted(input_images.keys()):
        imgA = input_images[key]
        imgB = output_images[key]

        psnr_value = calculate_psnr(imgA, imgB)
        psnr_values[key] = psnr_value
        print(f"{key}: PSNR = {psnr_value:.2f} dB")

    # Estatísticas
    values = np.array(list(psnr_values.values()))
    mean_psnr = np.mean(values)
    std_psnr = np.std(values)

    print("\n===== RESULTADOS =====")
    print(f"Média PSNR: {mean_psnr:.2f} dB")
    print(f"Desvio padrão: {std_psnr:.2f} dB")
    print(f"Total de imagens: {len(values)}")
    print("======================\n")


# ======== Execução local ========
if __name__ == "__main__":
    # Caminhos locais (ajuste conforme o seu PC)
    input_folder = r"C:\Users\jhona\Desktop\jhon\pdi\pipeline\main\LOLdataset\eval15\low"
    output_folder = r"C:\Users\jhona\Desktop\jhon\pdi\pipeline\main\imagem_filtradas"

    main(input_folder, output_folder)
