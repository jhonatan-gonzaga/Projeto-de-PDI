import os
import cv2
import numpy as np


# ======== Cálculo dos coeficientes MSCN (base do BRISQUE simplificado) ========
def calculate_mscn_coefficients(image):
    c = np.fft.fftshift(np.fft.fft2(image))
    magnitude = np.abs(c)
    log_magnitude = np.log1p(magnitude)
    return np.real(c), np.imag(c), magnitude, log_magnitude


def calculate_mscn_features(image):
    _, _, _, log_magnitude = calculate_mscn_coefficients(image)
    return [np.std(log_magnitude), np.mean(log_magnitude)]


def brisque_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return calculate_mscn_features(gray)


def get_brisque_score(image):
    # Modelo linear simplificado baseado em pesos conhecidos
    features = brisque_features(image)
    weights = [
        -0.0977446, 0.0270277, 0.00090095, 0.0793246, 0.0476165, -0.033992,
        -0.0535509, 0.276186, 0.189205, 0.255546, 0.120626, 0.0471861,
        -0.18469, 0.154051, -0.173411, -0.413456
    ]
    intercept = 18.9217
    score = intercept + sum(f * w for f, w in zip(features, weights[:len(features)]))
    return score


# ======== Leitura das imagens (sem redimensionar) ========
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

    # Verificação
    if set(input_images.keys()) != set(output_images.keys()):
        raise ValueError("As pastas devem conter o mesmo conjunto de imagens com nomes correspondentes.")

    scores = []
    for key in input_images:
        enhanced = output_images[key]
        score = get_brisque_score(enhanced)
        scores.append(score)
        print(f"{key}: BRISQUE = {score:.4f}")

    print("\n===== RESULTADOS =====")
    print(f"Média BRISQUE: {np.mean(scores):.4f}")
    print(f"Desvio padrão: {np.std(scores):.4f}")
    print(f"Total de imagens: {len(scores)}")


# ======== Execução ========
if __name__ == "__main__":
    input_folder = r"C:\Users\jhona\Desktop\jhon\pdi\pipeline\main\LOLdataset\eval15\low"
    output_folder = r"C:\Users\jhona\Desktop\jhon\pdi\pipeline\main\imagem_filtradas"
    main(input_folder, output_folder)
