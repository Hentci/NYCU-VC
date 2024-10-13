import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import time

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def dct2(block):
    M, N = block.shape
    a = np.arange(M)
    b = np.arange(N)
    
    alpha_m = np.sqrt(np.where(a == 0, 1/M, 2/M))
    alpha_n = np.sqrt(np.where(b == 0, 1/N, 2/N))
    
    factor = np.outer(alpha_m, alpha_n)
    
    cos_m = np.cos(np.outer(a, 2*np.arange(M)+1) * np.pi / (2*M))
    cos_n = np.cos(np.outer(b, 2*np.arange(N)+1) * np.pi / (2*N))
    
    return factor * np.dot(np.dot(cos_m.T, block), cos_n)

def idct2(block):
    M, N = block.shape
    a = np.arange(M)
    b = np.arange(N)
    
    alpha_m = np.sqrt(np.where(a == 0, 1/M, 2/M))
    alpha_n = np.sqrt(np.where(b == 0, 1/N, 2/N))
    
    factor = np.outer(alpha_m, alpha_n)
    
    cos_m = np.cos(np.outer(2*np.arange(M)+1, a) * np.pi / (2*M))
    cos_n = np.cos(np.outer(2*np.arange(N)+1, b) * np.pi / (2*N))
    
    return np.dot(np.dot(cos_m, factor * block), cos_n.T)

def dct1(vector):
    N = len(vector)
    k = np.arange(N)
    n = k.reshape((N, 1))
    cos_factors = np.cos(np.pi * (2*n + 1) * k / (2*N))
    alpha = np.sqrt(np.where(k == 0, 1/N, 2/N))
    return alpha * np.sum(vector.reshape((N, 1)) * cos_factors, axis=0)

def two_1d_dct(block):
    return np.apply_along_axis(dct1, 0, np.apply_along_axis(dct1, 1, block))

def psnr(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def visualize_and_save(images, titles, filename):
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=(6*n, 5))
    fig.suptitle('DCT Comparison', fontsize=16)
    
    for i, (img, title, ax) in enumerate(zip(images, titles, axes)):
        if 'DCT Coefficients' in title or 'Difference' in title:
            im = ax.imshow(np.log(np.abs(img) + 1), cmap='gray')
        else:
            im = ax.imshow(img, cmap='gray')
        ax.set_title(title, fontsize=12)
        ax.axis('off')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Load and convert image to grayscale
    img = Image.open("lena.png")
    img_gray = rgb2gray(np.array(img))

    print("Starting 2D-DCT...")
    start_time = time.time()
    dct_coeffs = dct2(img_gray)
    dct_time = time.time() - start_time
    print(f"2D-DCT completed in {dct_time:.4f} seconds")

    print("Starting 2D-IDCT...")
    start_time = time.time()
    reconstructed = idct2(dct_coeffs)
    idct_time = time.time() - start_time
    print(f"2D-IDCT completed in {idct_time:.4f} seconds")

    # Evaluate PSNR
    psnr_value = psnr(img_gray, reconstructed)
    print(f"PSNR: {psnr_value:.2f} dB")

    print("Starting Two 1D-DCT...")
    start_time = time.time()
    dct_coeffs_1d = two_1d_dct(img_gray)
    two_1d_time = time.time() - start_time
    print(f"Two 1D-DCT completed in {two_1d_time:.4f} seconds")

    # Compare runtime
    print(f"2D-DCT runtime: {dct_time:.4f} seconds")
    print(f"Two 1D-DCT runtime: {two_1d_time:.4f} seconds")
    print(f"Speed-up factor: {dct_time / two_1d_time:.2f}")

    # Visualize all results
    visualize_and_save(
        [img_gray, dct_coeffs, reconstructed],
        ['Original Image', '2D-DCT Coefficients', 'Reconstructed Image'],
        'dct_2d_results.png'
    )

    visualize_and_save(
        [img_gray, dct_coeffs_1d],
        ['Original Image', 'Two 1D-DCT Coefficients'],
        'dct_1d_results.png'
    )

    # Visualize the difference between 2D-DCT and Two 1D-DCT
    visualize_and_save(
        [dct_coeffs, dct_coeffs_1d, np.abs(dct_coeffs - dct_coeffs_1d)],
        ['2D-DCT Coefficients', 'Two 1D-DCT Coefficients', 'Absolute Difference'],
        'dct_comparison.png'
    )

if __name__ == "__main__":
    main()