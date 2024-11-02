import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.fftpack import dct, idct

def read_image(path):
    """Read image in grayscale and return as float32"""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return np.float32(img)

def split_into_blocks(img, block_size=8):
    """Split image into 8x8 blocks"""
    h, w = img.shape
    blocks = []
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = img[i:i+block_size, j:j+block_size]
            blocks.append(block)
    return blocks

def zigzag_scan(block):
    """Perform zigzag scan on 8x8 block"""
    rows, cols = block.shape
    solution=[[] for i in range(rows+cols-1)]
    
    for i in range(rows):
        for j in range(cols):
            sum=i+j
            if(sum%2==0):
                solution[sum].insert(0,block[i][j])
            else:
                solution[sum].append(block[i][j])
                
    return [x for sublist in solution for x in sublist]

def inverse_zigzag(zigzag, block_size=8):
    """Convert zigzag sequence back to block"""
    block = np.zeros((block_size, block_size))
    index = 0
    
    for i in range(2 * block_size - 1):
        if i < block_size:
            bound = 0 if i < block_size else i - block_size + 1
            for j in range(bound, min(i + 1, block_size)):
                if i % 2 == 1:
                    block[j][i-j] = zigzag[index]
                else:
                    block[i-j][j] = zigzag[index]
                index += 1
        else:
            bound = i - block_size + 1
            for j in range(bound, min(block_size, i + 1)):
                if i % 2 == 1:
                    block[j][i-j] = zigzag[index]
                else:
                    block[i-j][j] = zigzag[index]
                index += 1
    return block

def run_length_encode(sequence):
    """Perform run length encoding"""
    encoding = []
    prev_val = sequence[0]
    count = 1
    
    for value in sequence[1:]:
        if value == prev_val:
            count += 1
        else:
            encoding.append((prev_val, count))
            prev_val = value
            count = 1
            
    encoding.append((prev_val, count))
    return encoding

def run_length_decode(encoding):
    """Decode run length encoding"""
    sequence = []
    for value, count in encoding:
        sequence.extend([value] * count)
    return sequence

def quantize_block(block, qtable):
    """Quantize DCT coefficients"""
    return np.round(block / qtable)

def dequantize_block(block, qtable):
    """Dequantize DCT coefficients"""
    return block * qtable

def process_image(img_path, qtable):
    """Process image with DCT, quantization and RLE"""
    # Read image
    img = read_image(img_path)
    h, w = img.shape
    
    # Split into blocks
    blocks = split_into_blocks(img)
    
    # Process each block
    encoded_data = []
    for block in blocks:
        # Apply DCT
        dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')
        
        # Quantize
        quantized = quantize_block(dct_block, qtable)
        
        # Zigzag scan
        zigzag = zigzag_scan(quantized)
        
        # Run length encoding
        rle = run_length_encode(zigzag)
        encoded_data.append(rle)
    
    # Calculate encoded size (in bytes)
    encoded_size = sum(len(rle) * 8 for rle in encoded_data)  # Assuming each value-count pair takes 8 bytes
    
    # Reconstruct image
    reconstructed = np.zeros_like(img)
    block_idx = 0
    
    for i in range(0, h, 8):
        for j in range(0, w, 8):
            # Decode current block
            rle = encoded_data[block_idx]
            zigzag = run_length_decode(rle)
            block = inverse_zigzag(zigzag)
            
            # Dequantize
            dequantized = dequantize_block(block, qtable)
            
            # Inverse DCT
            reconstructed_block = idct(idct(dequantized.T, norm='ortho').T, norm='ortho')
            
            # Place block back in image
            reconstructed[i:i+8, j:j+8] = reconstructed_block
            block_idx += 1
    
    return reconstructed, encoded_size

def visualize_results(original, reconstructed1, reconstructed2, encoded_size1, encoded_size2):
    """Visualize original and reconstructed images"""
    plt.figure(figsize=(15, 5))
    
    plt.subplot(131)
    plt.imshow(original, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(132)
    plt.imshow(reconstructed1, cmap='gray')
    plt.title(f'Reconstructed (Q1)\nSize: {encoded_size1/1024:.2f} KB')
    plt.axis('off')
    
    plt.subplot(133)
    plt.imshow(reconstructed2, cmap='gray')
    plt.title(f'Reconstructed (Q2)\nSize: {encoded_size2/1024:.2f} KB')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('compression_results.png')
    plt.close()

# Define quantization tables from the image
Q1 = np.array([
    [10, 7, 6, 10, 14, 24, 31, 37],
    [7, 7, 8, 11, 16, 35, 36, 33],
    [8, 8, 10, 14, 24, 34, 41, 34],
    [8, 10, 13, 17, 31, 52, 48, 37],
    [11, 13, 22, 34, 41, 65, 62, 46],
    [14, 21, 33, 38, 49, 62, 68, 55],
    [29, 38, 47, 52, 62, 73, 72, 61],
    [43, 55, 57, 59, 67, 60, 62, 59]
])

Q2 = np.array([
    [10, 11, 14, 28, 59, 59, 59, 59],
    [11, 13, 16, 40, 59, 59, 59, 59],
    [14, 16, 34, 59, 59, 59, 59, 59],
    [28, 40, 59, 59, 59, 59, 59, 59],
    [59, 59, 59, 59, 59, 59, 59, 59],
    [59, 59, 59, 59, 59, 59, 59, 59],
    [59, 59, 59, 59, 59, 59, 59, 59],
    [59, 59, 59, 59, 59, 59, 59, 59]
])

def calculate_psnr(original, compressed):
    """Calculate PSNR between original and compressed images"""
    mse = np.mean((original - compressed) ** 2)
    if mse == 0:
        return float('inf')
    PIXEL_MAX = 255.0
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))

def main():
    # Process image with both quantization tables
    img_path = "lena.png"
    original = read_image(img_path)
    
    reconstructed1, size1 = process_image(img_path, Q1)
    reconstructed2, size2 = process_image(img_path, Q2)
    
    # Visualize results
    visualize_results(original, reconstructed1, reconstructed2, size1, size2)
    
    # Print compression results
    original_size = original.size * original.itemsize
    print(f"Original image size: {original_size/1024:.2f} KB")
    print(f"Compressed size (Q1): {size1/1024:.2f} KB")
    print(f"Compressed size (Q2): {size2/1024:.2f} KB")
    print(f"Compression ratio (Q1): {original_size/size1:.2f}:1")
    print(f"Compression ratio (Q2): {original_size/size2:.2f}:1")
    
    # 計算 PSNR
    psnr1 = calculate_psnr(original, reconstructed1)
    psnr2 = calculate_psnr(original, reconstructed2)
    
    print("\nQuality Metrics:")
    print(f"PSNR (Q1): {psnr1:.2f} dB")
    print(f"PSNR (Q2): {psnr2:.2f} dB")

if __name__ == "__main__":
    main()