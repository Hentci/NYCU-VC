import numpy as np
import cv2
import time
from typing import Tuple, List
import matplotlib.pyplot as plt

class MotionEstimation:
    def __init__(self, block_size: int = 8, search_range: int = 8):
        self.block_size = block_size
        self.search_range = search_range
        
    def mad(self, block1: np.ndarray, block2: np.ndarray) -> float:
        """計算平均絕對差異(Mean Absolute Difference)"""
        return np.mean(np.abs(block1 - block2))
    
    def full_search(self, curr_frame: np.ndarray, ref_frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """全搜尋演算法"""
        height, width = curr_frame.shape
        mv_x = np.zeros((height//self.block_size, width//self.block_size))
        mv_y = np.zeros((height//self.block_size, width//self.block_size))
        min_mad_values = np.zeros((height//self.block_size, width//self.block_size))
        
        start_time = time.time()
        
        for i in range(0, height-self.block_size+1, self.block_size):
            for j in range(0, width-self.block_size+1, self.block_size):
                curr_block = curr_frame[i:i+self.block_size, j:j+self.block_size]
                min_mad = float('inf')
                best_dx, best_dy = 0, 0
                
                # 在搜尋範圍內尋找最佳匹配
                for dy in range(-self.search_range, self.search_range+1):
                    for dx in range(-self.search_range, self.search_range+1):
                        ref_i = i + dy
                        ref_j = j + dx
                        
                        # 檢查邊界
                        if (ref_i < 0 or ref_i + self.block_size > height or
                            ref_j < 0 or ref_j + self.block_size > width):
                            continue
                        
                        ref_block = ref_frame[ref_i:ref_i+self.block_size, 
                                            ref_j:ref_j+self.block_size]
                        mad = self.mad(curr_block, ref_block)
                        
                        if mad < min_mad:
                            min_mad = mad
                            best_dx, best_dy = dx, dy
                
                mv_x[i//self.block_size, j//self.block_size] = best_dx
                mv_y[i//self.block_size, j//self.block_size] = best_dy
                min_mad_values[i//self.block_size, j//self.block_size] = min_mad
        
        processing_time = time.time() - start_time
        return mv_x, mv_y, processing_time
    
    def three_step_search(self, curr_frame: np.ndarray, ref_frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """三步搜尋演算法"""
        height, width = curr_frame.shape
        mv_x = np.zeros((height//self.block_size, width//self.block_size))
        mv_y = np.zeros((height//self.block_size, width//self.block_size))
        
        start_time = time.time()
        
        for i in range(0, height-self.block_size+1, self.block_size):
            for j in range(0, width-self.block_size+1, self.block_size):
                curr_block = curr_frame[i:i+self.block_size, j:j+self.block_size]
                
                # 初始步長為搜尋範圍的一半
                step_size = self.search_range // 2
                best_dx, best_dy = 0, 0
                min_mad = float('inf')
                
                while step_size >= 1:
                    # 搜尋九個點
                    for dy in [-step_size, 0, step_size]:
                        for dx in [-step_size, 0, step_size]:
                            new_dy = best_dy + dy
                            new_dx = best_dx + dx
                            
                            # 檢查邊界
                            ref_i = i + new_dy
                            ref_j = j + new_dx
                            if (ref_i < 0 or ref_i + self.block_size > height or
                                ref_j < 0 or ref_j + self.block_size > width):
                                continue
                            
                            ref_block = ref_frame[ref_i:ref_i+self.block_size,
                                                ref_j:ref_j+self.block_size]
                            mad = self.mad(curr_block, ref_block)
                            
                            if mad < min_mad:
                                min_mad = mad
                                best_dx, best_dy = new_dx, new_dy
                    
                    step_size //= 2
                
                mv_x[i//self.block_size, j//self.block_size] = best_dx
                mv_y[i//self.block_size, j//self.block_size] = best_dy
        
        processing_time = time.time() - start_time
        return mv_x, mv_y, processing_time

    def motion_compensation(self, curr_frame: np.ndarray, ref_frame: np.ndarray, 
                          mv_x: np.ndarray, mv_y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """執行動態補償"""
        height, width = curr_frame.shape
        reconstructed = np.zeros_like(curr_frame)
        
        for i in range(0, height-self.block_size+1, self.block_size):
            for j in range(0, width-self.block_size+1, self.block_size):
                block_i = i // self.block_size
                block_j = j // self.block_size
                
                dx = int(mv_x[block_i, block_j])
                dy = int(mv_y[block_i, block_j])
                
                ref_i = i + dy
                ref_j = j + dx
                
                if (ref_i >= 0 and ref_i + self.block_size <= height and
                    ref_j >= 0 and ref_j + self.block_size <= width):
                    reconstructed[i:i+self.block_size, j:j+self.block_size] = \
                        ref_frame[ref_i:ref_i+self.block_size, ref_j:ref_j+self.block_size]
        
        residual = curr_frame - reconstructed
        return reconstructed, residual

def calculate_psnr(original: np.ndarray, compressed: np.ndarray) -> float:
    """計算PSNR值"""
    mse = np.mean((original - compressed) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def compare_search_ranges(curr_frame: np.ndarray, ref_frame: np.ndarray) -> None:
    """比較不同搜尋範圍的結果"""
    search_ranges = [8, 16, 32]
    results = []
    
    for sr in search_ranges:
        me = MotionEstimation(search_range=sr)
        mv_x, mv_y, time_taken = me.full_search(curr_frame, ref_frame)
        reconstructed, _ = me.motion_compensation(curr_frame, ref_frame, mv_x, mv_y)
        psnr = calculate_psnr(curr_frame, reconstructed)
        results.append((sr, psnr, time_taken))
    
    print("\nSearch Range Comparison Results:")
    print("--------------------------------")
    print("Search Range | PSNR (dB) | Runtime (s)")
    print("--------------------------------")
    for sr, psnr, time_taken in results:
        print(f"    ±{sr:<7} | {psnr:8.2f} | {time_taken:10.3f}")

def compare_algorithms(curr_frame: np.ndarray, ref_frame: np.ndarray) -> None:
    """比較全搜尋和三步搜尋演算法"""
    me = MotionEstimation()
    
    # 全搜尋
    mv_x_fs, mv_y_fs, time_fs = me.full_search(curr_frame, ref_frame)
    reconstructed_fs, _ = me.motion_compensation(curr_frame, ref_frame, mv_x_fs, mv_y_fs)
    psnr_fs = calculate_psnr(curr_frame, reconstructed_fs)
    
    # 三步搜尋
    mv_x_tss, mv_y_tss, time_tss = me.three_step_search(curr_frame, ref_frame)
    reconstructed_tss, _ = me.motion_compensation(curr_frame, ref_frame, mv_x_tss, mv_y_tss)
    psnr_tss = calculate_psnr(curr_frame, reconstructed_tss)
    
    print("\nAlgorithm Comparison Results:")
    print("--------------------------------")
    print("Algorithm    | PSNR (dB) | Runtime (s)")
    print("--------------------------------")
    print(f"Full Search  | {psnr_fs:8.2f} | {time_fs:10.3f}")
    print(f"Three-Step   | {psnr_tss:8.2f} | {time_tss:10.3f}")

def main():
    # 讀取圖片
    frame1 = cv2.imread('one_gray.png', cv2.IMREAD_GRAYSCALE)
    frame2 = cv2.imread('two_gray.png', cv2.IMREAD_GRAYSCALE)
    
    if frame1 is None or frame2 is None:
        print("Error: Unable to read input images")
        return
    
    # 初始化動態估計物件
    me = MotionEstimation()
    
    # 執行動態估計
    mv_x, mv_y, _ = me.full_search(frame2, frame1)
    
    # 執行動態補償
    reconstructed, residual = me.motion_compensation(frame2, frame1, mv_x, mv_y)
    
    # 儲存結果
    cv2.imwrite('reconstructed.png', reconstructed)
    cv2.imwrite('residual.png', residual)
    
    # 計算PSNR
    psnr = calculate_psnr(frame2, reconstructed)
    print(f"PSNR of reconstructed frame: {psnr:.2f} dB")
    
    # 比較不同搜尋範圍
    compare_search_ranges(frame2, frame1)
    
    # 比較不同演算法
    compare_algorithms(frame2, frame1)

if __name__ == "__main__":
    main()