# **Video Compression Hw**#3 – Motion Estimation & Compensation

#### 313551055 柯柏旭

## Introduction

### 實驗目的

實作並比較不同動態估計演算法的效能，包含：
1. 不同搜尋範圍的影響
2. 全搜尋與三步搜尋演算法的比較

### 實驗設定

- 區塊大小：8×8
- 搜尋範圍：±8, ±16, ±32
- 評估指標：PSNR 和執行時間

## Experimental Results

### 1.  重建結果

- Reconstructed frame

![image-20241022164842933](/Users/hentci/Library/Application Support/typora-user-images/image-20241022164842933.png)

- Residual

![image-20241022164854632](/Users/hentci/Library/Application Support/typora-user-images/image-20241022164854632.png)

Residual frame PSNR：29.27 dB

### 2. 搜尋範圍比較

| 搜尋範圍 | PSNR (dB) | 執行時間 (s) |
| -------- | --------- | ------------ |
| ±8       | 29.27     | 1.972        |
| ±16      | 29.33     | 7.433        |
| ±32      | 29.26     | 27.773       |

觀察：
- PSNR差異不大（約0.07 dB）
- 執行時間隨搜尋範圍增加而顯著增加
- ±16提供最好的PSNR，但執行時間增加約4倍
- ±32的執行時間增加顯著（約14倍），但PSNR反而略低

### 3. 演算法比較

| 演算法      | PSNR (dB) | 執行時間 (s) |
| ----------- | --------- | ------------ |
| Full Search | 29.27     | 2.117        |
| Three-Step  | 28.60     | 0.206        |

觀察：
- 三步搜尋比全搜尋快約10倍
- PSNR降低約0.67 dB
- 在效能與品質間提供了良好的平衡

## Discussion

1. 搜尋範圍：
   - ±8提供了最佳的效能/品質平衡
   - 擴大搜尋範圍並未帶來明顯的品質提升

2. 搜尋演算法：
   - 三步搜尋大幅降低計算時間
   - 品質降低幅度可接受
   - 適合即時處理應用