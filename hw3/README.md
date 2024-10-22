# Motion Estimation Implementation

A Python implementation of motion estimation and compensation algorithms.

## How to Run

1. **Install Requirements**
   ```bash
   pip install numpy opencv-python matplotlib
   ```

2. **Prepare Files**
   - Put `src.py` and two grayscale images in the same folder
   - Rename your images to:
     - `one_gray.png` (first frame)
     - `two_gray.png` (second frame)

3. **Run Program**
   ```bash
   python src.py
   ```

4. **Check Outputs**
   - `reconstructed.png`: Reconstructed frame
   - `residual.png`: Residual frame
   - Console will show PSNR and comparison results

## Requirements
- Python 3.6+
- numpy
- opencv-python
- matplotlib