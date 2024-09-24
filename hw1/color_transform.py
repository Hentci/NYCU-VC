import numpy as np
from PIL import Image

# 讀取圖像
image_path = './lena.png'
image = Image.open(image_path)
image = image.convert('RGB')  # 確保是RGB圖像
rgb_array = np.array(image)

# 分離R、G、B通道
R = rgb_array[:, :, 0]
G = rgb_array[:, :, 1]
B = rgb_array[:, :, 2]

# RGB轉YUV
Y = 0.299 * R + 0.587 * G + 0.114 * B
U = -0.169 * R - 0.331 * G + 0.5 * B + 128
V = 0.5 * R - 0.419 * G - 0.081 * B + 128

# RGB轉YCbCr
Y_YCbCr = 0.299 * R + 0.587 * G + 0.114 * B
Cb = -0.168736 * R - 0.331264 * G + 0.5 * B + 128
Cr = 0.5 * R - 0.418688 * G - 0.081312 * B + 128

# 創建灰階圖像
def save_grayscale_image(channel, filename):
    image = Image.fromarray(np.uint8(channel))
    image.save(filename)

# 儲存R、G、B、Y、U、V、Cb、Cr灰階圖像
save_grayscale_image(R, 'R_channel.png')
save_grayscale_image(G, 'G_channel.png')
save_grayscale_image(B, 'B_channel.png')
save_grayscale_image(Y, 'Y_channel.png')
save_grayscale_image(U, 'U_channel.png')
save_grayscale_image(V, 'V_channel.png')
save_grayscale_image(Cb, 'Cb_channel.png')
save_grayscale_image(Cr, 'Cr_channel.png')