# Video Compression Hw#1: Color Transform

#### 313551055 柯柏旭

 ## RGB -> YUV

```python
# RGB轉YUV
Y = 0.299 * R + 0.587 * G + 0.114 * B
U = -0.169 * R - 0.331 * G + 0.5 * B + 128
V = 0.5 * R - 0.419 * G - 0.081 * B + 128
```

## RGB -> YCbCr

```python
# RGB轉YCbCr
Y_YCbCr = 0.299 * R + 0.587 * G + 0.114 * B
Cb = -0.168736 * R - 0.331264 * G + 0.5 * B + 128
Cr = 0.5 * R - 0.418688 * G - 0.081312 * B + 128
```

## Input

![image-20240924162951041](/Users/hentci/Library/Application Support/typora-user-images/image-20240924162951041.png)

## Output results

- R_channel

![image-20240924162753595](/Users/hentci/Library/Application Support/typora-user-images/image-20240924162753595.png)

- G_channel

![image-20240924162832203](/Users/hentci/Library/Application Support/typora-user-images/image-20240924162832203.png)

- B_channel

![image-20240924162841548](/Users/hentci/Library/Application Support/typora-user-images/image-20240924162841548.png)

- Y_channel

![image-20240924162558236](/Users/hentci/Library/Application Support/typora-user-images/image-20240924162558236.png)

- U_channel

![image-20240924162609142](/Users/hentci/Library/Application Support/typora-user-images/image-20240924162609142.png)

- V_channel

![image-20240924162616917](/Users/hentci/Library/Application Support/typora-user-images/image-20240924162616917.png)

- Cb_channel

![image-20240924162910431](/Users/hentci/Library/Application Support/typora-user-images/image-20240924162910431.png)

- Cr_channel

![image-20240924162916774](/Users/hentci/Library/Application Support/typora-user-images/image-20240924162916774.png)