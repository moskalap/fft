import numpy as np
from scipy import ndimage, rot90

pattern = ndimage.imread('/res/fish1.png', flatten=True)
im = ndimage.imread('/res/school.jpg', flatten=True)
imn = ndimage.imread('/res/school.jpg')
print(im.shape)

fp = np.fft.fft2(rot90(pattern, 2), im.shape)
fi = np.fft.fft2(im)
m = np.multiply(fp, fi)
corr = np.fft.ifft2(m)
corr = np.abs(corr)
corr = corr.astype(float)
print(corr.size)
i_M, j_M = corr.shape
it = 0
corr[corr < 0.5 * np.amax(corr)] = 0
for i in range(i_M):
    for j in range(j_M):
        it += 1
        if corr[i, j] > 0:
            print(corr[i, j])
            imn[i, j][0] = 255
            imn[i, j][1] = 255
            imn[i, j][2] = 255

import matplotlib.pyplot as plt

plt.imshow(imn)
plt.show()