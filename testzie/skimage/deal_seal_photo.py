import numpy as np
import matplotlib.pyplot as plt
from skimage import data, draw, color, transform, feature, measure, \
    filters, morphology, segmentation, restoration, io, img_as_float

image = io.imread('./testzie/dataset/seal_real.jpg')
image = io.imread('./testzie/dataset/seal_bad.jpg')
image = io.imread('./testzie/dataset/seal.jpg')

# image_gray = color.rgb2gray(image)
image_gray = (255 - np.dot(image, [0, 1/2, 1/2])).astype('uint8')
thresh = filters.threshold_otsu(image_gray)
thresh = filters.threshold_yen(image_gray)
thresh = (filters.threshold_otsu(image_gray) + filters.threshold_yen(image_gray)) / 2
bw = image_gray > thresh
# bw = morphology.closing(bw, morphology.square(3))
# bw = morphology.opening(bw, morphology.square(3))
# bw = morphology.closing(bw, morphology.square(3))
# bw = morphology.opening(bw, morphology.square(3))
# bw = morphology.opening(bw, morphology.square(3))
# bw = morphology.opening(bw, morphology.square(3))
plt.imshow(image_gray)
plt.imshow(bw)

image_gray_fix = image_gray - image_gray.min()
image_gray_fix = image_gray_fix / image_gray_fix.max()
plt.imshow(image_gray_fix, 'gray')

image_red = np.zeros(image_gray_fix.shape + (4, ), 'float32')
image_red[:, :, 0] = 1
image_red[:, :, 1] = (1 - image_gray_fix)
image_red[:, :, 2] = (1 - image_gray_fix)
image_red[:, :, 3] = 0.75
image_red = image_red * bw[:, :, np.newaxis]
plt.imshow(image_red)

fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(8, 6))
ax0.imshow(image)
ax1.imshow(image_red)
plt.imsave('c:/users/dell/desktop/seal_real.png', image_red)

image = img_as_float(image)
blurred = filters.gaussian(image, sigma=1, multichannel=True)
sharper = np.clip(image * 1.5 - blurred * 0.5, 0, 1.0)
io.imshow(blurred)
io.imshow(sharper)
image_gray = (1 - np.dot(sharper, [0, 1/2, 1/2]))
