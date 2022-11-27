
##histogram

import numpy as np
import cv2
import matplotlib.pyplot as plt

img1 = cv2.imread('1.jpg')
image1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
##src=cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)

img2 = cv2.imread('2.jpg')
image2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
##ref=cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY)

if len(np.shape(image1)) != len(np.shape(image2)):
    if len(np.shape(image1)) == 3:
        image1 = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)
    elif len(np.shape(image2)) == 3:
        image2 = cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY)


def mixhist(src, ref):
    ms, ns = np.shape(src)
    mr, nr = np.shape(ref)

    def hist_find(im, m, n):
        h = np.zeros(256, 'int32')
        for i in range(256):
            h[i] = len(np.where(im == i)[0])

        return h

    def hist_cdf(h, m, n):
        cdf = np.copy(h)
        for i in range(255):
            cdf[i + 1] = cdf[i] + h[i + 1]
        cdf = cdf / (m * n)
        return cdf

    def hist_match(src, ref, fs, fr, m, n):
        img = np.copy(src)
        for i in range(256):
            if len(np.where(src == i)[0]) >= 1:
                dif = np.abs(fr - fs[i])
                k = np.where(dif == np.min(dif))
                if len(k[0]) == 1:
                    img[np.where(src == i)[0], np.where(src == i)[1]] = k[0][0]
                elif len(k[0]) >= 2:
                    diff2 = np.abs(k[0] - i)
                    kk = np.where(diff2 == np.min(diff2))
                    img[np.where(src == i)[0], np.where(src == i)[1]] = diff2[kk] + i
        return img

    hs = hist_find(src, ms, ns)
    hr = hist_find(ref, mr, nr)
    cdfs = hist_cdf(hs, ms, ns)
    cdfr = hist_cdf(hr, mr, nr)
    fimage = hist_match(src, ref, cdfs, cdfr, ms, ns)

    return fimage


if len(np.shape(image1)) == 3:
    src = np.copy(image1)
    fimage = np.copy(image1)
    ref = np.copy(image2)
    for i in range(3):
        fimage[:, :, i] = mixhist(src[:, :, i], ref[:, :, i])
    plt.subplot(221)
    plt.imshow(src)
    plt.axis('off')
    plt.title('source image')

    plt.subplot(222)
    plt.imshow(ref)
    plt.axis('off')
    plt.title('refrence image')

    plt.subplot(223)
    plt.imshow(fimage)
    plt.axis('off')
    plt.title('mixed image')
    plt.show()
elif len(np.shape(image1)) == 2:
    mixhist(image1, image2)
    plt.subplot(221)
    plt.imshow(cv2.cvtColor(src, cv2.COLOR_GRAY2RGB))
    plt.axis('off')
    plt.title('source image')

    plt.subplot(222)
    plt.imshow(cv2.cvtColor(ref, cv2.COLOR_GRAY2RGB))
    plt.axis('off')
    plt.title('refrence image')

    plt.subplot(223)
    plt.imshow(cv2.cvtColor(fimage, cv2.COLOR_GRAY2RGB))
    plt.axis('off')
    plt.title('mixed image')
    plt.show()
