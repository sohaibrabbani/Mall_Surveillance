import cv2
import numpy as np
from matplotlib import pyplot as plt


# top_view = cv2.cvtColor(cv2.imread('data/top_view_720p.jpg'), cv2.COLOR_BGR2RGB)/255
# dummy_points = [[159, 390],
#                 # [158, 391], [158, 391], [165, 391], [166, 391], [157, 391], [164, 391], [164, 391], [184, 396], [185, 396], [199, 396], [218, 400], [218, 400], [194, 400], [192, 394], [192, 394], [191, 394], [191, 394], [199, 395], [220, 398], [132, 379],
#                 [132, 379]]


def heatmap(points, top_view):
    top_view = top_view / 255
    k = 21
    gauss = cv2.getGaussianKernel(k, np.sqrt(64))
    gauss = gauss * gauss.T
    gauss = gauss / gauss[k//2, k//2]
    spark = cv2.applyColorMap((gauss * 255).astype(np.uint8), cv2.COLORMAP_HOT).astype(np.float32)/255

    heat = np.zeros(top_view.shape).astype(np.float32)

    for p in points:
        heat[p[1] - k // 2: 1 + p[1] + k // 2, p[0] - k // 2: 1 + p[0] + k // 2, :] += spark

    heat = heat / (np.max(heat, axis=(0, 1)) + 0.0001)
    gray = cv2.cvtColor(heat, cv2.COLOR_RGB2GRAY)
    mask_heat = np.where(gray > 0.1, 1, 0).astype(np.float32)[:, :, None]
    mask_top = np.ones(top_view.shape) * (1 - mask_heat)
    new_top = top_view * mask_top + heat * mask_heat
    return new_top * 255


# plt.imshow(heatmap(dummy_points, top_view), cmap='gray')
# plt.show()

