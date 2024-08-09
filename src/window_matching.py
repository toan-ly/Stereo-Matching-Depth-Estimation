import cv2
import numpy as np
# from scipy.ndimage import convolve
from src.similarity import distance

def window_based_matching(left_im_path, right_im_path, disparity_range, kernel_size=5, method='L1', save_result=True):
    left = cv2.imread(left_im_path, 0).astype(np.float32)
    right = cv2.imread(right_im_path, 0).astype(np.float32)

    height, width = left.shape[:2]

    depth = np.zeros((height, width), np.uint8)
    kernel_half = int((kernel_size - 1) / 2)
    scale = 3
    max_value = 255 * 9
    if method == 'L2':
        max_value = 255**2 * 9
        
    for y in range(kernel_half, height - kernel_half):
        for x in range(kernel_half, width - kernel_half):
            disparity = 0
            cost_min = 65534
    
            for d in range(disparity_range):
                total = 0
                value = 0
                for v in range(-kernel_half, kernel_half + 1):
                    for u in range(-kernel_half, kernel_half + 1):
                        value = max_value
                        if (x + u - d) >= 0:
                            value = distance(
                                left[y + v, x + u],
                                right[y + v, x + u - d],
                                method
                            )
                        total += value
                
                if total < cost_min:
                    cost_min = total
                    disparity = d
            
            depth[y, x] = disparity * scale
    
    if save_result:
        print('Saving result...')
        cv2.imwrite(f'assets/window_based_{method.lower()}.png', depth)
        cv2.imwrite(f'assets/window_based_{method.lower()}_color.png', cv2.applyColorMap(depth, cv2.COLORMAP_JET))
    print('Done!')
    return depth

# Optimize using matrix and convolution
def window_based_matching_optimized(left_im_path, right_im_path, disparity_range, kernel_size=5, method='L1', save_result=True):
    left = cv2.imread(left_im_path, 0).astype(np.float32)
    right = cv2.imread(right_im_path, 0).astype(np.float32)

    height, width = left.shape[:2]
    depth = np.zeros((height, width), np.uint8)
    scale = 3

    kernel = np.ones((kernel_size, kernel_size), dtype=np.float32)

    # Precompute the cost volumes for all disparities
    cost_volumes = np.zeros((disparity_range, height, width), dtype=np.float32)

    for d in range(disparity_range):
        shifted_right = np.roll(right, d, axis=1)
        shifted_right[:, :d] = 0
        cost = distance(left, shifted_right, method)

        cost_volumes[d] = cv2.filter2D(cost, -1, kernel)

    # Find the disparity with the minimum cost
    disparity = np.argmin(cost_volumes, axis=0)
    depth = (disparity * scale).astype(np.uint8)

    if save_result:
        print('Saving result...')
        cv2.imwrite(f'assets/window_based_{method.lower()}_optimized.png', depth)
        cv2.imwrite(f'assets/window_based_{method.lower()}_color_optimized.png', cv2.applyColorMap(depth, cv2.COLORMAP_JET))
    print('Done!')

    return depth
 