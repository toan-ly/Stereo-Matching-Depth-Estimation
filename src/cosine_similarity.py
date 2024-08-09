import cv2
import numpy as np
from src.similarity import cosine_similarity

def window_based_matching(left_im_path, right_im_path, disparity_range, kernel_size=5, save_result=True):
    left = cv2.imread(left_im_path, 0).astype(np.float32)
    right = cv2.imread(right_im_path, 0).astype(np.float32)

    height, width = left.shape[:2]

    depth = np.zeros((height, width), np.uint8)
    kernel_half = int((kernel_size - 1) / 2)
    scale = 3

    for y in range(kernel_half, height - kernel_half):
        for x in range(kernel_half, width - kernel_half):
            disparity = 0
            cost_optimal = -1

            for d in range(disparity_range):
                cost = -1
                if (x - d - kernel_half) > 0:
                    wp = left[(y-kernel_half):(y+kernel_half+1), (x-kernel_half):(x+kernel_half+1)]
                    wqd =  right[(y-kernel_half):(y+kernel_half+1), (x-kernel_half-d):(x-d+kernel_half+1)]

                    wp_flattened = wp.flatten()
                    wqd_flattened = wqd.flatten()

                    cost = cosine_similarity(wp_flattened, wqd_flattened)
            
                if cost > cost_optimal:
                    cost_optimal = cost
                    disparity = d
                
            depth[y, x] = disparity * scale
    
    if save_result:
        print('Saving result...')
        cv2.imwrite('assets/window_based_cosine_similarity.png', depth)
        cv2.imwrite('assets/window_based_cosine_similarity_color.png', cv2.applyColorMap(depth, cv2.COLORMAP_JET))
    print('Done!')
    return depth

def window_based_matching_optimized(left_im_path, right_im_path, disparity_range, kernel_size=5, save_result=True):
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

        numerator = cv2.filter2D(left * shifted_right, -1, kernel)
        print(numerator.shape)
        left_squared_sum = cv2.filter2D(left**2, -1, kernel)
        right_squared_sum = cv2.filter2D(shifted_right**2, -1, kernel)

        denominator = np.sqrt(left_squared_sum * right_squared_sum)
        denominator[denominator == 0] = 1

        similarity = numerator / denominator
        cost_volumes[d] = similarity
        
    disparity = np.argmax(cost_volumes, axis=0)
    depth = (disparity * scale).astype(np.uint8)

    if save_result:
        print('Saving result...')
        cv2.imwrite('assets/window_based_cosine_similarity_optimized.png', depth)
        cv2.imwrite('assets/window_based_cosine_similarity_color_optimized.png', cv2.applyColorMap(depth, cv2.COLORMAP_JET))
    print('Done!')
    return depth
    
    
    