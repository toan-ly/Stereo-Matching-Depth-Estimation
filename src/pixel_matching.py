import cv2
import numpy as np
from src.similarity import distance

def pixel_wise_matching(left_im_path, right_im_path, disparity_range, method='L1', save_result=True):
    # Read left and right images then convert to grayscale
    left = cv2.imread(left_im_path, 0).astype(np.float32)
    right = cv2.imread(right_im_path, 0).astype(np.float32)

    height, width = left.shape[:2]

    # Create blank disparity map
    depth = np.zeros((height, width), np.uint8)
    scale = 16
    max_value = 255
    if method == 'L2':
        max_value **= 2

    for y in range(height):
        for x in range(width):
            disparity = 0
            cost_min = max_value

            for d in range(disparity_range):
                cost = max_value if (x - d) < 0 else distance(int(left[y, x]), int(right[y, x - d]), method)
                if cost < cost_min:
                    cost_min = cost
                    disparity = d
            
            depth[y, x] = disparity * scale
        
    if save_result:
        print('Saving result...')
        cv2.imwrite(f'assets/pixel_wise_{method.lower()}.png', depth)
        cv2.imwrite(f'assets/pixel_wise_{method.lower()}_color.png', cv2.applyColorMap(depth, cv2.COLORMAP_JET))
    
    print('Done!')
    return depth

    
