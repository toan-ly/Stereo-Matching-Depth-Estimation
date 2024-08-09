import matplotlib.pyplot as plt
import cv2
import time

from src.pixel_matching import pixel_wise_matching
from src.window_matching import *
import src.cosine_similarity as cs

def plot_images(left_im_path, right_im_path, disparity_map, method, distance_metric, extra_info=None):
    left_im = cv2.imread(left_im_path)
    right_im = cv2.imread(right_im_path)

    extra_info_str = f'kernel {extra_info}' if extra_info else ""
    title_str = f'{method}, {distance_metric}, {extra_info_str}'

    plt.figure(figsize=(12, 10))

    plt.subplot(221)
    plt.title('Left Image')
    plt.imshow(cv2.cvtColor(left_im, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(222)
    plt.title('Right Image')
    plt.imshow(cv2.cvtColor(right_im, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(223)
    plt.title(f'Disparity Map Grayscale ({title_str})')
    plt.imshow(disparity_map, cmap='gray')
    plt.axis('off')

    plt.subplot(224)
    plt.title(f'Disparity Map ({title_str})')
    plt.imshow(disparity_map, cmap='jet')
    plt.axis('off')

    output_filename = f'assets/disparity_map_{method.lower()}_{distance_metric.lower()}{extra_info_str.split(" ")[-1]}.png'
    plt.tight_layout()
    plt.savefig(output_filename)
    plt.show()

def compute_and_plot_disparity_map(left_img_path, right_img_path, disparity_range, method, kernel_size=None):
    distance_metrics = ['L1', 'L2'] if method != 'Cosine Similarity' else ['Cosine Similarity']

    for distance_metric in distance_metrics:
        start_time = time.time()
        if method == 'Pixel-wise':
            disparity_map = pixel_wise_matching(left_img_path, right_img_path, disparity_range, method=distance_metric, save_result=True)
        elif method == 'Window-based':
            disparity_map = window_based_matching_optimized(left_img_path, right_img_path, disparity_range, kernel_size=kernel_size, method=distance_metric, save_result=True)
        elif method == 'Cosine Similarity':
            disparity_map = cs.window_based_matching_optimized(left_img_path, right_img_path, disparity_range, kernel_size=kernel_size, save_result=True)
            distance_metric = ''
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f'{method} ({distance_metric}) took {elapsed_time:.3f} seconds.')

        plot_images(left_img_path, right_img_path, disparity_map, method, distance_metric, extra_info=kernel_size)


if __name__ == '__main__':
    left_img_path = 'data/tsukuba/left.png'
    right_img_path = 'data/tsukuba/right.png'

    # Parameters
    disparity_range = 16

    # Compute disparity maps for pixel wise matching method
    compute_and_plot_disparity_map(
        left_img_path,
        right_img_path,
        disparity_range,
        method='Pixel-wise'
    )

    left_img_path = 'data/Aloe/Aloe_left_1.png'
    right_img_path = 'data/Aloe/Aloe_right_1.png'
    disparity_range = 64
    kernel_sizes = [3, 5]

    for kernel_size in kernel_sizes:
        # Compute disparity maps for window based matching method
        compute_and_plot_disparity_map(
            left_img_path,
            right_img_path,
            disparity_range,
            method='Window-based',
            kernel_size=kernel_size
        )
        
    right_img_path = 'data/Aloe/Aloe_right_2.png'
    kernel_size = 5
    # Compute disparity maps for window based matching method
    compute_and_plot_disparity_map(
        left_img_path,
        right_img_path,
        disparity_range,
        method='Window-based',
        kernel_size=kernel_size
    )
    
    # Compute disparity maps for window based matching method with cosine smilarity
    compute_and_plot_disparity_map(
        left_img_path,
        right_img_path,
        disparity_range,
        method='Cosine Similarity',
        kernel_size=kernel_size
    )
        
        
