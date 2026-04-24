import skimage as ski
import numpy as np
from scipy import ndimage as ndi
import matplotlib.pyplot as plt

def canny_edge_detection(img_in, sigma, low_threshold, high_threshold): 
    """
    Performs Canny Edge Detection on an input image.
    """
    rows, cols = img_in.shape

    # 1. Gaussian Filtering for Noise Reduction
    filter_size = int(6*sigma)
    filter_size = filter_size + (filter_size % 2 == 0)
    cent = filter_size // 2
    x = np.linspace(-cent, cent, filter_size)
    y = np.linspace(-cent, cent, filter_size)
    xg, yg = np.meshgrid(x, y)
    h_gauss = np.exp(-(xg**2 + yg**2) / (2 * sigma**2))
    h_gauss /= h_gauss.sum()

    img_filt = ndi.correlate(img_in, h_gauss, mode='nearest')

    # 2. Gradient Calculation using Sobel Operators
    Hx = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    Hy = np.transpose(Hx)

    Gx = ndi.correlate(img_filt, Hx, mode='nearest') 
    Gy = ndi.correlate(img_filt, Hy, mode='nearest') 
    
    # 3. Magnitude and Angle Calculation
    mag = np.sqrt(np.square(Gx) + np.square(Gy))
    angle = np.rad2deg(np.arctan2(Gy, Gx))

    # 4. Gradient Quantization (0, 45, 90, 135 degrees)
    quant_angle = np.zeros(angle.shape)
    angle_180 = angle % 180

    mask_0 = (angle_180 < 22.5) | (angle_180 >= 157.5)
    mask_45 = (angle_180 >= 22.5) & (angle_180 < 67.5)
    mask_90 = (angle_180 >= 67.5) & (angle_180 < 112.5)
    mask_135 = (angle_180 >= 112.5) & (angle_180 < 157.5)

    quant_angle[mask_0] = 0
    quant_angle[mask_45] = 45
    quant_angle[mask_90] = 90
    quant_angle[mask_135] = -45

    # 5. Non-Maximum Suppression (NMS)
    img_nms = np.zeros(img_in.shape)
    for i in range(1, rows-1):
        for j in range(1, cols-1):
            mag_curr = mag[i,j]
            ang_curr = quant_angle[i,j]
            t1, t2 = 0, 0

            if ang_curr == 90:
                t1, t2 = mag[i, j+1], mag[i, j-1]
            elif ang_curr == -45:
                t1, t2 = mag[i-1, j+1], mag[i+1, j-1]
            elif ang_curr == 45:
                t1, t2 = mag[i+1, j+1], mag[i-1, j-1]
            elif ang_curr == 0:
                t1, t2 = mag[i+1, j], mag[i-1, j]

            if (mag_curr >= t1) and (mag_curr >= t2):
                img_nms[i,j] = mag_curr
            else:
                img_nms[i,j] = 0

    # 6. Double Thresholding
    strong_edges = (img_nms >= high_threshold).astype(np.float64)
    weak_edges = ((img_nms >= low_threshold) & (img_nms < high_threshold)).astype(np.float64)

    # 7. Edge Tracking by Hysteresis
    img_final = strong_edges.copy()
    diff = True
    while diff:
        prev = img_final.copy()
        wi, wj = np.where((weak_edges == 1) & (img_final == 0))
        for k in range(len(wi)):
            i, j = wi[k], wj[k]
            i1, i2 = max(0, i-1), min(rows-1, i+1)
            j1, j2 = max(0, j-1), min(cols-1, j+1)
            
            if np.any(prev[i1:i2+1, j1:j2+1] == 1):
                img_final[i, j] = 1.0

        if np.array_equal(img_final, prev):
            diff = False

    return img_final


# testing the function
if __name__ == "__main__":
    # Load a built-in test image
    from skimage import data
    
    # Using the 'camera' image as a standard benchmark
    image = ski.img_as_float(data.camera())
    
    # Run your implementation
    # Parameters: sigma=1.4, low_threshold=0.1, high_threshold=0.3
    edges = canny_edge_detection(image, 1.4, 0.1, 0.3)
    
    # Save or show the result
    #plt.figure(figsize=(12, 6))
    #plt.subplot(1, 2, 1)
    #plt.imshow(image, cmap='gray')
    #plt.axis('off')
    
    #plt.subplot(1, 2, 2)
    #plt.imshow(edges, cmap='gray')
    #plt.axis('off')
    
    #plt.tight_layout()
    #plt.savefig('canny_demo_result.png')
    ski.io.imsave('canny_demo_result.png', (edges * 255).astype(np.uint8))
    print("Demo completed. Result saved as canny_demo_result.png")