"""
    ----------------- FINAL PROJECT ----------------
    TEAM:
    (1) : Veera Raghava Reddy Katamreddy (A20339534)
    (2) : Noor Afshan Fathima (A20385838)

    Course  : Computer Vision (CS-512)
    Year    : FALL-2017
    ------------------------------------------------
"""

import copy

import cv2
import numpy as np
import sys


def load_image(file_name=None):
    image = cv2.imread(filename=file_name)
    return image


def convert_to_grayscale(src=None):
    gray_image = cv2.cvtColor(src=src, code=cv2.COLOR_BGR2GRAY)
    return gray_image


def resize_image(src=None):
    resized_image = cv2.resize(src=src, dsize=(1280, 720))
    return resized_image


def gaussian_smooth(src=None, ksize=None, sigma_x=None, sigma_y=None):
    gauss_smoothed_image = cv2.GaussianBlur(src=src, ksize=(ksize, ksize), sigmaX=sigma_x, sigmaY=sigma_y)
    return gauss_smoothed_image


def get_x_derivative(src=None, ksize=None):
    x_derivative = cv2.Sobel(src=src, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=ksize)
    return x_derivative


def get_y_derivative(src=None, ksize=None):
    y_derivative = cv2.Sobel(src=src, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=ksize)
    return y_derivative


def get_gradient_magnitude(x_derivative=None, y_derivative=None):
    gradient_magnitude = np.hypot(x_derivative, y_derivative)
    return gradient_magnitude


def get_angles(x_derivative=None, y_derivative=None):
    angles = np.arctan2(y_derivative, x_derivative)
    return angles


def get_absolute_angles(angles_mat=None):
    absolute_angles = angles_mat * 180 / np.pi
    return absolute_angles


def non_maximum_suppression(ht=None, wt=None, ang=None, mag=None, gx=None, gy=None):
    non_max_suppressed_image = np.zeros(shape=(ht, wt))
    for h in range(0, ht - 1):
        for w in range(0, wt - 1):
            if 0 <= ang[h, w] <= 45 or -135 > ang[h, w] >= -180:
                yb = (mag[h, w + 1], mag[h + 1, w + 1])
                yt = (mag[h, w - 1], mag[h - 1, w - 1])
                if mag[h, w] == 0:
                    xe = 0
                else:
                    xe = np.absolute(gy[h, w] / mag[h, w])
                if mag[h, w] >= (yb[1] - yb[0]) * xe + yb[0] and mag[h, w] >= (yt[1] - yt[0]) * xe + yt[0]:
                    non_max_suppressed_image[h, w] = mag[h, w]
            elif 45 < ang[h, w] <= 90 or -90 > ang[h, w] >= -135:
                yb = (mag[h + 1, w], mag[h + 1, w + 1])
                yt = (mag[h - 1, w], mag[h - 1, w - 1])
                if mag[h, w] == 0:
                    xe = 0
                else:
                    xe = np.absolute(gx[h, w] / mag[h, w])
                if mag[h, w] >= (yb[1] - yb[0]) * xe + yb[0] and mag[h, w] >= (yt[1] - yt[0]) * xe + yt[0]:
                    non_max_suppressed_image[h, w] = mag[h, w]
            elif 90 < ang[h, w] <= 135 or -45 > ang[h, w] >= -90:
                yb = (mag[h + 1, w], mag[h + 1, w - 1])
                yt = (mag[h - 1, w], mag[h - 1, w + 1])
                if mag[h, w] == 0:
                    xe = 0
                else:
                    xe = np.absolute(gx[h, w] / mag[h, w])
                if mag[h, w] >= (yb[1] - yb[0]) * xe + yb[0] and mag[h, w] >= (yt[1] - yt[0]) * xe + yt[0]:
                    non_max_suppressed_image[h, w] = mag[h, w]
            elif 135 < ang[h, w] <= 180 or 0 > ang[h, w] >= -45:
                yb = (mag[h, w - 1], mag[h + 1, w - 1])
                yt = (mag[h, w + 1], mag[h - 1, w + 1])
                if mag[h, w] == 0:
                    xe = 0
                else:
                    xe = np.absolute(gx[h, w] / mag[h, w])
                if mag[h, w] >= (yb[1] - yb[0]) * xe + yb[0] and mag[h, w] >= (yt[1] - yt[0]) * xe + yt[0]:
                    non_max_suppressed_image[h, w] = mag[h, w]
    return non_max_suppressed_image


def threshold_image(htr=None, ltr=None, mag=None):
    high_threshold = np.max(mag) * htr
    low_threshold = high_threshold * ltr
    strong_edges = np.array((mag > high_threshold), dtype=np.uint8)
    weak_edges = np.array((mag > low_threshold), dtype=np.uint8)
    thresholded_edges = strong_edges + weak_edges
    weak_edges = np.array(np.bitwise_and((mag > low_threshold), (mag < high_threshold)), dtype=np.uint8)
    return strong_edges, thresholded_edges, weak_edges


def hysteresis_tracing(th_img=None, th_edg=None):
    final_edges = copy.deepcopy(th_img)
    current_pixels = list()
    height, width = th_img.shape
    for row in range(1, height - 1):
        for col in range(1, width - 1):
            if th_edg[row, col] != 1:
                continue
            local_patch = th_edg[row - 1:row + 2, col - 1:col + 2]
            patch_max = local_patch.max()
            if patch_max == 2:
                current_pixels.append((row, col))
                final_edges[row, col] = 1
    while len(current_pixels) > 0:
        new_pix = list()
        for row, col in current_pixels:
            for dr in range(-1, 2):
                for dc in range(-1, 2):
                    if dr == 0 and dc == 0:
                        continue
                    r2 = row + dr
                    c2 = col + dc
                    if th_edg[r2, c2] == 1 and final_edges[r2, c2] == 0:
                        new_pix.append((r2, c2))
                        final_edges[r2, c2] = 1
        current_pixels = new_pix
    return final_edges


def normalize_image(src=None):
    normalized_image = src / np.max(src)
    return normalized_image


def show_image(window_name=None, image=None, wait_time=0):
    cv2.imshow(winname=window_name, mat=image)
    cv2.waitKey(delay=wait_time)


def close_windows(window_name=None):
    if window_name is not None:
        cv2.destroyWindow(winname=window_name)
    else:
        cv2.destroyAllWindows()


def convert_image_to_float64(src=None):
    image_64f = np.float64(src)
    return image_64f


def main(src_img=None):
    mat = resize_image(src=src_img)
    gray = convert_to_grayscale(src=mat)
    height, width = gray.shape

    gray_64f = convert_image_to_float64(src=gray)
    gauss_gray_64f = gaussian_smooth(src=gray_64f, ksize=5, sigma_x=1, sigma_y=1)
    norm_gauss_gray = normalize_image(src=gauss_gray_64f)
    show_image(window_name="Original Image", image=mat, wait_time=1)
    show_image(window_name="Grayscale Image", image=gray, wait_time=1)
    show_image(window_name="Gaussian Smoothed Image", image=norm_gauss_gray, wait_time=1)

    x_derv = get_x_derivative(src=gauss_gray_64f, ksize=3)
    y_derv = get_y_derivative(src=gauss_gray_64f, ksize=3)
    show_image(window_name="X-Derivative", image=cv2.convertScaleAbs(x_derv), wait_time=1)
    show_image(window_name="Y-Derivative", image=cv2.convertScaleAbs(y_derv), wait_time=1)

    gauss_x_derv = gaussian_smooth(src=x_derv, ksize=5, sigma_x=1.5, sigma_y=1.5)
    gauss_y_derv = gaussian_smooth(src=y_derv, ksize=5, sigma_x=1.5, sigma_y=1.5)
    show_image(window_name="Gaussian Smoothed X-Derivative", image=cv2.convertScaleAbs(gauss_x_derv), wait_time=1)
    show_image(window_name="Gaussian Smoothed Y-Derivative", image=cv2.convertScaleAbs(gauss_y_derv), wait_time=1)

    grad_mag_gauss = get_gradient_magnitude(x_derivative=gauss_x_derv, y_derivative=gauss_y_derv)
    show_image(window_name="Gradient Magnitude", image=cv2.convertScaleAbs(grad_mag_gauss), wait_time=1)

    gauss_angles = get_angles(x_derivative=gauss_x_derv, y_derivative=gauss_y_derv)
    gauss_absolute_angles = get_absolute_angles(angles_mat=gauss_angles)

    non_max_suppress = non_maximum_suppression(ht=height, wt=width, ang=gauss_absolute_angles, mag=grad_mag_gauss,
                                               gx=gauss_x_derv, gy=gauss_y_derv)
    kernel = np.ones((5, 5), np.uint8)
    show_image(window_name="Non-Maximum Suppression", image=np.uint8(non_max_suppress), wait_time=1)

    high_threshold_ratio = h_t
    low_threshold_ratio = l_t

    strong_edg, thresh_edg, weak_edg = threshold_image(htr=high_threshold_ratio, ltr=low_threshold_ratio,
                                                       mag=non_max_suppress)
    show_image(window_name="Thresholded Image - Strong Edges", image=cv2.convertScaleAbs(strong_edg * 255), wait_time=1)
    show_image(window_name="Thresholded Image - Weak Edges", image=cv2.convertScaleAbs(weak_edg * 255), wait_time=1)
    show_image(window_name="Thresholded Image - Combined Edges", image=cv2.convertScaleAbs(thresh_edg * 255),
               wait_time=1)

    hysteresis_traced_image = hysteresis_tracing(th_img=strong_edg, th_edg=thresh_edg)
    norm_hysteresis_traced_image = np.uint8(normalize_image(src=hysteresis_traced_image) * 255)
    show_image(window_name="Hysteresis Traced Image", image=norm_hysteresis_traced_image, wait_time=1)

    closing = cv2.morphologyEx(norm_hysteresis_traced_image, cv2.MORPH_CLOSE, kernel, iterations=1)
    img_dilation = cv2.dilate(closing, kernel, iterations=2)
    im_flood_fill = img_dilation.copy()
    h, w = img_dilation.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(im_flood_fill, mask, (0, 0), 255)
    cv2.floodFill(im_flood_fill, mask, (int(h / 2) + 25, int(w / 2) + 25), 255)
    im_flood_fill_inv = cv2.bitwise_not(im_flood_fill)
    im_out = img_dilation | im_flood_fill_inv
    im_out = cv2.erode(im_out, kernel, iterations=2)
    show_image(window_name="Morphological Transformation", image=im_out, wait_time=1)

    im_org_out = cv2.bitwise_or(mat, mat, mask=im_out)
    show_image(window_name="Final", image=im_org_out, wait_time=0)


if __name__ == '__main__':
    if "-h" in sys.argv or "--help" in sys.argv:
        print(
            "HELP: \n This Program is intended to perform Boundary estimation using Edge Detection\nProgram takes 3 "
            "input parameters:\n \t(1). File name (Location of the file)\n  \t(2). High Threshold Ratio (0.10 - "
            "0.25)\n \t(3). Low Threshold Ratio (0.05 - 0.20)")
    elif len(sys.argv) == 2:
        f_name = sys.argv[1]
        src_mat = load_image(file_name=f_name)
        h_t = 0.15
        l_t = 0.10
        main(src_img=src_mat)
    elif len(sys.argv) == 4:
        f_name = sys.argv[1]
        h_t = float(sys.argv[2])
        if h_t > 0.25:
            h_t = 0.25
        if h_t < 0.10:
            h_t = 0.10
        l_t = float(sys.argv[3])
        if l_t > 0.20:
            l_t = 0.20
        if l_t < 0.05:
            l_t = 0.05
        src_mat = load_image(file_name=f_name)
        main(src_img=src_mat)
    elif len(sys.argv) == 3:
        f_name = sys.argv[1]
        h_t = float(sys.argv[2])
        if h_t > 0.25:
            h_t = 0.25
        if h_t < 0.10:
            h_t = 0.10
        l_t = h_t - 0.05
        src_mat = load_image(file_name=f_name)
        main(src_img=src_mat)
    elif len(sys.argv) < 2:
        print("Please provide a valid image location!!")
        exit(1)
