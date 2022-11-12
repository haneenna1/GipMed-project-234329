import os
import PIL.Image
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from colorsys import rgb_to_hsv


def segmentationPerImage(thumbFile: str, outPath: str, magnification):
    """
    @segmentationPerImage: the wrapper function for making a segmentation per image
    @args:
        1. thumb - the WSI with thumb format
        2. outPath - output path with string format
        3. magnification - TODO:AMIR
    """
    fn, data_format = os.path.splitext(os.path.basename(thumbFile))
    thumb = PIL.Image.open(thumbFile)

    # this helps avoid the grid, we don't have it within our datasets
    use_otsu3 = False
    thmb_seg_map, edge_image = segmentationPerImageInt(thumb, magnification, use_otsu3=use_otsu3)
    thmb_seg_image = PIL.Image.blend(thumb, edge_image, 0.5)

    # Saving segmentation map, segmentation image:
    thmb_seg_map.save(os.path.join(outPath, fn + '_SegMap.png'))
    thmb_seg_image.save(os.path.join(outPath, fn + '_SegImage.jpg'))


def segmentationPerImageInt(image: PIL.Image, magnification: int, use_otsu3: bool) -> (PIL.Image, PIL.Image):
    """
    @segmentationPerImageInt: the internal function for calculation the segmentation for an image
    @args:
        1. image - the WSI with thumb format
        2. magnification - TODO:AMIR
        3. use_otsu3 - boolean indicates using the improved outso thresholding - mainly when the image
        has three different colors for example having brown grid in our image in addition to the white background
    """

    # Converting the image from RGBA to CMYK and to a numpy array (from PIL):
    image_array = np.array(image.convert('CMYK'))[:, :, 1]

    # Make almost total black into total white to ignore black areas in otsu
    image_array_rgb = np.array(image)
    image_is_black = np.prod(image_array_rgb, axis=2) < 20 ** 3
    image_array[image_is_black] = 0

    # otsu Thresholding:
    _, seg_map = cv.threshold(image_array, 0, 255, cv.THRESH_OTSU)

    # Try otsu3 in HED color space
    temp = False
    if temp:
        import HED_space
        image_HED = HED_space.RGB2HED(np.array(image))
        image_HED_normed = (image_HED - np.min(image_HED)) / (np.max(image_HED) - np.min(image_HED))  # rescale to 0-1
        HED_space.plot_image_in_channels(image_HED_normed, '')
        image_HED_int = (image_HED_normed * 255).astype(np.uint8)
        HED_space.plot_image_in_channels(image_HED_int, '')
        thresh_HED = otsu3(image_HED_int[:, :, 0])
        _, seg_map_HED = cv.threshold(image_HED[:, :, 0], thresh_HED[1], 255, cv.THRESH_BINARY)
        plt.imshow(seg_map_HED)

    # Test median pixel color to inspect segmentation
    if use_otsu3:
        image_array_rgb = np.array(image)
        pixel_vec = image_array_rgb.reshape(-1, 3)[seg_map.reshape(-1) > 0]
        median_color = np.median(pixel_vec, axis=0)
        median_hue = rgb_to_hsv(*median_color / 256)[0] * 360
        # if all(median_color > 180): #median pixel is white-ish, changed from 200
        if (median_hue < 250):  # RanS 19.5.21, median seg hue is not purple/red
            # take upper threshold
            thresh = otsu3(image_array)
            _, seg_map = cv.threshold(image_array, thresh[1], 255, cv.THRESH_BINARY)

    # Smoothing the tissue segmentation imaqe:
    size = 10 * magnification
    kernel_smooth = np.ones((size, size), dtype=np.float32) / size ** 2
    seg_map_filt = cv.filter2D(seg_map, -1, kernel_smooth)

    th_val = 5
    seg_map_filt[seg_map_filt > th_val] = 255
    seg_map_filt[seg_map_filt <= th_val] = 0

    # find small contours and delete them from segmentation map
    size_thresh = 5000  # Lung cancer biopsies can be very small
    contours, _ = cv.findContours(seg_map_filt, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    contour_area = np.zeros(len(contours))

    for ii in range(len(contours)):
        contour_area[ii] = cv.contourArea(contours[ii])

    max_contour = np.max(contour_area)
    small_contours_bool = (contour_area < size_thresh) & (contour_area < max_contour * 0.02)

    if not use_otsu3:  # Do not filter small parts filtering for lung cancer
        small_contours = [contours[ii] for ii in range(len(contours)) if small_contours_bool[ii] == True]
        seg_map_filt = cv.drawContours(seg_map_filt, small_contours, -1, (0, 0, 255),thickness=cv.FILLED)  # delete the small contours

    # Check contour color only for large contours
    hsv_image = np.array(image.convert('HSV'))  # temp RanS 24.2.21
    rgb_image = np.array(image)
    large_contour_ind = np.where(small_contours_bool == False)[0]
    white_mask = np.zeros(seg_map.shape, np.uint8)
    white_mask[np.any(rgb_image < 240, axis=2)] = 255
    gray_contours_bool = np.zeros(len(contours), dtype=bool)
    for ii in large_contour_ind:
        # get contour mean color
        mask = np.zeros(seg_map.shape, np.uint8)
        cv.drawContours(mask, [contours[ii]], -1, 255, thickness=cv.FILLED)
        contour_color, _ = cv.meanStdDev(rgb_image, mask=mask)  # RanS 24.2.21
        contour_std = np.std(contour_color)
        if contour_std < 5:
            hist_mask = cv.bitwise_and(white_mask, mask)
            mean_col, _ = cv.meanStdDev(hsv_image, mask=hist_mask)  # temp RanS 24.2.21
            mean_hue = mean_col[0]
            if mean_hue < 100:
                gray_contours_bool[ii] = True

    gray_contours = [contours[ii] for ii in large_contour_ind if gray_contours_bool[ii] == True]
    seg_map_filt = cv.drawContours(seg_map_filt, gray_contours, -1, (0, 0, 255),thickness=cv.FILLED)  # delete the small contours

    # Multiply seg_map with seg_map_filt
    seg_map *= (seg_map_filt > 0)
    seg_map_PIL = PIL.Image.fromarray(seg_map)

    edge_image = cv.Canny(seg_map, 1, 254)
    # Make the edge thicker by dilating:
    kernel_dilation = np.ones((3, 3))
    edge_image = PIL.Image.fromarray(cv.dilate(edge_image, kernel_dilation, iterations=magnification * 2)).convert(
        'RGB')

    return seg_map_PIL, edge_image


def otsu3(img)->(int,int):
    """
    @otsu3 : imporoved otsu
    """
    hist = cv.calcHist([img], [0], None, [256], [0, 256])
    hist_norm = hist.ravel() / hist.sum()
    Q = hist_norm.cumsum()
    bins = np.arange(256)
    fn_min = np.inf
    thresh = -1, -1
    for i in range(1, 256):
        for j in range(i + 1, 256):
            p1, p2, p3 = np.hsplit(hist_norm, [i, j])  # probabilities
            q1, q2, q3 = Q[i], Q[j] - Q[i], Q[255] - Q[j]  # cum sum of classes
            if q1 < 1.e-6 or q2 < 1.e-6 or q3 < 1.e-6:
                continue
            b1, b2, b3 = np.hsplit(bins, [i, j])  # weights
            # finding means and variances
            m1, m2, m3 = np.sum(p1 * b1) / q1, np.sum(p2 * b2) / q2, np.sum(p3 * b3) / q3
            v1, v2, v3 = np.sum(((b1 - m1) ** 2) * p1) / q1, np.sum(((b2 - m2) ** 2) * p2) / q2, np.sum(
                ((b3 - m3) ** 2) * p3) / q3
            # calculates the minimization function
            fn = v1 * q1 + v2 * q2 + v3 * q3
            if fn < fn_min:
                fn_min = fn
                thresh = i, j
    return thresh
