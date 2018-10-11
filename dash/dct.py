import itertools
import math

import cv2 as cv
import numpy as np

import utils


def compress(img, num_coeffs=None, scale_factor=1):
    """
    Approximates a single channel image by using only the first coefficients of the DCT.
     First, the image is chopped into 8x8 pixels patches and the DCT is applied to each patch.
     Then, if num_coeffs is provided, only the first K DCT coefficients are kept.
     If not, all the elements are quantized using the JPEG quantization matrix and the scale_factor.
     Finally, the resulting coefficients are used to approximate the original patches with the IDCT, and the image is
     reconstructed back again from these patches.
    :param img: Image to be approximated.
    :param num_coeffs: Number of DCT coefficients to use.
    :param scale_factor: Scale factor to use in the quantization step.
    :return: The approximated image.
    """

    img = np.float32(img)

    # prevent against multiple-channel images
    if len(img.shape) != 2:
        raise ValueError('Input image must be a single channel 2D array')

    # shape of image
    h, w = img.shape

    # No of blocks needed : Calculation

    # new block height
    n_height = np.int32(math.ceil(h / 8)) * 8

    # new block width
    n_width = np.int32(math.ceil(w / 8)) * 8

    # create a numpy zero matrix with size of H,W
    padded_img = np.zeros((n_height, n_width))

    padded_img[0:h, 0:w] = img

    img = np.float32(padded_img)
    height, width = img.shape

    # split into blocks
    img_blocks = [img[j:j + 8, i:i + 8]
                  for (j, i) in itertools.product(range(0, height, 8),
                                                  range(0, width, 8))]

    # DCT transform every block
    dct_blocks = [cv.dct(img_block) for img_block in img_blocks]

    if num_coeffs is not None:
        # keep only the first K DCT coefficients of every block
        reduced_dct_coeffs = [utils.zig_zag(dct_block, num_coeffs) for dct_block in dct_blocks]
    else:
        # quantize all the DCT coefficients using the quantization matrix and the scaling factor
        reduced_dct_coeffs = [np.round(dct_block / (utils.jpeg_quantiz_matrix * scale_factor))
                              for dct_block in dct_blocks]

        # and get the original coefficients back
        reduced_dct_coeffs = [reduced_dct_coeff * (utils.jpeg_quantiz_matrix * scale_factor)
                              for reduced_dct_coeff in reduced_dct_coeffs]

    # IDCT of every block
    rec_img_blocks = [cv.idct(coeff_block) for coeff_block in reduced_dct_coeffs]

    # reshape the reconstructed image blocks
    rec_img = []
    for chunk_row_blocks in utils.chunks(rec_img_blocks, width // 8):
        for row_block_num in range(8):
            for block in chunk_row_blocks:
                rec_img.extend(block[row_block_num])
    rec_img = np.array(rec_img).reshape(height, width)

    # round to the nearest integer [0,255] value
    rec_img[rec_img < 0] = 0
    rec_img[rec_img > 255] = 255
    rec_img = np.uint8(rec_img)

    return rec_img[0:h, 0:w]
