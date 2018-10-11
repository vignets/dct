import math

import cv2 as cv
import dash_core_components as dcc
import numpy as np
import plotly.graph_objs as go
from skimage.measure import compare_ssim as compute_ssim

import dct


def main(img, num_coeffs=None, scale_factor=1):
    rec_img = dct.compress(img,
                           num_coeffs=num_coeffs,
                           scale_factor=scale_factor)

    # show PSNR and SSIM of the approximation
    err_img = abs(np.array(rec_img, dtype=float) - np.array(img, dtype=float))
    mse = (err_img ** 2).mean()
    psnr = 10 * math.log10((255 ** 2) / mse)
    ssim = compute_ssim(np.float32(rec_img), np.float32(img))
    # print('coeff', num_coeffs)
    # print('scale', scale_factor)
    # print('MSE: %s' % mse)
    # print('PSNR: %s dB' % psnr)
    # print('SSIM: %s' % ssim)
    return [mse, psnr, ssim]


def mseplot(img, mode):
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    mse_list, psnr_list, ssim_list = [], [], []
    scale_factors = list(range(1, 6))
    for factor in scale_factors:
        if mode == 'scale_factor':
            values = main(img, scale_factor=factor)
        else:
            factor = int(factor)
            values = main(img, num_coeffs=factor)
        mse_list.append(values[0])
        psnr_list.append(values[1])
        ssim_list.append(values[2])

    mse_trace = go.Scatter(
        x=scale_factors,
        y=mse_list,
        name='mse'
    )
    ssim_trace = go.Scatter(
        x=scale_factors,
        y=ssim_list,
        xaxis='x2',
        yaxis='y2',
        name='ssim'
    )
    psnr_trace = go.Scatter(
        x=scale_factors,
        y=psnr_list,
        text=['{:.3f} db'.format(x) for x in psnr_list],
        hoverinfo='text',
        xaxis='x3',
        yaxis='y3',
        name='psnr'
    )
    data = [mse_trace, ssim_trace, psnr_trace]
    layout = go.Layout(
        xaxis=dict(
            domain=[0, 0.33]
        ),
        yaxis=dict(
            hoverformat='.3f'
        ),
        xaxis2=dict(
            domain=[0.34, 0.66]
        ),
        yaxis2=dict(
            anchor='x2',
            hoverformat='.3f'
        ),
        xaxis3=dict(
            domain=[0.67, 1]
        ),
        yaxis3=dict(
            anchor='x3',
            hoverformat='.3f'
        )
    )
    fig = go.Figure(data=data, layout=layout)
    return dcc.Graph(
        figure=fig,
        config={
            'modeBarButtonsToRemove': [
                'sendDataToCloud',
            ]
        }
    )
