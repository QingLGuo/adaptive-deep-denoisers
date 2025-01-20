import os
import numpy as np
import cv2
from skimage.metrics import structural_similarity as compare_ssim
import torch
import pandas as pd
from adaptive_deep_dsg_nlm import ADDSGNLM

model = ADDSGNLM()
checkpoint = torch.load('ADDSGNLM_noise10.pth')
model.load_state_dict(checkpoint)

def proj(im_input, minval, maxval):
    im_out = np.where(im_input > maxval, maxval, im_input)
    im_out = np.where(im_out < minval, minval, im_out)
    return im_out
def psnr(x, im_orig):
    norm2 = np.mean((x - im_orig) ** 2)
    psnr = -10 * np.log10(norm2)
    return psnr
def funcAtranspose(im_input, mask, fx, fy):
    m, n = im_input.shape
    fx = int(1 / fx)
    fy = int(1 / fy)
    im_inputres = np.zeros([m * fx, m * fy], im_input.dtype)
    for i in range(m):
        for j in range(n):
            im_inputres[fx * i, fy * j] = im_input[i, j]
    m, n = im_inputres.shape
    w = len(mask[0])
    r = int((w - 1) / 2)
    im_inputres = cv2.copyMakeBorder(im_inputres, r, r, r, r, borderType=cv2.BORDER_WRAP)
    im_output = cv2.filter2D(im_inputres, -1, mask)
    im_output = im_output[r:r + m, r:r + n]
    return im_output
def funcA(im_input, mask, fx, fy):
    m, n = im_input.shape
    w = len(mask[0])
    r = int((w - 1) / 2)
    im_input = cv2.copyMakeBorder(im_input, r, r, r, r, borderType=cv2.BORDER_WRAP)
    im_output = cv2.filter2D(im_input, -1, mask)
    im_output = im_output[r:r + m, r:r + n]
    im_outputres = cv2.resize(im_output, (0, 0), fx=fx, fy=fy, interpolation=cv2.INTER_NEAREST)
    return im_outputres
def pnp_fbs_superresolution(im_input, im_ref, fx, fy, mask, **opts):

    lamda = opts.get('lamda', 2.0)
    rho = opts.get('rho', 1.0)
    maxitr = opts.get('maxitr', 100)
    verbose = opts.get('verbose',1)
    sigma = opts.get('sigma', 5)

    """ Initialization. """
    index = np.nonzero(mask)
    y = funcAtranspose(im_input, mask, fx, fy)
    m, n = y.shape
    x = cv2.resize(im_input, (m, n))
    x_start = torch.tensor(x)
    x_start = torch.unsqueeze(x_start, 0).unsqueeze(0)

    x_start = x_start.float()

    psnr_x = psnr(x,im_orig)
    ssim_x = compare_ssim(x, im_orig, data_range=1.)
    result_iteration_psnr_ssim["Iteration"].append(1)
    result_iteration_psnr_ssim["PSNR"].append(psnr_x)
    result_iteration_psnr_ssim["SSIM"].append(ssim_x)
    result_iteration_psnr_ssim["res"].append(0)

    ytilde = x
    res = np.zeros(maxiter)
    funceval = np.zeros(maxiter)

    """ Main loop. """
    for i in range(maxitr):

        xold = np.copy(x)

        """ Update gradient. """
        xoldhat = funcA(x, mask, fx, fy)
        gradx = funcAtranspose(xoldhat, mask, fx, fy) - y

        """ Denoising step. """
        x = xold - rho * gradx
        x = torch.tensor(x)
        x = torch.unsqueeze(x, 0).unsqueeze(0)

        x = x.float()

        model.eval()
        with torch.no_grad():
            x = model(x)
            x = x.double()
            x = np.squeeze(x)
            x = x.numpy()
            res[i] = np.linalg.norm(x - xold)
        imagename2 = f'ADDSGNLM_superresolution_rho={rho}_k={K}_{maxiter}_{i}_{imagename}'
        output_str1 = outdir_name1 + imagename2
        cv2.imwrite(output_str1, x * 255.0)

        """ Monitoring. """
        if verbose:
            result_psnr = psnr(x,im_ref)
            result_ssim = compare_ssim(x, im_ref, data_range=1.)
            result_iteration_psnr_ssim["Iteration"].append(f"{i+2}")
            result_iteration_psnr_ssim["PSNR"].append(result_psnr)
            result_iteration_psnr_ssim["SSIM"].append(result_ssim)
            result_iteration_psnr_ssim["res"].append(res[i])
            print("i: {}, \t psnr: {} ssim= {} "\
                  .format(i+1, result_psnr, result_ssim))

    return x

if __name__ == "__main__":
    dir_name = 'data/TestData/set12/'
    outdir_name = 'output_image/'
    outdir_name1 = 'output_image/iterations/'
    # imagename = '07.png'
    # outdir_name1='Five_hundred_DynDD_NLM_noise5_k=2_09_hat/'

    os.makedirs(outdir_name,exist_ok=True)
    image_files = os.listdir(dir_name)
    for imagename in image_files:
        if imagename.endswith(".png"):
            input_str = os.path.join(dir_name,imagename)
            print(imagename)

            K = 4  # downsampling factor
            # ---- load the ground truth ----
            im_orig = cv2.imread(input_str, 0) / 255.0
            m, n = im_orig.shape

            # ---- blur the image
            kernel = cv2.getGaussianKernel(9, 1)

            mask = np.outer(kernel, kernel.transpose())
            w = len(mask[0])
            r = int((w - 1) / 2)
            im_orig = cv2.copyMakeBorder(im_orig, r, r, r, r, borderType=cv2.BORDER_WRAP)
            # 模糊操作
            im_blur = cv2.filter2D(im_orig, -1, mask)
            im_blur = im_blur[r:r + m, r:r + n]
            im_orig = im_orig[r:r + m, r:r + m]

            # ---- Downsample the image
            fx = 1. / K
            fy = 1. / K
            # 下采样
            im_down = cv2.resize(im_blur, (0, 0), fx=fx, fy=fy, interpolation=cv2.INTER_NEAREST)

            # ---- add noise -----
            noise_level = 10.0 / 255.0
            gauss = np.random.normal(0.0, noise_level, im_down.shape)
            im_noisy = im_down + gauss

            # im_noisy2 = np.clip(im_noisy, 0., 1.)
            # cv2.imwrite(f'noise_image_{imagename}_{noise_level*255}.png', im_noisy2 * 255.)

            psnr_final = 0.

            # ---- set options -----
            rho = 8 # 梯度下降那一步里的步长
            maxiter = 2
            result_iteration_psnr_ssim = {"Iteration":[], "PSNR":[], "SSIM":[],"res":[]}

            bicubic_img = cv2.resize(im_noisy, None, fx=K, fy=K, interpolation=cv2.INTER_CUBIC)
            # cv2.imwrite(f'superreimage_bicubic_{imagename}_{noise_level*255}.png', bicubic_img * 255.)
            psnr_ours = psnr(bicubic_img, im_orig)
            ssim_ours = compare_ssim(bicubic_img, im_orig, data_range=1.)
            result_iteration_psnr_ssim["Iteration"].append(0)
            result_iteration_psnr_ssim["PSNR"].append(psnr_ours)
            result_iteration_psnr_ssim["SSIM"].append(ssim_ours)
            result_iteration_psnr_ssim["res"].append(0)
            print('rho = {} - PNSR: {}, SSIM = {}'.format(rho, psnr_ours, ssim_ours))

            opts = dict(rho=rho, maxitr=maxiter, verbose=True)
            # ---- plug and play -----
            # plug-and-play 超分

            out = pnp_fbs_superresolution(im_noisy, im_orig, fx, fy, mask, **opts)

            imagename1 = f'ADDSGNLM_superresolution_rho={rho}_k={K}_{maxiter}_{imagename}'
            output_str = outdir_name + imagename1
            cv2.imwrite(output_str, out * 255.0)

            # ---- results ----
            res = pd.DataFrame(result_iteration_psnr_ssim)
            res.to_csv(f'ADDSGNLM_superresolution_rho={rho}_k={K}_res_{maxiter}_{imagename}.csv', index=False)
            psnr_ours = psnr(out, im_orig)
            ssim_ours = compare_ssim(out, im_orig, data_range=1.)
            print('rho = {} - PNSR: {}, SSIM = {}'.format(rho, psnr_ours, ssim_ours))
