import numpy as np


def shift_schedule(alpha_cumprods, shift_ratio):
    # shift_ratio = original_resolution (512) ** 2 / (new_resolution ** 2 * n_images)
    sigma_cp = 1. - alpha_cumprods
    snr = alpha_cumprods / sigma_cp   

    # log_snr_shifted = np.log(snr) - np.log(shift_ratio)
    log_snr_shifted = np.log(snr) + np.log(shift_ratio)  
    alpha_shifted = np.exp(log_snr_shifted) / (1 + np.exp(log_snr_shifted))
    betas_shifted = 1 - np.concatenate([[1], (alpha_shifted[1:] / alpha_shifted[:-1])])

    return alpha_shifted, betas_shifted


# https://arxiv.org/pdf/2305.08891
def enforce_zero_terminal_snr(betas):
    # Convert betas to alphas_bar_sqrt
    alphas = 1 - betas
    alphas_bar = alphas.cumprod(0)
    alphas_bar_sqrt = np.sqrt(alphas_bar)

    # Store old values.
    alphas_bar_sqrt_0 = alphas_bar_sqrt[0].copy()
    alphas_bar_sqrt_T = alphas_bar_sqrt[-1].copy()
    # Shift so last timestep is zero.
    alphas_bar_sqrt -= alphas_bar_sqrt_T
    # Scale so first timestep is back to old value.
    alphas_bar_sqrt *= alphas_bar_sqrt_0 / (alphas_bar_sqrt_0 - alphas_bar_sqrt_T)

    # Convert alphas_bar_sqrt to betas
    alphas_bar = alphas_bar_sqrt ** 2
    alphas = alphas_bar[1:] / alphas_bar[:-1]
    alphas = np.concatenate([alphas_bar[0:1], alphas])
    betas = 1 - alphas
    return betas