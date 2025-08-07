import numpy as np

def media_global(hist_norm):
    return np.sum(i * hist_norm[i] for i in range(256))

def prob(hist_norm, t):
    w0 = np.sum(hist_norm[:t+1])
    w1 = np.sum(hist_norm[t+1:])
    return w0, w1

def media_classes(hist_norm, t):
    w0, w1 = prob(hist_norm, t)

    sum0 = np.sum(i * hist_norm[i] for i in range(t + 1))
    sum1 = np.sum(i * hist_norm[i] for i in range(t + 1, 256))

    mu0 = sum0 / w0 if w0 > 0 else 0
    mu1 = sum1 / w1 if w1 > 0 else 0

    return mu0, mu1

def find_optimal_threshold(hist_norm):
    max_variance = -1
    optimal_threshold = 0
    media_global_val = media_global(hist_norm)

    print(f"Média Global do Histograma: {media_global_val}")

    for t in range(256):
        w0, w1 = prob(hist_norm, t)

        if w0 == 0 or w1 == 0:
            continue

        mu0, mu1 = media_classes(hist_norm, t)

        # calculo de varianca
        variance = w0 * w1 * (mu0 - mu1) ** 2 + media_global_val * (w0 + w1)

        if variance > max_variance:
            max_variance = variance
            optimal_threshold = t

    return optimal_threshold
