import matplotlib.pyplot as plt
import numpy as np

def plot_spectrum(freqs: np.ndarray, spectrum: np.ndarray):
    fig, ax = plt.subplots(figsize=(10, 5))
    # mask = freqs >= 0

    # # переводим в дБ, но нули оставляем как 0
    # spectrum_db = 20 * np.log10(np.maximum(spectrum, 1e-12))
    # spectrum_db[spectrum_db < 0] = 0


    ax.plot(freqs, spectrum, color="red")
    ax.set_xlabel("Частота (Гц)")
    ax.set_ylabel("Амплитуда (дБ)")
    ax.set_title("Спектр суммарного сигнала")
    ax.grid(True)
    return fig
