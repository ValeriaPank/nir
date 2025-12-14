import numpy as np
import scipy.io.wavfile as wav
import scipy.io as sio

def load_signal_from_wav(filepath):
    fd, data = wav.read(filepath)
    if data.dtype == np.int16:
        signal = data.astype(np.float32) / 32767.0
    elif data.dtype == np.int32:
        signal = data.astype(np.float32) / 2147483647.0
    else:
        # если уже float (например, 32‑битный WAV)
        signal = data.astype(np.float32)
    frequency_est = estimate_frequency(signal, fd)
    return signal, fd, frequency_est

def estimate_frequency(signal, fd):
    # считаем спектр
    spectrum = np.abs(np.fft.rfft(signal))
    freqs = np.fft.rfftfreq(len(signal), 1/fd)
    # индекс максимума
    peak_idx = np.argmax(spectrum)
    return freqs[peak_idx]

def load_signal_from_mat(filepath, key='signal'):
    mat = sio.loadmat(filepath)
    signal = mat[key].squeeze()
    fd = float(np.array(mat.get('fd', [24000]))[0])
    frequency_est = estimate_frequency(signal, fd)
    return signal, fd, frequency_est

def load_signal_from_csv(filepath):
    data = np.loadtxt(filepath, delimiter=",")
    # если есть два столбца: [t, signal]
    if data.ndim > 1 and data.shape[1] >= 2:
        t = data[:,0]
        signal = data[:,1]
        fd = 1.0 / (t[1] - t[0])  # оцениваем частоту дискретизации
    else:
        signal = data
        fd = 24000  # дефолт
    frequency_est = estimate_frequency(signal, fd)
    return signal, fd, frequency_est

def load_signal_from_txt(filepath):
    signal = np.loadtxt(filepath)
    fd = 24000  # дефолт или задаётся отдельно
    frequency_est = estimate_frequency(signal, fd)
    return signal, fd, frequency_est
