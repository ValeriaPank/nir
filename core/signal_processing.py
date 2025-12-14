import numpy as np
from numpy.fft import fft, ifft, rfft, irfft, fftfreq, rfftfreq

class SignalProcessor:
    """
    Класс для обработки многоканальных сигналов.
    
    Основные возможности:
    --------------------
    1. Когерентное суммирование (coherent_sum):
       - Сигналы с разных приёмников суммируются с учётом задержек.
       - Используется прямоугольное окно (rectangular), чтобы не терять амплитуду.
       - Поддерживается перекрытие блоков (overlap).
    
    2. Спектральный анализ (compute_spectrum):
       - Выполняется FFT суммарного сигнала.
       - Перед преобразованием сигнал умножается на окно (по умолчанию rectangular).
       - Возвращает массив частот и спектр (амплитуды).
    
    3. Частотная фильтрация и сдвиг (apply_band_window_and_shift):
       - Выделяется диапазон частот [shift_low_freq, shift_high_freq].
       - К этому диапазону применяется выбранное окно (Hanning/Hamming/Blackman).
       - Диапазон сдвигается на shift_amount_bins отсчётов.
       - Возвращает модифицированный спектр, из которого можно восстановить сигнал через IFFT.
    """
    
    def __init__(self, fd: float, n_fft: int = 1024, block_overlap: int = 0, window: str = "rectangular"):
        self.fd = fd
        self.n_fft = n_fft
        self.block_overlap = block_overlap
        self.window = window  # окно для базовой обработки (обычно rectangular)

    def _get_window(self, length: int, window_type: str = None):
        """
        Возвращает окно указанного типа.
        Если тип не задан, используется self.window.
        """
        wtype = window_type if window_type else self.window
        if wtype == "hanning":
            return np.hanning(length)
        elif wtype == "hamming":
            return np.hamming(length)
        elif wtype == "blackman":
            return np.blackman(length)
        else:  # rectangular
            return np.ones(length)

    def make_band_window(self,
                         low_bin: int,
                         high_bin: int,
                         margin_bins: int,
                         num_receivers: int,
                         window_type: str = None):
        # пустая маска по частотам
        freq_mask = np.zeros(self.n_fft)

        # длина окна с учётом напуска
        window_length = (high_bin - low_bin) + 2 * margin_bins + 1

        # формируем окно нужного типа
        bandpass_window = self._get_window(window_length, window_type=window_type)
        
        # bandpass_window = self._get_window(window_length, window_type="rectangular")

        # вставляем окно в маску
        freq_mask[low_bin - margin_bins : high_bin + margin_bins + 1] = bandpass_window
        print(freq_mask[low_bin - margin_bins : high_bin + margin_bins + 1])
        # копируем маску для всех каналов
        window_matrix = np.tile(freq_mask, (num_receivers, 1))

        return window_matrix
    
    def coherent_sum(self,
                     signals: np.ndarray,
                     delays: np.ndarray,
                     low_bin: int,
                     high_bin: int,
                     shift_low_bin: int,
                     margin_bins: int,
                     band_window_matrix: np.ndarray,
                     enable_shift: bool = False):
        """
        Когерентное суммирование сигналов с перекрытием блоков.
        signals: [каналы, время]
        delays: [каналы]
        """
        total_length = signals.shape[1]
        step = self.n_fft - self.block_overlap
        block_start_index = self.block_overlap//2 + 1
        summed = np.zeros(total_length)
        shifted_signal = np.zeros(total_length)
        df = self.fd/self.n_fft
        fk = df*np.arange(self.n_fft)
        freqs = np.concatenate([fk[:self.n_fft//2+1], fk[self.n_fft//2+1:]-self.fd])

        for start in range(0, total_length - self.n_fft + 1, step):
            block = signals[:, start:start+self.n_fft]


            # FFT каждого канала
            block_fft = fft(block, self.n_fft, axis=1)
            if enable_shift:
                block_fft = block_fft * band_window_matrix
            kolf = np.exp(1j * 2*np.pi*delays[:, None] * freqs[None,:])
            compensated = block_fft * kolf
            # суммирование каналов
            summed_fft = np.sum(compensated, axis=0)
            summed_block = ifft(summed_fft, self.n_fft)
            summed[start:start+step] = np.real(summed_block[block_start_index:block_start_index+step])

            if enable_shift:
                filtered_fft = np.zeros_like(summed_fft)
                filtered_fft[low_bin-margin_bins : high_bin+margin_bins+1] = summed_fft[low_bin-margin_bins : high_bin+margin_bins+1]
                band_length = (high_bin - low_bin) + 1
                shifted_fft = np.zeros_like(summed_fft)
                # band_length = (high_bin - low_bin) + 2*margin_bins + 1 
                shifted_fft[shift_low_bin : shift_low_bin + band_length] = filtered_fft[low_bin : high_bin+1]
                shifted_summed_block = ifft(shifted_fft, self.n_fft)
                shifted_signal[start:start+step] =  np.real(shifted_summed_block[block_start_index:block_start_index+step])

        if enable_shift:
            return summed, shifted_signal
        else:
            return summed, None

    def compute_spectrum(self, signal: np.ndarray):
        """
        FFT спектр суммарного сигнала.
        """
        # spectrum = np.abs(fft(signal, self.n_fft))
        # freqs = self.fd/self.n_fft*np.arange(self.n_fft)
        # spectrum_db = 20 * np.log10(spectrum)
        spectrum = np.abs(fft(signal, self.n_fft))
        freqs = self.fd/self.n_fft * np.arange(self.n_fft//2 + 1)
        spectrum_db = 20 * np.log10(spectrum[:self.n_fft//2 + 1])

        return freqs, spectrum_db
    def shift_band(self, signal, low_bin, high_bin, shift_low_bin, window, margin_bins):
        mask_for_one = window[0]
        spectrum = fft(signal, self.n_fft)
        filtered_spectrum = spectrum * mask_for_one
        filtered_signal = np.real(ifft(filtered_spectrum, self.n_fft))
        return filtered_signal
    def estimate_lfilter_threshold(self, freq_mask: np.ndarray, settle_db: float = -60, guard: int = 8) -> int:
        """
        Оценка эффективной длительности ИПХ по порогу (в дБ).
        Возвращает длину в отсчётах.
        """
        # ИПХ фильтра: iFFT маски (циклическая природа)
        h = np.fft.ifft(freq_mask)
        # центрируем главный импульс для симметричной оценки
        h_abs = np.abs(np.fft.fftshift(h))
        h_norm = h_abs / (h_abs.max() + 1e-12)
        thr = 10 ** (settle_db / 20.0)
        N = len(h_norm)
        mid = N // 2

        # идём от центра влево/вправо, пока локальные окна > порога
        left = mid
        while left > 0 and np.max(h_norm[max(0, left-guard):left]) > thr:
            left -= 1

        right = mid
        while right < N-1 and np.max(h_norm[right:min(N, right+guard)]) > thr:
            right += 1

        return right - left + 1

    def estimate_lfilter_energy(self, freq_mask: np.ndarray, energy_ratio: float = 0.999) -> int:
        """
        Оценка эффективной длительности ИПХ по доле энергии (например, 99.9%).
        Возвращает длину в отсчётах.
        """
        h = np.fft.ifft(freq_mask)
        e = np.abs(np.fft.fftshift(h))**2
        N = len(e)
        mid = N // 2
        total = e.sum()
        acc = e[mid]
        L = 1
        left = mid - 1
        right = mid + 1
        while acc / total < energy_ratio and (left >= 0 or right < N):
            if left >= 0:
                acc += e[left]
                left -= 1
                L += 1
            if acc / total >= energy_ratio:
                break
            if right < N:
                acc += e[right]
                right += 1
                L += 1
        return L