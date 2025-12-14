from core.geometry import AntennaArray
from core.signal_generation import SignalGenerator
from core.signal_processing import SignalProcessor
from core.parameters import SimulationParameters
from scipy.fftpack import ifft
def freq_to_index(f, fd, n_fft):
    return int(round(f / fd * n_fft))

def run_calculation(params: SimulationParameters):
    # --- Геометрия ---
    array = AntennaArray(num_receivers=params.num_receivers,
                         radius=params.radius,
                         start_angle_deg=270)
    x, y = array.get_coordinates()
    delays = array.compute_delays(phi_angle_deg=params.phi_angle, c=params.c)
    low_bin = freq_to_index(params.shift_low_freq, params.fd, params.n_fft)
    high_bin = freq_to_index(params.shift_high_freq, params.fd, params.n_fft)
    shift_low_bin = low_bin - params.shift_amount_bins
    shift_high_bin  = high_bin - params.shift_amount_bins
    # --- Генерация сигналов ---
    generator = SignalGenerator(params)
    signals, t = generator.generate(delays)

    # --- Обработка сигналов ---
    processor = SignalProcessor(
        fd=params.fd,
        n_fft=params.n_fft,
        block_overlap=params.block_overlap,
        window=params.band_window
    )
    if params.enable_shift:
        band_window_matrix = processor.make_band_window(
            low_bin=low_bin,
            high_bin=high_bin,
            margin_bins=12,         
            num_receivers=params.num_receivers,
            window_type=params.band_window
        )
        freq_mask = band_window_matrix[0]
        # Оценки длительности ИПХ
        L_thr = processor.estimate_lfilter_threshold(freq_mask, settle_db=-60)
        L_eng = processor.estimate_lfilter_energy(freq_mask, energy_ratio=0.999)
        print(f"L_filter threshold(-60 dB): {L_thr} samples")
        print(f"L_filter energy(99.9%):     {L_eng} samples")
    else:
        band_window_matrix = None
    summed_signal, shifted_signal = processor.coherent_sum(
        signals=signals,
        delays=delays,
        low_bin=low_bin,
        high_bin=high_bin,
        shift_low_bin=shift_low_bin,
        margin_bins=12,
        band_window_matrix=band_window_matrix,
        enable_shift=params.enable_shift
    )
    if params.enable_shift:
        freqs_shifted, spectrum_shifted = processor.compute_spectrum(shifted_signal)
    else:
        print(summed_signal)
        shifted_signal = None
        freqs_shifted, spectrum_shifted = None, None

    freqs, spectrum = processor.compute_spectrum(summed_signal)
        
    # --- Отладочный вывод ---
    print("Задержки:")
    for i, d in enumerate(delays, start=1):
        print(f"Приёмник {i}: {d:.6f} сек")

    # --- Возврат данных для GUI ---
    return x, y, signals, t, summed_signal, freqs, spectrum, shifted_signal, spectrum_shifted, freqs_shifted
