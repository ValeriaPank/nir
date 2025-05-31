import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft
from mpl_toolkits.mplot3d import Axes3D
from numpy.fft import rfft, irfft
from numpy.fft import rfftfreq as np_rfftfreq
from scipy.interpolate import interp1d
import tkinter as tk
from tkinter import messagebox
import matplotlib
matplotlib.use('TkAgg')

class AcousticApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Акустическая система: обработка сигналов")
        
        self.params = {}
        self.create_widgets()
        
    def create_widgets(self):
        parameters = [
            ("Частота сигнала (Гц)", "50"),
            ("Амплитуда сигнала", "2"),
            ("Радиус антенны (м)", "1"),
            ("Количество приемников", "5"),
            ("Длина источника (м)", "200"),
            ("Скорость звука (м/с)", "1500"),
            ("Дистанция до источника (м)", "100"),
            ("Частота дискретизации (Гц)", "24000"),
            ("Длительность сигнала (с)", "1.0"),
            ("Коэффициент изменения высоты тона", "1.5"),
            ("Межэлементное расстояние d (м)", "0.5"),
            ("Граница сектора обзора phi (градусы)", "30"),
            ("Перекрытие блока (отсчёты)", "32")
        ]
        
        self.entries = []
        row_idx = 0
        for i, (label, default) in enumerate(parameters):
            tk.Label(self.root, text=label).grid(row=row_idx, column=0, padx=5, pady=5, sticky='e')
            entry = tk.Entry(self.root)
            entry.insert(0, default)
            entry.grid(row=row_idx, column=1, padx=5, pady=5)
            self.entries.append(entry)
            row_idx += 1

        # Параметры для сдвига частоты
        self.enable_freq_shift_var = tk.BooleanVar(value=False)
        tk.Checkbutton(self.root, text="Включить сдвиг частоты", variable=self.enable_freq_shift_var).grid(row=row_idx, column=0, columnspan=2, pady=2)
        row_idx += 1
        
        shift_parameters = [
            ("Нижняя граница сдвига (Гц)", "0", None),
            ("Верхняя граница сдвига (Гц)", " 12000", None),
            ("k (сдвиг в отсчетах)", "10", "k_entry_var")
        ]
        
        self.shift_entries_dict = {}
        self.k_entry_var = tk.StringVar()
        self.k_entry_var.trace_add("write", self._update_hz_shift_display_on_k_change)

        for label_text, default_val, entry_id in shift_parameters:
            tk.Label(self.root, text=label_text).grid(row=row_idx, column=0, padx=5, pady=5, sticky='e')
            if entry_id == "k_entry_var":
                entry = tk.Entry(self.root, textvariable=self.k_entry_var)
                entry.insert(0, default_val)
                self.k_entry_widget = entry
            else:
                entry = tk.Entry(self.root)
                entry.insert(0, default_val)
            entry.grid(row=row_idx, column=1, padx=5, pady=5)
            self.shift_entries_dict[label_text] = entry
            row_idx += 1

        self.hz_shift_display_var = tk.StringVar()
        self.hz_shift_display_var.set("(сдвиг в Гц будет здесь)")
        tk.Label(self.root, textvariable=self.hz_shift_display_var).grid(row=row_idx, column=0, columnspan=2, pady=2)
        row_idx += 1
            
        tk.Button(self.root, text="Рассчитать и визуализировать", 
                command=self.get_parameters).grid(row=row_idx, column=0, 
                                                columnspan=2, pady=10)
    
    def get_parameters(self):
        try:
            self.params = {
                'frequency': float(self.entries[0].get()),
                'amplitude': float(self.entries[1].get()),
                'radius': float(self.entries[2].get()),
                'num_receivers': int(self.entries[3].get()),
                'source_length': float(self.entries[4].get()),
                'c': float(self.entries[5].get()),
                'source_distance': float(self.entries[6].get()),
                'fd': float(self.entries[7].get()),
                'duration': float(self.entries[8].get()),
                'pitch_shift_factor': float(self.entries[9].get()),
                'd_spacing': float(self.entries[10].get()),
                'phi_angle': float(self.entries[11].get()),
                'block_overlap': int(self.entries[12].get()),
                # Параметры сдвига
                'enable_freq_shift': self.enable_freq_shift_var.get(),
                'shift_low_freq': float(self.shift_entries_dict["Нижняя граница сдвига (Гц)"].get()),
                'shift_high_freq': float(self.shift_entries_dict["Верхняя граница сдвига (Гц)"].get()),
                'shift_amount_bins': int(self.k_entry_var.get())
            }

            self._update_hz_shift_display()

            self.run_simulation_and_plotting()
            
        except ValueError:
            messagebox.showerror("Ошибка", "Проверьте правильность ввода данных!")

    def _update_hz_shift_display(self):
        try:
            enable_freq_shift_val = self.enable_freq_shift_var.get()
            k_val_str = self.k_entry_var.get()
            fd_val_str = self.entries[7].get()
            n_fft_val = 512

            if not k_val_str:
                self.hz_shift_display_var.set("(введите k)")
                return

            k_val = int(k_val_str)
            fd_val = float(fd_val_str)

            if enable_freq_shift_val:
                if fd_val > 0 and n_fft_val > 0:
                    freq_resolution = fd_val / n_fft_val
                    calculated_hz_shift = k_val * freq_resolution
                    self.hz_shift_display_var.set(f"Сдвиг k={k_val} ≈ {calculated_hz_shift:.2f} Гц")
                else:
                    self.hz_shift_display_var.set("Для расчета Гц: fd и n_fft > 0")
            else:
                self.hz_shift_display_var.set("Сдвиг частоты выключен")

        except ValueError:
            self.hz_shift_display_var.set("(ошибка ввода k или fd)")
        except Exception as e:
            self.hz_shift_display_var.set("(ошибка обновления)")

    def _update_hz_shift_display_on_k_change(self, *args):
        self._update_hz_shift_display()

    def _extract_parameters(self, p_dict):
        fd = p_dict['fd']
        dt = 1 / fd
        t = np.arange(0, p_dict['duration'] + dt, dt)
        c_val = p_dict['c']
        source_length = p_dict['source_length']
        source_distance = p_dict['source_distance']
        n_fft = 512
        frequency = p_dict['frequency']
        amplitude = p_dict['amplitude']
        radius = p_dict['radius']
        num_receivers = p_dict['num_receivers']
        pitch_shift_factor = p_dict['pitch_shift_factor']
        d_spacing = p_dict['d_spacing']
        phi_angle_deg = p_dict['phi_angle']
        enable_freq_shift = p_dict['enable_freq_shift']
        shift_low_freq = p_dict['shift_low_freq']
        shift_high_freq = p_dict['shift_high_freq']
        shift_amount_bins = p_dict['shift_amount_bins']
        block_overlap = p_dict['block_overlap']
        
        return (fd, dt, t, c_val, source_length, source_distance, n_fft, frequency, amplitude, radius, 
                num_receivers, pitch_shift_factor, d_spacing, phi_angle_deg,
                enable_freq_shift, shift_low_freq, shift_high_freq, shift_amount_bins, block_overlap)

    def _calculate_overlap_N(self, M, d, phi_deg, c, fd, n_fft):
        if M <= 1:
            return 0
        phi_rad = np.deg2rad(phi_deg)
        tau_max = (M - 1) * d * np.sin(phi_rad) / c
        N = int(tau_max * fd)
        
        N = max(0, N)
        N = min(N, n_fft - 1)
        return N

    def _calculate_geometry(self, radius, num_receivers, source_distance, source_length):
        if num_receivers == 1:
            receiver_x = np.array([0.0])
            receiver_y = np.array([0.0])
        else:
            angles = np.linspace(3 * np.pi / 2, 5 * np.pi / 2, num_receivers)
            receiver_x = radius * np.cos(angles)
            receiver_y = radius * np.sin(angles)
        source_x_coord = source_distance
        source_y_coords_vis = np.linspace(-source_length / 2, source_length / 2, 20)
        return receiver_x, receiver_y, source_x_coord, source_y_coords_vis

    def _calculate_delays(self, receiver_x, receiver_y, source_x_coord, c_val, radius, source_distance, phi_angle_deg=0):
        # phi_angle_deg — угол прихода волны в градусах (можно брать из параметров)
        phi_angle_deg = 0
        phi = phi_angle_deg * np.pi / 180
        print(phi)  # Должно быть 0.0
        num_receivers = len(receiver_x)
        # phi_m — углы положения приемников на окружности
        if num_receivers == 1:
            phi_m = np.array([0.0])
        else:
            phi_m = np.linspace(3 * np.pi / 2, 5 * np.pi / 2, num_receivers)
        # Формула задержки
        delays_flat_wave_raw = -1 / c_val * radius * (np.cos(phi) * np.cos(phi_m) + np.sin(phi) * np.sin(phi_m))
        if delays_flat_wave_raw.size > 0:
            min_flat_delay = np.min(delays_flat_wave_raw)
            delays_flat_wave_normalized = delays_flat_wave_raw - min_flat_delay
        else:
            delays_flat_wave_normalized = delays_flat_wave_raw
        print(f"ЗАДЕРЖКИ (по формуле с phi): {delays_flat_wave_normalized}")
        # Модель 2 (оставим без изменений)
        source_y_for_delay_calc_dist = receiver_y 
        distances = np.sqrt((receiver_x - source_x_coord)**2 + (receiver_y - source_y_for_delay_calc_dist)**2)
        distances_compensated = distances - (source_distance - radius) 
        delays_distance_based_raw = distances_compensated / c_val
        if delays_distance_based_raw.size > 0:
            min_distance_delay = np.min(delays_distance_based_raw)
            delays_distance_based_normalized = delays_distance_based_raw - min_distance_delay
        else:
            delays_distance_based_normalized = delays_distance_based_raw
        print(f"ЗАДЕРЖКИ (Модель 2: Расстояние до линейного источника, НОРМАЛИЗОВАННЫЕ): {delays_distance_based_normalized}")
        return delays_flat_wave_normalized 

    def _generate_multichannel_signals(self, t_array, amplitude, frequency, delays_array):
        signals = np.array([amplitude * np.sin(2 * np.pi * frequency * (t_array - delay)) for delay in delays_array])
        return signals

    def _perform_coherent_summation(self, signals_array, delays_array, n_fft, dt, block_overlap,
                                    enable_freq_shift=False, shift_low_freq=0,
                                    shift_high_freq=0, shift_amount_bins=0, fd=0, window_name=None):
        total_length = signals_array.shape[1]
        block_processing_length = n_fft - block_overlap
        summed_signal = np.zeros(total_length)
        frequencies_block = np.fft.fftfreq(n_fft, d=dt)

        for i, start_index in enumerate(range(0, total_length, block_processing_length)):
            end_index_fft = start_index + n_fft
            signals_block_temp = np.zeros((signals_array.shape[0], n_fft))
            actual_slice_len = max(0, min(n_fft, total_length - start_index))
            if actual_slice_len <= 0:
                break
            signals_block_temp[:, :actual_slice_len] = signals_array[:, start_index : start_index + actual_slice_len]
            
            # 1. FFT для каждого канала
            signals_fft_block = np.array([fft(signal_frame, n_fft) for signal_frame in signals_block_temp])
            
            # 2. Компенсация задержек (фазовый множитель)
            compensated_signals_fft = signals_fft_block * np.exp(1j * 2 * np.pi * delays_array[:, None] * frequencies_block)
            
            # 3. Суммирование каналов
            current_summed_fft = np.sum(compensated_signals_fft, axis=0)
            
            # 4. Фильтрация (окно) — только к итоговому спектру!
            if window_name and fd > 0 and (window_name in ["hanning", "hamming", "blackman", "rectangular"]):
                positive_freq_indices = np.where(frequencies_block >= 0)[0]
                low_idx_orig_in_positive = np.searchsorted(frequencies_block[positive_freq_indices], shift_low_freq)
                high_idx_orig_in_positive = np.searchsorted(frequencies_block[positive_freq_indices], shift_high_freq, side='right')
                
                if high_idx_orig_in_positive > low_idx_orig_in_positive:
                    K = high_idx_orig_in_positive - low_idx_orig_in_positive
                    if window_name == "hanning":
                        fo = np.hanning(K)
                    elif window_name == "hamming":
                        fo = np.hamming(K)
                    elif window_name == "blackman":
                        fo = np.blackman(K)
                    elif window_name == "rectangular":
                        fo = np.ones(K)
                    window_multiplier = np.zeros(n_fft, dtype=float)
                    positive_band_actual_indices = positive_freq_indices[low_idx_orig_in_positive:high_idx_orig_in_positive]
                    if len(positive_band_actual_indices) > 0:
                        print(f"Окно применяется к частотам: {frequencies_block[positive_band_actual_indices][0]} - {frequencies_block[positive_band_actual_indices][-1]}")
                    window_multiplier[positive_band_actual_indices] = fo
                    for j in range(K):
                        actual_pos_idx = positive_band_actual_indices[j]
                        if actual_pos_idx != 0:
                            is_nyquist = (n_fft % 2 == 0) and (actual_pos_idx == n_fft // 2)
                            if not is_nyquist:
                                symmetric_neg_idx = n_fft - actual_pos_idx
                                if 0 <= symmetric_neg_idx < n_fft:
                                    window_multiplier[symmetric_neg_idx] = fo[j]
                    current_summed_fft *= window_multiplier
                else:
                    current_summed_fft[:] = 0.0

            # 5. Сдвиг полосы (только к итоговому спектру)
            if enable_freq_shift and fd > 0:
                shifted_summed_fft = current_summed_fft.copy()
                positive_freq_indices = np.where(frequencies_block >= 0)[0]
                low_idx_orig_in_positive = np.searchsorted(frequencies_block[positive_freq_indices], shift_low_freq)
                high_idx_orig_in_positive = np.searchsorted(frequencies_block[positive_freq_indices], shift_high_freq, side='right')
                if high_idx_orig_in_positive > low_idx_orig_in_positive:
                    band_to_shift_positive = current_summed_fft[positive_freq_indices[low_idx_orig_in_positive:high_idx_orig_in_positive]].copy()
                    shifted_summed_fft[positive_freq_indices[low_idx_orig_in_positive:high_idx_orig_in_positive]] = 0
                    for k_idx in range(low_idx_orig_in_positive, high_idx_orig_in_positive):
                        original_positive_idx = positive_freq_indices[k_idx]
                        if original_positive_idx != 0 and original_positive_idx * 2 != n_fft:
                            shifted_summed_fft[n_fft - original_positive_idx] = 0
                    for k_orig_band, k_target_positive_idx_offset in enumerate(range(low_idx_orig_in_positive + shift_amount_bins, high_idx_orig_in_positive + shift_amount_bins)):
                        if 0 <= k_target_positive_idx_offset < len(positive_freq_indices):
                            target_positive_idx = positive_freq_indices[k_target_positive_idx_offset]
                            shifted_summed_fft[target_positive_idx] += band_to_shift_positive[k_orig_band]
                            if target_positive_idx != 0 and target_positive_idx * 2 != n_fft:
                                shifted_summed_fft[n_fft - target_positive_idx] += np.conj(band_to_shift_positive[k_orig_band])
                current_summed_fft = shifted_summed_fft

            # 6. IFFT
            print(f"current_summed_fft: {current_summed_fft}")
            summed_block_time = ifft(current_summed_fft).real
            write_len = min(block_processing_length, total_length - start_index)
            if write_len <= 0:
                break
            summed_signal[start_index : start_index + write_len] = summed_block_time[:write_len]
        return summed_signal

    def _compute_signal_spectra(self, summed_sig, time_step):
        n_fft_1 = 2048
        n_fft_2 = 4096

        summed_spectrum_2048 = np.abs(fft(summed_sig, n_fft_1))
        spectrum_frequencies_2048 = np.fft.fftfreq(n_fft_1, d=time_step)

        summed_spectrum_4096 = np.abs(fft(summed_sig, n_fft_2))
        spectrum_frequencies_4096 = np.fft.fftfreq(n_fft_2, d=time_step)

        return (summed_spectrum_2048, spectrum_frequencies_2048, summed_spectrum_4096, spectrum_frequencies_4096)

    def _create_visualization_plots(self, fd_hz, t_array_base, n_fft_val, initial_signals_arr, 
                                    summed_signal_hanning, summed_signal_hamming, summed_signal_blackman, summed_signal_rectangular, summed_signal_no_window, 
                                    spectra_data_hanning, spectra_data_hamming, spectra_data_blackman, spectra_data_rectangular, spectra_data_no_window, num_of_receivers):
        sum_spec_hanning, sum_freq_hanning, sum_spec_l_hanning, sum_freq_l_hanning = spectra_data_hanning
        sum_spec_hamming, sum_freq_hamming, sum_spec_l_hamming, sum_freq_l_hamming = spectra_data_hamming
        sum_spec_blackman, sum_freq_blackman, sum_spec_l_blackman, sum_freq_l_blackman = spectra_data_blackman
        sum_spec_rectangular, sum_freq_rectangular, sum_spec_l_rectangular, sum_freq_l_rectangular = spectra_data_rectangular
        sum_spec_no_window, sum_freq_no_window, sum_spec_l_no_window, sum_freq_l_no_window = spectra_data_no_window
        
        fig = plt.figure(figsize=(14, 8))
        
        # Plot 1: Summed signals
        plt.subplot(2, 2, 1)
        time_axis_summed = np.arange(len(summed_signal_hanning)) / fd_hz
        plt.plot(time_axis_summed, summed_signal_hanning, label='Суммарный (Hanning)', color='blue', linewidth=1)
        plt.plot(time_axis_summed, summed_signal_hamming, label='Суммарный (Hamming)', color='green', linewidth=1, linestyle='-.')
        plt.plot(time_axis_summed, summed_signal_blackman, label='Суммарный (Blackman)', color='purple', linewidth=1, linestyle=':')
        plt.plot(time_axis_summed, summed_signal_rectangular, label='Суммарный (Rectangular)', color='cyan', linewidth=1, linestyle='--')
        plt.plot(time_axis_summed, summed_signal_no_window, label='Суммарный (без окна)', color='red', linestyle='--', linewidth=1)
        plt.xlabel("Время (с)")
        plt.ylabel("Амплитуда")
        plt.title("Суммарные компенсированные сигналы")
        plt.legend()
        plt.grid()
        
        # Plot 2: Initial signals
        plt.subplot(2, 2, 2)
        display_signal_length = min(n_fft_val * 2, initial_signals_arr.shape[1])
        time_axis_signals = t_array_base[:display_signal_length]
        for i, signal_item in enumerate(initial_signals_arr):
            plt.plot(time_axis_signals, signal_item[:display_signal_length], label=f'Сигнал {i+1}')
        plt.xlabel("Время (с)")
        plt.ylabel("Амплитуда")
        plt.title("Изначальные сигналы (начало)")
        plt.legend(loc='lower right')
        plt.grid()
        ax = fig.add_subplot(2, 2, 3, projection='3d')
        display_3d_length = min(n_fft_val, initial_signals_arr.shape[1])
        time_axis_3d = t_array_base[:display_3d_length]
        X, Y = np.meshgrid(time_axis_3d, np.arange(num_of_receivers))
        for i in range(num_of_receivers):
            ax.plot(time_axis_3d, np.full_like(time_axis_3d, i), initial_signals_arr[i, :display_3d_length], label=f'Приемник {i+1}')
        ax.set_xlabel("Время (с)")
        ax.set_ylabel("Приемники")
        ax.set_zlabel("Амплитуда")
        ax.set_title("3D сигналы (начало)")
        ax.view_init(elev=30, azim=-60)
        
        # Plot 4: Spectra (2048 points)
        plt.subplot(2, 2, 4)
        
        epsilon = 1e-12  # To avoid log10(0)

        positive_freqs_2048_hanning = sum_freq_hanning > 0
        plt.plot(sum_freq_hanning[positive_freqs_2048_hanning], 20 * np.log10(sum_spec_hanning[positive_freqs_2048_hanning] + epsilon), color='blue', label='Спектр (Hanning, 2048)')
        
        positive_freqs_2048_hamming = sum_freq_hamming > 0
        plt.plot(sum_freq_hamming[positive_freqs_2048_hamming], 20 * np.log10(sum_spec_hamming[positive_freqs_2048_hamming] + epsilon), color='green', linestyle='-.', label='Спектр (Hamming, 2048)')
        
        positive_freqs_2048_blackman = sum_freq_blackman > 0
        plt.plot(sum_freq_blackman[positive_freqs_2048_blackman], 20 * np.log10(sum_spec_blackman[positive_freqs_2048_blackman] + epsilon), color='purple', linestyle=':', label='Спектр (Blackman, 2048)')
        
        positive_freqs_2048_rectangular = sum_freq_rectangular > 0
        plt.plot(sum_freq_rectangular[positive_freqs_2048_rectangular], 20 * np.log10(sum_spec_rectangular[positive_freqs_2048_rectangular] + epsilon), color='cyan', linestyle='--', label='Спектр (Rectangular, 2048)')
        
        positive_freqs_2048_no_window = sum_freq_no_window > 0
        plt.plot(sum_freq_no_window[positive_freqs_2048_no_window], 20 * np.log10(sum_spec_no_window[positive_freqs_2048_no_window] + epsilon), color='red', linestyle='--', label='Спектр (без окна, 2048)')
        
        # Optional: Plot for 4096 points if needed, example for Hanning:
        # positive_freqs_4096_hanning = sum_freq_l_hanning > 0
        # plt.plot(sum_freq_l_hanning[positive_freqs_4096_hanning], 20 * np.log10(sum_spec_l_hanning[positive_freqs_4096_hanning] + epsilon), color='cyan', linestyle=':', label='Спектр (Hanning, 4096)')

        plt.xlabel("Частота (Гц)")
        plt.ylabel("Амплитуда (дБ)")
        plt.title("Спектры суммарного сигнала (2048 точек)")
        plt.legend()
        plt.grid()
        plt.xlim(0, fd_hz / 2)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

    def run_simulation_and_plotting(self): 
        plt.close('all') 
        p = self.params
        (fd, dt, t, c_val, source_length, source_distance, n_fft, frequency, amplitude, radius, 
         num_receivers, pitch_shift_factor, d_spacing, phi_angle_deg,
         enable_freq_shift, shift_low_freq, shift_high_freq, shift_amount_bins, block_overlap) = self._extract_parameters(p)
        
        if enable_freq_shift and fd > 0 and n_fft > 0:
            freq_resolution = fd / n_fft
            print(f"Сдвиг частоты включен: k={shift_amount_bins} отсчетов, что примерно равно {shift_amount_bins * freq_resolution:.2f} Гц")

        block_overlap = 32
        print(f"Рассчитанная величина перекрытия (N) для block_overlap: {block_overlap} отсчетов")

        receiver_x, receiver_y, source_x_coord, _ = self._calculate_geometry(radius, num_receivers, source_distance, source_length)
        delays = self._calculate_delays(receiver_x, receiver_y, source_x_coord, c_val, radius, source_distance, phi_angle_deg)

        # --- Проверка задержек относительно блока ---
        max_delay_seconds = np.max(np.abs(delays))
        max_delay_samples = int(np.ceil(max_delay_seconds * fd))
        print(f"Максимальная задержка: {max_delay_samples} отсчётов")
        print(f"Размер блока: {n_fft}, перекрытие: {block_overlap}, полезная длина блока: {n_fft - block_overlap}")
        if block_overlap < max_delay_samples:
            print("ВНИМАНИЕ: перекрытие меньше максимальной задержки! Возможна потеря данных.")
        else:
            print("Параметры выбраны корректно: потерь не будет.")

        initial_signals = self._generate_multichannel_signals(t, amplitude, frequency, delays)
        
        # Signal with Hanning window (if frequency shifting is enabled)
        summed_signal_hanning = self._perform_coherent_summation(
            initial_signals, delays, n_fft, dt, block_overlap,
            enable_freq_shift, shift_low_freq, shift_high_freq, shift_amount_bins, fd,
            window_name="hanning" if enable_freq_shift else None
        )

        # Signal with Hamming window (if frequency shifting is enabled)
        summed_signal_hamming = self._perform_coherent_summation(
            initial_signals, delays, n_fft, dt, block_overlap,
            enable_freq_shift, shift_low_freq, shift_high_freq, shift_amount_bins, fd,
            window_name="hamming" if enable_freq_shift else None
        )

        # Signal with Blackman window (if frequency shifting is enabled)
        summed_signal_blackman = self._perform_coherent_summation(
            initial_signals, delays, n_fft, dt, block_overlap,
            enable_freq_shift, shift_low_freq, shift_high_freq, shift_amount_bins, fd,
            window_name="blackman" if enable_freq_shift else None
        )

        # Signal with Rectangular window (if frequency shifting is enabled)
        summed_signal_rectangular = self._perform_coherent_summation(
            initial_signals, delays, n_fft, dt, block_overlap,
            enable_freq_shift, shift_low_freq, shift_high_freq, shift_amount_bins, fd,
            window_name="rectangular" if enable_freq_shift else None
        )

        # Signal without any window (corresponds to old "without filter")
        summed_signal_no_window = self._perform_coherent_summation(
            initial_signals, delays, n_fft, dt, block_overlap,
            enable_freq_shift, shift_low_freq, shift_high_freq, shift_amount_bins, fd,
            window_name=None
        )
        
        spectra_data_hanning = self._compute_signal_spectra(summed_signal_hanning, dt)
        spectra_data_hamming = self._compute_signal_spectra(summed_signal_hamming, dt)
        spectra_data_blackman = self._compute_signal_spectra(summed_signal_blackman, dt)
        spectra_data_rectangular = self._compute_signal_spectra(summed_signal_rectangular, dt)
        spectra_data_no_window = self._compute_signal_spectra(summed_signal_no_window, dt)

        # --- Явный вывод значений спектра вне окна ---
        sum_spec_hanning, sum_freq_hanning, _, _ = spectra_data_hanning
        mask_low = sum_freq_hanning < 4000
        mask_high = sum_freq_hanning > 8000
        print(f"Максимум спектра Hanning до 4000 Гц: {np.max(np.abs(sum_spec_hanning[mask_low]))}")
        print(f"Максимум спектра Hanning после 8000 Гц: {np.max(np.abs(sum_spec_hanning[mask_high]))}")

        self._create_visualization_plots(fd, t, n_fft, initial_signals, 
                                         summed_signal_hanning, summed_signal_hamming, summed_signal_blackman, summed_signal_rectangular, summed_signal_no_window, 
                                         spectra_data_hanning, spectra_data_hamming, spectra_data_blackman, spectra_data_rectangular, spectra_data_no_window, num_receivers)

if __name__ == "__main__":
    root = tk.Tk()
    app = AcousticApp(root)
    root.mainloop()
