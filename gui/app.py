import tkinter as tk
from tkinter import ttk
import numpy as np
from core.parameters import SimulationParameters
from core.run_calculation import run_calculation
from visualization.signal_plot import plot_signals
from visualization.summed_signal_plot import plot_summed_signal
from visualization.spectrum_plot import plot_spectrum
from visualization.antenna_plot import plot_semicircle_with_wave
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.pyplot as plt
from numpy.fft import rfft, rfftfreq
from tkinter import filedialog
from core.signal_import import load_signal_from_wav, load_signal_from_mat, load_signal_from_txt, load_signal_from_csv
import sounddevice as sd
from scipy.io.wavfile import write
def create_app():
    root = tk.Tk()
    root.title("Акустическая система")
    root.geometry("900x700")
    
    # обработчик выхода
    def on_close():
        root.quit()      # завершает цикл mainloop
        root.destroy()   # уничтожает окно
    
    root.protocol("WM_DELETE_WINDOW", on_close)
    
    root.figures = {}  # словарь для хранения фигур

    def save_figure(name, filename):
        fig = root.figures.get(name)
        if fig is not None:
            fig.savefig(filename, dpi=300)
            print(f"Сохранено: {filename}")

    def save_all_figures():
        for name, fig in root.figures.items():
            fig.savefig(f"{name}.png", dpi=300)
            print(f"Сохранено: {name}.png")
    
    # --- Меню ---
    menubar = tk.Menu(root)
    root.config(menu=menubar)
    file_menu = tk.Menu(menubar, tearoff=0)
    menubar.add_cascade(label="Файл", menu=file_menu)

    file_menu.add_command(label="Сохранить все графики", command=save_all_figures)
    file_menu.add_separator()
    file_menu.add_command(label="Сохранить сигнал", command=lambda: save_figure("signals", "signals.png"))
    file_menu.add_command(label="Сохранить антенну", command=lambda: save_figure("antenna", "antenna.png"))
    file_menu.add_command(label="Сохранить суммарный сигнал", command=lambda: save_figure("summed_signal", "summed_signal.png"))
    file_menu.add_command(label="Сохранить спектр", command=lambda: save_figure("spectrum", "spectrum.png"))
    file_menu.add_command(label="Сохранить сдвинутый сигнал", command=lambda: save_figure("shifted_signal", "shifted_signal.png"))
    file_menu.add_command(label="Сохранить сдвинутый спектр", command=lambda: save_figure("shifted_spectrum", "shifted_spectrum.png"))
    # --- Notebook для ввода параметров ---
    input_frame = ttk.Frame(root)
    input_frame.pack(fill="both", expand=True)

    # --- Блоки параметров (с внутренним padding, но без внешних отступов) ---
    synthetic_frame   = ttk.LabelFrame(input_frame, text="Сигнал", padding=(5,5))
    import_frame      = ttk.LabelFrame(input_frame, text="Импорт", padding=(5,5))
    shift_frame       = ttk.LabelFrame(input_frame, text="Сдвиг", padding=(5,5))
    antenna_frame     = ttk.LabelFrame(input_frame, text="Антенна", padding=(5,5))
    processing_frame  = ttk.LabelFrame(input_frame, text="Обработка", padding=(5,5))
    window_frame      = ttk.LabelFrame(input_frame, text="Окно", padding=(5,5))
    # --- Расположение в сетке (впритык, sticky="nsew") ---
    synthetic_frame.grid(row=0, column=0, sticky="nsew")
    import_frame.grid(row=0, column=1, sticky="nsew")
    shift_frame.grid(row=0, column=2, sticky="nsew")

    antenna_frame.grid(row=1, column=0, sticky="nsew")
    processing_frame.grid(row=1, column=1, sticky="nsew")
    window_frame.grid(row=1, column=2, sticky="nsew")
    for i in range(3):
        input_frame.columnconfigure(i, weight=1, minsize=300)
    # --- Универсальная функция для создания полей ---
    def add_fields(parent, fields, entries_dict, start_row=0):
        for i, (label, default) in enumerate(fields, start=start_row):
            ttk.Label(parent, text=label, anchor="w").grid(row=i, column=0, padx=2, pady=1, sticky="w")
            entry = ttk.Entry(parent)
            entry.insert(0, default)
            entry.grid(row=i, column=1, padx=2, pady=1, sticky="ew")
            entries_dict[label] = entry
        parent.columnconfigure(1, weight=1)
    
    # --- поля ввода ---
    # --- Поля для блока "Сигнал" ---
    signal_entries = {}
    signal_fields = [
        ("Частота сигнала (Гц)", "50"),
        ("Амплитуда сигнала", "2"),
        ("Длительность сигнала (с)", "1.0"),
        ("Частота дискретизации (Гц)", "24000")
    ]
    add_fields(synthetic_frame, signal_fields, signal_entries) 
     # --- Блок "Импорт" ---
    file_status_label = ttk.Label(import_frame, text="Файл не загружен")
    file_status_label.grid(row=2, column=0, columnspan=2, pady=5, sticky="ew")
    def import_signal():
        filepath = filedialog.askopenfilename(filetypes=[("WAV files","*.wav"),("MAT files","*.mat")])
        if not filepath:
            return
        if filepath.endswith(".wav"):
            signal, fd, frequency_est   = load_signal_from_wav(filepath)
        elif filepath.endswith(".mat"):
            signal, fd, frequency_est  = load_signal_from_mat(filepath)
        elif filepath.endswith(".csv"):
            signal, fd, frequency_est = load_signal_from_csv(filepath)
        elif filepath.endswith(".txt"):
            signal, fd, frequency_est = load_signal_from_txt(filepath)
        else:
            print("Неподдерживаемый формат")
            return

        # сохраняем в root, чтобы потом передать в SimulationParameters
        root.imported_signal = signal
        root.imported_fd = fd
        duration = len(signal) / fd
        amplitude = float(np.max(np.abs(signal)))
        signal_entries["Частота дискретизации (Гц)"].delete(0, tk.END)
        signal_entries["Частота дискретизации (Гц)"].insert(0, str(fd))

        signal_entries["Длительность сигнала (с)"].delete(0, tk.END)
        signal_entries["Длительность сигнала (с)"].insert(0, f"{duration:.3f}")

        signal_entries["Амплитуда сигнала"].delete(0, tk.END)
        signal_entries["Амплитуда сигнала"].insert(0, f"{amplitude:.3f}")
        signal_entries["Частота сигнала (Гц)"].delete(0, tk.END)
        signal_entries["Частота сигнала (Гц)"].insert(0, f"{frequency_est:.1f}")
        file_status_label.config(text=f"Загружен файл: {filepath.split('/')[-1]}")
        remove_button.grid(row=1, column=0, sticky="ew")   # теперь кнопка появится
    def remove_signal():
        root.imported_signal = None
        root.imported_fd = None

        # вернуть дефолтные значения
        defaults = {
            "Частота сигнала (Гц)": "50",
            "Амплитуда сигнала": "2",
            "Длительность сигнала (с)": "1.0",
            "Частота дискретизации (Гц)": "24000"
        }
        for label, value in defaults.items():
            signal_entries[label].delete(0, tk.END)
            signal_entries[label].insert(0, value)

        file_status_label.config(text="Файл не загружен")
        remove_button.grid_forget()
    ttk.Button(import_frame, text="Загрузить файл",command=import_signal).grid(row=0, column=0, sticky="ew")
    remove_button = ttk.Button(import_frame, text="Удалить файл", command=remove_signal)

    
    
    
    # --- Блок "Сдвиг" ---
    enable_shift_var = tk.BooleanVar()
    ttk.Checkbutton(shift_frame, text="Включить", variable=enable_shift_var)\
        .grid(row=0, column=0, columnspan=2, sticky="w")

    shift_entries = {}
    shift_fields = [
        ("Нижняя граница сдвига (Гц)", "0"),
        ("Верхняя граница сдвига (Гц)", "12000"),
        ("k (сдвиг в отсчетах)", "10")
    ]
    add_fields(shift_frame, shift_fields, shift_entries, start_row=1)


    # --- Поля для блока "Антенна" ---
    antenna_entries = {}
    antenna_fields = [
        ("Радиус антенны (м)", "1"),
        ("Количество приемников", "5"),
        ("Угол прихода волны (градусы)", "0")
    ]
    add_fields(antenna_frame, antenna_fields, antenna_entries)

    # --- Поля для блока "Обработка" ---
    processing_entries = {}
    processing_fields = [
        ("Скорость звука (м/с)", "1500"),
        ("Длина блока FFT (отсчёты)", "512"),
        ("Перекрытие блока (отсчёты)", "32"),
    ]
    add_fields(processing_frame, processing_fields, processing_entries)
   # --- Блок "Окно" ---
    band_window_var = tk.StringVar(value="hanning")
    ttk.Label(window_frame, text="Функция окна").grid(row=0, column=0, sticky="w")
    band_window_combo = ttk.Combobox(window_frame, textvariable=band_window_var,
                                    values=["hanning", "hamming", "blackman", "rectangular"],
                                    state="readonly")
    band_window_combo.grid(row=0, column=1, sticky="ew")
    window_frame.columnconfigure(1, weight=1)
    control_frame = ttk.Frame(root)
    control_frame.pack(fill="x", pady=5)
    
    # --- Notebook для вывода результатов ---
    output_notebook = ttk.Notebook(root)
    output_notebook.pack(fill='both', expand=True)

    # существующие вкладки
    signal_plot_tab = ttk.Frame(output_notebook)
    antenna_plot_tab = ttk.Frame(output_notebook)

    output_notebook.add(signal_plot_tab, text="График сигналов")
    output_notebook.add(antenna_plot_tab, text="Схема антенны")


    # новые вкладки
    summed_plot_tab = ttk.Frame(output_notebook)
    spectrum_plot_tab = ttk.Frame(output_notebook)

    output_notebook.add(summed_plot_tab, text="Суммарный сигнал")
    output_notebook.add(spectrum_plot_tab, text="Спектр")
    player_frame = ttk.Frame(output_notebook)
    output_notebook.add(player_frame, text="Прослушать сигнал")
    shifted_plot_tab = ttk.Frame(output_notebook)
    shifted_spectrum_tab = ttk.Frame(output_notebook)

    output_notebook.add(shifted_plot_tab, text="Сдвинутый сигнал")
    output_notebook.add(shifted_spectrum_tab, text="Сдвинутый спектр")


    # --- Кнопка запуска ---
    def collect_parameters():
        params = SimulationParameters(
            frequency=float(signal_entries["Частота сигнала (Гц)"].get()),
            amplitude=float(signal_entries["Амплитуда сигнала"].get()),
            duration=float(signal_entries["Длительность сигнала (с)"].get()),
            fd=float(signal_entries["Частота дискретизации (Гц)"].get()),
            phi_angle=float(antenna_entries["Угол прихода волны (градусы)"].get()),
            radius=float(antenna_entries["Радиус антенны (м)"].get()),
            num_receivers=int(antenna_entries["Количество приемников"].get()),
            c=float(processing_entries["Скорость звука (м/с)"].get()),
            n_fft=int(processing_entries["Длина блока FFT (отсчёты)"].get()),
            block_overlap=int(processing_entries["Перекрытие блока (отсчёты)"].get()),
            enable_shift=enable_shift_var.get(),
            shift_low_freq=float(shift_entries["Нижняя граница сдвига (Гц)"].get()),
            shift_high_freq=float(shift_entries["Верхняя граница сдвига (Гц)"].get()),
            shift_amount_bins=int(shift_entries["k (сдвиг в отсчетах)"].get()),
            band_window=band_window_var.get(),
            imported_signal=getattr(root, "imported_signal", None)
        )
        if hasattr(root, "imported_fd"):
            params.fd = root.imported_fd

        def play_signal(signal, fd):
            sd.play(signal, fd)
            sd.wait()


        def save_signal(signal, fd):
            filepath = filedialog.asksaveasfilename(defaultextension=".wav",
                                                    filetypes=[("WAV files", "*.wav")])
            if filepath:
                scaled = np.int16(signal/np.max(np.abs(signal)) * 32767)
                write(filepath, int(fd), scaled)
                print(f"Сигнал сохранён: {filepath}")
        # расчёт
        # x, y, signals, t, summed_signal, freqs, spectrum = run_calculation(params)
        x, y, signals, t, summed_signal, freqs, spectrum, shifted_signal, shifted_spectrum, freqs_shifted  = run_calculation(params)
        # очистка старых графиков
        for widget in signal_plot_tab.winfo_children():
            widget.destroy()
        for widget in antenna_plot_tab.winfo_children():
            widget.destroy()
        for widget in summed_plot_tab.winfo_children():
            widget.destroy()
        for widget in spectrum_plot_tab.winfo_children():
            widget.destroy()
        for widget in shifted_plot_tab.winfo_children():
            widget.destroy()
        for widget in shifted_spectrum_tab.winfo_children():
            widget.destroy()
        for widget in player_frame.winfo_children():
            widget.destroy()  # очистка старых кнопок
        for widget in player_frame.winfo_children():
            widget.destroy()
        # новые графики
        # 1. Сигналы по каналам
        # 1. Сигналы по каналам
        fig1 = plot_signals(signals, t)
        root.figures["signals"] = fig1
        canvas1 = FigureCanvasTkAgg(fig1, master=signal_plot_tab)
        toolbar1 = NavigationToolbar2Tk(canvas1, signal_plot_tab)  # тулбар
        toolbar1.update()
        canvas1.get_tk_widget().pack(fill="both", expand=True)
        canvas1.draw()

        # 2. Антенна
        fig2 = plot_semicircle_with_wave(x, y, params.radius, params.phi_angle, start_angle_deg=270)
        root.figures["antenna"] = fig2
        canvas2 = FigureCanvasTkAgg(fig2, master=antenna_plot_tab)
        toolbar2 = NavigationToolbar2Tk(canvas2, antenna_plot_tab)
        toolbar2.update()
        canvas2.get_tk_widget().pack(fill="both", expand=True)
        canvas2.draw()

        # 3. Суммарный сигнал
        fig3 = plot_summed_signal(summed_signal, t)
        root.figures["summed_signal"] = fig3
        canvas3 = FigureCanvasTkAgg(fig3, master=summed_plot_tab)
        toolbar3 = NavigationToolbar2Tk(canvas3, summed_plot_tab)
        toolbar3.update()
        canvas3.get_tk_widget().pack(fill="both", expand=True)
        canvas3.draw()

        # 4. Спектр
        fig4 = plot_spectrum(freqs, spectrum)
        root.figures["spectrum"] = fig4
        canvas4 = FigureCanvasTkAgg(fig4, master=spectrum_plot_tab)
        toolbar4 = NavigationToolbar2Tk(canvas4, spectrum_plot_tab)
        toolbar4.update()
        canvas4.get_tk_widget().pack(fill="both", expand=True)
        canvas4.draw()

        # 5. Сдвинутый сигнал
        if params.enable_shift:
            fig5 = plot_summed_signal(shifted_signal, t)
            root.figures["shifted_signal"] = fig5
            canvas5 = FigureCanvasTkAgg(fig5, master=shifted_plot_tab)
            toolbar5 = NavigationToolbar2Tk(canvas5, shifted_plot_tab)
            toolbar5.update()
            canvas5.get_tk_widget().pack(fill="both", expand=True)
            canvas5.draw()

            # 6. Сдвинутый спектр
            fig6 = plot_spectrum(freqs_shifted, shifted_spectrum)
            root.figures["shifted_spectrum"] = fig6
            canvas6 = FigureCanvasTkAgg(fig6, master=shifted_spectrum_tab)
            toolbar6 = NavigationToolbar2Tk(canvas6, shifted_spectrum_tab)
            toolbar6.update()
            canvas6.get_tk_widget().pack(fill="both", expand=True)
            canvas6.draw()
        ttk.Button(player_frame, text="Прослушать суммарный сигнал",
           command=lambda: play_signal(summed_signal, params.fd)).pack(pady=10)

        ttk.Button(player_frame, text="Сохранить суммарный сигнал",
                command=lambda: save_signal(summed_signal, params.fd)).pack(pady=10)
    def compare_windows():
        # собираем параметры один раз
        params = SimulationParameters(
            frequency=float(signal_entries["Частота сигнала (Гц)"].get()),
            amplitude=float(signal_entries["Амплитуда сигнала"].get()),
            duration=float(signal_entries["Длительность сигнала (с)"].get()),
            fd=float(signal_entries["Частота дискретизации (Гц)"].get()),
            phi_angle=float(antenna_entries["Угол прихода волны (градусы)"].get()),
            radius=float(antenna_entries["Радиус антенны (м)"].get()),
            num_receivers=int(antenna_entries["Количество приемников"].get()),
            c=float(processing_entries["Скорость звука (м/с)"].get()),
            n_fft=int(processing_entries["Длина блока FFT (отсчёты)"].get()),
            block_overlap=int(processing_entries["Перекрытие блока (отсчёты)"].get()),
            enable_shift=True,  # сравнение имеет смысл только со сдвигом
            shift_low_freq=float(shift_entries["Нижняя граница сдвига (Гц)"].get()),
            shift_high_freq=float(shift_entries["Верхняя граница сдвига (Гц)"].get()),
            shift_amount_bins=int(shift_entries["k (сдвиг в отсчетах)"].get()),
            band_window="hanning" 
        )

        # очистка старых графиков
        for widget in shifted_spectrum_tab.winfo_children():
            widget.destroy()

        fig, ax = plt.subplots(figsize=(8,5))
        window_types = ["hanning", "hamming", "blackman", "rectangular"]

        for wtype in window_types:
            params.band_window = wtype
            # запускаем расчёт
            _, _, _, _, _, _, _, shifted_signal, spectrum_shifted, freqs_shifted = run_calculation(params)
            # считаем спектр
            ax.plot(freqs_shifted, spectrum_shifted, label=wtype)
        root.figures["shifted_spectrum"] = fig
        ax.set_title("Сравнение окон")
        ax.set_xlabel("Частота (Гц)")
        ax.set_ylabel("Амплитуда")
        ax.legend()

        canvas = FigureCanvasTkAgg(fig, master=shifted_spectrum_tab)
        canvas.get_tk_widget().pack(fill="both", expand=True)
        canvas.draw()
        

    ttk.Button(control_frame, text="Рассчитать",command=collect_parameters).pack(side="left", padx=5)
    # ttk.Button(top_frame, text="Рассчитать и визуализировать", command=collect_parameters).pack(pady=5)

    # ttk.Button(root, text="Рассчитать и визуализировать", command=collect_parameters).pack(side="top", pady=10)
    compare_button = ttk.Button(root, text="Сравнить окна", command=compare_windows)

    def toggle_compare_button():
        if enable_shift_var.get():
            compare_button.pack(pady=10)
        else:
            compare_button.pack_forget()
    
    # привязываем обработчик к галочке
    enable_shift_var.trace_add("write", lambda *args: toggle_compare_button())

    return root
