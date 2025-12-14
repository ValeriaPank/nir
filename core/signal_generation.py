# # Функции для генерации многоканальных сигналов:
# # - создание временной оси
# # - генерация синусоид для каждого приёмника с учётом задержек
# import numpy as np
# from core.parameters import SimulationParameters

# def generate_multichannel_signals(params: SimulationParameters, delays):
#     t = np.linspace(0, params.duration, int(params.fd * params.duration))
#     signals = []
#     for d in delays:
#         sig = params.amplitude * np.sin(2*np.pi*params.frequency*(t - d))
#         signals.append(sig)
#     return np.array(signals), t

import numpy as np
from core.parameters import SimulationParameters

class SignalGenerator:
    """
    Класс для генерации многоканальных сигналов.
    """

    def __init__(self, params: SimulationParameters):
        self.params = params
        # создаём временную ось сразу при инициализации
        if params.imported_signal is not None:
            self.base_signal = params.imported_signal
            self.t = np.arange(len(self.base_signal)) / params.fd
        else:
            dt = 1 / params.fd
            self.t = np.arange(0, params.duration, dt)
            self.base_signal = None


    def generate(self, delays, signal_type="harmonic", extra_freqs=None):
        """
        Генерация сигналов для каждого приёмника с учётом задержек.
        signal_type: "harmonic", "noise", "multitone"
        extra_freqs: список дополнительных частот для многотонального сигнала
        """
        signals = []
        for d in delays:
            if self.base_signal is not None:
                delay_samples = int(d * self.params.fd)
                sig_delayed = np.roll(self.base_signal, delay_samples)
                sig_delayed[:delay_samples] = 0  # обнуляем начало
                signals.append(sig_delayed)
            else:
                if signal_type == "harmonic":
                    # Обычная синусоида
                    sig = self.params.amplitude * np.sin(
                        2 * np.pi * self.params.frequency * (self.t - d)
                    )
                
                elif signal_type == "noise":
                    # Белый шум
                    sig = np.random.normal(0, 1, len(self.t))
                
                elif signal_type == "multitone":
                    extra_freqs=[500, 5200]
                    # Сумма нескольких синусоид
                    freqs = [self.params.frequency] + (extra_freqs or [])
                    sig = sum(
                        self.params.amplitude * np.sin(2 * np.pi * f * (self.t - d))
                        for f in freqs
                    )
                
                else:
                    raise ValueError("Unknown signal_type")
            
                signals.append(sig)
        
        return np.array(signals), self.t

