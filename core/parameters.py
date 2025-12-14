# Хранит все параметры моделирования 
from dataclasses import dataclass
from typing import Optional
import numpy as np

@dataclass
class SimulationParameters:
    # --- Сигнал ---
    frequency: float
    amplitude: float
    duration: float
    fd: float
    phi_angle: float

    # --- Антенна ---
    radius: float
    num_receivers: int

    # --- Обработка ---
    c: float
    n_fft: int 
    block_overlap: int

    # --- Сдвиг частоты ---
    enable_shift: bool
    shift_low_freq: float
    shift_high_freq: float
    shift_amount_bins: int
    band_window: str

    imported_signal: Optional[np.ndarray] = None
