import numpy as np

class AntennaArray:
    def __init__(self, num_receivers: int, radius: float, start_angle_deg: float = 270.0):
        self.num_receivers = num_receivers
        self.radius = radius
        self.start_angle_deg = start_angle_deg

        # координаты приёмников
        self.x, self.y, self.phi_m = self._place_semicircular_array()
    
    def _place_semicircular_array(self):
        phi_start = np.deg2rad(self.start_angle_deg)
        phi_end = phi_start + np.pi
        phi_m = np.linspace(phi_start, phi_end, self.num_receivers)
        x = self.radius * np.cos(phi_m)
        y = self.radius * np.sin(phi_m)
        return x, y, phi_m
    
    def compute_delays(self, phi_angle_deg: float, c: float):
        phi = np.deg2rad(phi_angle_deg)
        delays = -(self.radius / c) * (np.cos(phi) * np.cos(self.phi_m) +
                                       np.sin(phi) * np.sin(self.phi_m))
        # нормируем относительно минимальной задержки
        return delays - np.min(delays)

    def get_coordinates(self):
        """Возвращает координаты приёмников."""
        return self.x, self.y

# def place_semicircular_array(num_receivers: int, radius: float, start_angle_deg: float = 270.0):
#     """Рассчитывает координаты приёмников на окружности."""
#     phi_start = np.deg2rad(start_angle_deg)
#     phi_end = phi_start + np.pi
#     phi_m = np.linspace(phi_start, phi_end, num_receivers)
#     x = radius * np.cos(phi_m)
#     y = radius * np.sin(phi_m)
#     return x, y, phi_m

# def delays_plane_wave_semicircular(radius: float, phi_angle_deg: float, phi_m, c: float):
#     """Рассчитывает задержки прихода плоской волны под углом phi_angle."""
#     phi = np.deg2rad(phi_angle_deg)
#     delays = -(radius/c) * (np.cos(phi)*np.cos(phi_m) + np.sin(phi)*np.sin(phi_m))
#     return delays - np.min(delays)
