import matplotlib.pyplot as plt
import numpy as np

def plot_semicircle_with_wave(x, y, radius, phi_angle_deg, start_angle_deg):
    """Рисует полукруговую антенну и возвращает Figure."""
    fig, ax = plt.subplots(figsize=(6,6))
    phi_arc = np.linspace(np.deg2rad(start_angle_deg),
                          np.deg2rad(start_angle_deg)+np.pi, 100)
    xc, yc = radius*np.cos(phi_arc), radius*np.sin(phi_arc)
    ax.plot(xc, yc, '--', color='gray', label="Дуга антенны")
    ax.scatter(x, y, c='blue', s=100, label="Приёмники")

    phi = np.deg2rad(phi_angle_deg)
    L = radius*1.3
    ax.arrow(0, 0, L*np.cos(phi), L*np.sin(phi),
             head_width=0.15, head_length=0.25, fc='red', ec='red', label="Приход волны")

    ax.set_aspect('equal')
    ax.grid(True)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Полукруговая антенна")
    ax.legend()
    return fig
