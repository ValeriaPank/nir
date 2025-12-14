import matplotlib.pyplot as plt

def plot_signals(signals, t):
    """Рисует многоканальные сигналы с задержками и возвращает Figure."""
    fig, ax = plt.subplots(figsize=(6,4))
    for i, sig in enumerate(signals):
        ax.plot(t, sig, label=f"Приёмник {i+1}")
    ax.set_xlabel("Время (с)")
    ax.set_ylabel("Амплитуда")
    ax.set_title("Сигналы с задержками")
    ax.legend()
    ax.grid()
    return fig
