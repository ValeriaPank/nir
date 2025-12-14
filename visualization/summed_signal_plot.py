import matplotlib.pyplot as plt

def plot_summed_signal(summed_signal, t, max_samples=500):
    
    """Рисует суммарный сигнал после когерентного сложения и возвращает Figure."""
    fig, ax = plt.subplots(figsize=(6,4))
    max_samples = len(t)-1000
    ax.plot(t[:max_samples], summed_signal[:max_samples], color="black", linewidth=1.5)
    ax.set_xlabel("Время (с)")
    ax.set_ylabel("Амплитуда")
    ax.set_title("Суммарный сигнал")
    ax.grid()
    return fig
