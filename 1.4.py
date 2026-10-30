import numpy as np
import matplotlib.pyplot as plt

class Detector:
    def __init__(self, R: float, N: int, r: float, a: float):
        self.R = R
        self.N = N
        self.r = r
        self.a = a
        self.pmt_positions = self._generate_pmt_positions()

    def _generate_pmt_positions(self) -> np.ndarray:
        indices = np.arange(0, self.N, dtype=float) + 0.5
        phi = np.arccos(1 - 2 * indices / self.N)
        theta = np.pi * (1 + 5**0.5) * indices
        x = np.cos(theta) * np.sin(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(phi)
        return np.vstack((x, y, z)).T * self.R

    def trace_photons(self, origin: np.ndarray, directions: np.ndarray) -> np.ndarray:
        b = 2.0 * (directions @ origin)
        c = float(np.dot(origin, origin) - self.R ** 2)
        discr = b ** 2 - 4 * c
        t = (-b + np.sqrt(np.maximum(discr, 0.0))) / 2.0
        t = np.maximum(t, 0.0)
        hit_points = origin + t[:, None] * directions
        diff = hit_points[:, None, :] - self.pmt_positions[None, :, :]
        dists = np.linalg.norm(diff, axis=2)
        closest = np.argmin(dists, axis=1)
        min_dists = np.min(dists, axis=1)
        return np.where(min_dists < self.r, closest, -1)

    def generate_photoelectrons(self, mean: float | None = None) -> int:
        mu = self.a if mean is None else mean
        return int(np.random.poisson(mu))


class Source:
    def __init__(self, Y: float):
        self.Y = Y

    def generate_event(self, E: float) -> np.ndarray:
        N = np.random.poisson(E * self.Y)
        if N == 0:
            return np.empty((0, 3))
        phi = np.random.uniform(0, 2*np.pi, size=N)
        cos_theta = np.random.uniform(-1, 1, size=N)
        sin_theta = np.sqrt(1 - cos_theta**2)
        x = sin_theta * np.cos(phi)
        y = sin_theta * np.sin(phi)
        z = cos_theta
        return np.vstack((x, y, z)).T


R = 2.0
N_pmt = 40
r = 0.1
a = 0.5
Y = 10000.0
N_events = 1000

detector = Detector(R=R, N=N_pmt, r=r, a=a)
source = Source(Y=Y)


def simulate_events(energy, origin, n_events=1000):
    charges = np.zeros(n_events)
    for i in range(n_events):
        dirs = source.generate_event(energy)
        hit_idx = detector.trace_photons(origin, dirs)
        hits = hit_idx[hit_idx >= 0]
        q = 0
        for _ in hits:
            q += detector.generate_photoelectrons()
        charges[i] = q
    return charges


def analytic_Q(E, origin):
    return a * Y * E * (N_pmt * r ** 2) / (4 * R ** 2)


print("--- Пункт 1: E = 1.0 МэВ, источник в центре ---")
origin_center = np.array([0.0, 0.0, 0.0])
charges_E1 = simulate_events(1.0, origin_center, N_events)
Q_theory = analytic_Q(1.0, origin_center)

mean_Q = charges_E1.mean()
std_Q = charges_E1.std()
rel_width = std_Q / mean_Q

print(f"Среднее Q = {mean_Q:.2f}")
print(f"Стд. откл Q = {std_Q:.2f}")
print(f"Относительная ширина σ/Q = {rel_width:.4f}")
print(f"Теория Q = {Q_theory:.2f}")

plt.figure(figsize=(8, 4))
plt.hist(charges_E1, bins=40, color='steelblue', edgecolor='black', alpha=0.8)
plt.axvline(Q_theory, color='red', linestyle='--', linewidth=2, label=f'Теория = {Q_theory:.1f}')
plt.axvline(mean_Q, color='orange', linestyle='--', linewidth=2, label=f'Среднее = {mean_Q:.1f}')
plt.xlabel('Полный заряд Q')
plt.ylabel('Число событий')
plt.title('Распределение заряда для E = 1.0 МэВ')
plt.legend()
plt.grid(alpha=0.3)
plt.show()


print("\n--- Пункт 2: Зависимость ширины от энергии ---")
energies = [0.2, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0]
results = []

for E in energies:
    charges = simulate_events(E, origin_center, N_events)
    m = charges.mean()
    s = charges.std()
    rw = s / m if m > 0 else 0
    results.append((E, m, s, rw))
    print(f"E = {E:.1f} МэВ | <Q> = {m:.1f} | σ = {s:.1f} | σ/<Q> = {rw:.4f}")

energies_arr = np.array([res[0] for res in results])
means_arr = np.array([res[1] for res in results])
stds_arr = np.array([res[2] for res in results])
rel_widths = np.array([res[3] for res in results])

fig, axes = plt.subplots(1, 2, figsize=(14, 4))

axes[0].plot(energies_arr, means_arr, 'o-', color='blue', label='Моделирование')
axes[0].plot(energies_arr, [analytic_Q(E, origin_center) for E in energies_arr], 'r--', label='Теория')
axes[0].set_xlabel('Энергия E, МэВ')
axes[0].set_ylabel('<Q>')
axes[0].set_title('Средний заряд от энергии')
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].loglog(energies_arr, rel_widths, 'o-', color='green', label='σ/<Q>')
C = rel_widths[0] * np.sqrt(energies_arr[0])
axes[1].loglog(energies_arr, C / np.sqrt(energies_arr), 'r--', label='~ 1/√E')
axes[1].set_xlabel('Энергия E, МэВ')
axes[1].set_ylabel('Относительная ширина σ/<Q>')
axes[1].set_title('Относительная ширина распределения')
axes[1].legend()
axes[1].grid(alpha=0.3, which='both')

plt.tight_layout()
plt.show()


print("\n--- Пункт 3: Источник смещен на 1 м по оси Z ---")
origin_shifted = np.array([0.0, 0.0, 1.0])
charges_shifted = simulate_events(1.0, origin_shifted, N_events)

mean_shifted = charges_shifted.mean()
std_shifted = charges_shifted.std()
rel_width_shifted = std_shifted / mean_shifted

print(f"Среднее Q (смещенный) = {mean_shifted:.2f}")
print(f"Стд. откл (смещенный) = {std_shifted:.2f}")
print(f"Относительная ширина (смещенный) = {rel_width_shifted:.4f}")

print(f"Изменение среднего: {mean_shifted - mean_Q:.2f} ({(mean_shifted/mean_Q - 1)*100:.1f}%)")

plt.figure(figsize=(8, 4))
plt.hist(charges_E1, bins=40, color='steelblue', edgecolor='black', alpha=0.6, label='Центр', density=True)
plt.hist(charges_shifted, bins=40, color='orange', edgecolor='black', alpha=0.6, label='Смещен на 1 м', density=True)
plt.axvline(mean_Q, color='blue', linestyle='--', linewidth=2, label=f'Среднее (центр) = {mean_Q:.1f}')
plt.axvline(mean_shifted, color='darkorange', linestyle='--', linewidth=2, label=f'Среднее (смещ.) = {mean_shifted:.1f}')
plt.xlabel('Полный заряд Q')
plt.ylabel('Плотность вероятности')
plt.title('Сравнение распределений для центра и смещенного источника')
plt.legend()
plt.grid(alpha=0.3)
plt.show()