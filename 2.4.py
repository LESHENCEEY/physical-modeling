import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.spatial import cKDTree

class DetectorLarge:
    def __init__(self, R: float, N: int, r: float, a: float, L: float | None = None):
        self.R = R
        self.N = N
        self.r = r
        self.a = a
        self.L = L
        self.pmt_positions = self._generate_pmt_positions()
        self.tree = cKDTree(self.pmt_positions)

    def _generate_pmt_positions(self) -> np.ndarray:
        indices = np.arange(0, self.N, dtype=float) + 0.5
        phi = np.arccos(1 - 2 * indices / self.N)
        theta = np.pi * (1 + 5**0.5) * indices
        x = np.cos(theta) * np.sin(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(phi)
        return np.vstack((x, y, z)).T * self.R

    def trace_photons(self, origin: np.ndarray, directions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        b = 2.0 * (directions @ origin)
        c = float(np.dot(origin, origin) - self.R ** 2)
        discr = b ** 2 - 4 * c
        t = (-b + np.sqrt(np.maximum(discr, 0.0))) / 2.0
        t = np.maximum(t, 0.0)

        hit_points = origin + t[:, None] * directions
        dists, closest = self.tree.query(hit_points, k=1)
        hit_idx = np.where(dists < self.r, closest, -1)
        return hit_idx, t


class Source:
    def __init__(self, Y: float):
        self.Y = Y

    def generate_event_array(self, E: float) -> np.ndarray:
        N_total = np.random.poisson(E * self.Y)
        if N_total == 0:
            return np.empty((0, 3))
        phi = np.random.uniform(0, 2*np.pi, size=N_total)
        cos_theta = np.random.uniform(-1, 1, size=N_total)
        sin_theta = np.sqrt(1 - cos_theta**2)
        x = sin_theta * np.cos(phi)
        y = sin_theta * np.sin(phi)
        z = cos_theta
        return np.vstack((x, y, z)).T


def simulate_events(energy, origin, detector, source, n_events):
    charges = np.zeros(n_events)
    for i in range(n_events):
        dirs = source.generate_event_array(energy)
        hit_idx, dists = detector.trace_photons(origin, dirs)
        hits_mask = hit_idx >= 0

        if detector.L is not None:
            survival_prob = np.exp(-dists[hits_mask] / detector.L)
            survived = np.random.uniform(0, 1, size=len(survival_prob)) < survival_prob
            n_valid_hits = np.sum(survived)
        else:
            n_valid_hits = np.sum(hits_mask)

        if n_valid_hits > 0:
            pes = np.random.poisson(detector.a, size=n_valid_hits)
            charges[i] = np.sum(pes)
    return charges


R = 15.0
N_pmt = 2000
r = 0.1
a = 0.5
Y = 10000.0
E = 1.0
N_events = 1000
origin = np.array([0.0, 0.0, 0.0])

detector = DetectorLarge(R=R, N=N_pmt, r=r, a=a)
source = Source(Y=Y)

Q_theory = a * Y * (N_pmt * r**2) / (4 * R**2)
print(f"Теоретический заряд: {Q_theory:.2f}")

t0 = time.perf_counter()
charges = simulate_events(E, origin, detector, source, N_events)
t_elapsed = time.perf_counter() - t0

print(f"Время моделирования {N_events} событий: {t_elapsed:.3f} с")
print(f"Средний заряд: {charges.mean():.2f}")
print(f"Стд. отклонение: {charges.std():.2f}")

plt.figure(figsize=(8, 4))
plt.hist(charges, bins=40, color='steelblue', edgecolor='black', alpha=0.8)
plt.axvline(Q_theory, color='red', linestyle='--', linewidth=2, label=f'Теория = {Q_theory:.1f}')
plt.axvline(charges.mean(), color='orange', linestyle='--', linewidth=2, label=f'Среднее = {charges.mean():.1f}')
plt.xlabel('Полный заряд Q')
plt.ylabel('Число событий')
plt.title('Детектор большого объема (R=15 м, N=2000)')
plt.legend()
plt.grid(alpha=0.3)
plt.show()