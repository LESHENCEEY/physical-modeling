import numpy as np
import matplotlib.pyplot as plt
import time

class Detector:
    def __init__(self, R: float, N: int, r: float, a: float, L: float | None = None):
        self.R = R
        self.N = N
        self.r = r
        self.a = a
        self.L = L
        self.pmt_positions = self._generate_pmt_positions()

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
        diff = hit_points[:, None, :] - self.pmt_positions[None, :, :]
        dists = np.linalg.norm(diff, axis=2)

        closest = np.argmin(dists, axis=1)
        min_dists = np.min(dists, axis=1)
        hit_idx = np.where(min_dists < self.r, closest, -1)
        
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

def simulate_slow(energy, origin, detector, source, n_events):
    charges = np.zeros(n_events)
    for i in range(n_events):
        dirs = source.generate_event_array(energy)
        hit_idx, dists = detector.trace_photons(origin, dirs)
        hits_mask = hit_idx >= 0
        
        if detector.L is not None:
            survival_prob = np.exp(-dists[hits_mask] / detector.L)
            survived = np.random.uniform(0, 1, size=len(survival_prob)) < survival_prob
            valid_hits = hit_idx[hits_mask][survived]
        else:
            valid_hits = hit_idx[hits_mask]

        q = 0
        for _ in valid_hits:
            q += int(np.random.poisson(detector.a))
        charges[i] = q
    return charges

def simulate_fast(energy, origin, detector, source, n_events):
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
        else:
            charges[i] = 0
    return charges

R = 2.0
N_pmt = 40
r = 0.1
a = 0.5
Y = 10000.0
L = 8.0
N_events = 500
origin = np.array([0.0, 0.0, 0.0])

detector = Detector(R=R, N=N_pmt, r=r, a=a, L=L)
source = Source(Y=Y)

print(f"Замер времени для {N_events} событий (E=1 МэВ)...")

t0 = time.perf_counter()
q_slow = simulate_slow(1.0, origin, detector, source, N_events)
t_slow = time.perf_counter() - t0
print(f"Медленный (цикл по фотонам): {t_slow:.4f} с | <Q> = {q_slow.mean():.1f}")

t0 = time.perf_counter()
q_fast = simulate_fast(1.0, origin, detector, source, N_events)
t_fast = time.perf_counter() - t0
print(f"Быстрый (векторизованный):   {t_fast:.4f} с | <Q> = {q_fast.mean():.1f}")

speedup = t_slow / t_fast if t_fast > 0 else float('inf')
print(f"Ускорение: {speedup:.1f}x")

plt.figure(figsize=(10, 4))
plt.hist(q_slow, bins=40, alpha=0.5, color='blue', label='Циклы (медленно)', density=True)
plt.hist(q_fast, bins=40, alpha=0.5, color='red', label='Векторизация (быстро)', density=True)
plt.xlabel('Полный заряд Q')
plt.ylabel('Плотность')
plt.title('Сравнение распределений (должны совпадать)')
plt.legend()
plt.grid(alpha=0.3)
plt.show()