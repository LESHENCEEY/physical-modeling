import numpy as np
import matplotlib.pyplot as plt

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

    def generate_photoelectrons(self, mean: float | None = None) -> int:
        mu = self.a if mean is None else mean
        return int(np.random.poisson(mu))


class Source:
    def __init__(self, Y: float):
        self.Y = Y

    def generate_event(self, E: float):
        N_total = np.random.poisson(E * self.Y)
        generated = 0
        while generated < N_total:
            chunk_size = min(100, N_total - generated)
            phi = np.random.uniform(0, 2*np.pi, size=chunk_size)
            cos_theta = np.random.uniform(-1, 1, size=chunk_size)
            sin_theta = np.sqrt(1 - cos_theta**2)
            x = sin_theta * np.cos(phi)
            y = sin_theta * np.sin(phi)
            z = cos_theta
            chunk = np.vstack((x, y, z)).T
            yield chunk
            generated += chunk_size


def simulate_events(energy, origin, detector, source, n_events=1000):
    charges = np.zeros(n_events)
    for i in range(n_events):
        q = 0
        for dirs_chunk in source.generate_event(energy):
            hit_idx, dists = detector.trace_photons(origin, dirs_chunk)
            hits_mask = hit_idx >= 0
            
            if detector.L is not None:
                survival_prob = np.exp(-dists[hits_mask] / detector.L)
                survived = np.random.uniform(0, 1, size=len(survival_prob)) < survival_prob
                valid_hits = hit_idx[hits_mask][survived]
            else:
                valid_hits = hit_idx[hits_mask]

            for _ in valid_hits:
                q += detector.generate_photoelectrons()
        charges[i] = q
    return charges


R = 2.0
N_pmt = 40
r = 0.1
a = 0.5
Y = 10000.0
L = 8.0
N_events = 1000

detector_no_att = Detector(R=R, N=N_pmt, r=r, a=a, L=None)
detector_att = Detector(R=R, N=N_pmt, r=r, a=a, L=L)
source = Source(Y=Y)

z_positions = [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5]
results_no_att = []
results_att = []

print(f"Моделирование {N_events} событий для разных положений источника (ось Z):")
print(f"{'Z, м':>5} | {'<Q> без зат.':>12} | {'<Q> с зат.':>10} | {'Отношение':>10}")
print("-" * 45)

for z in z_positions:
    origin = np.array([0.0, 0.0, z])
    
    q_no_att = simulate_events(1.0, origin, detector_no_att, source, N_events)
    q_att = simulate_events(1.0, origin, detector_att, source, N_events)
    
    mean_no_att = q_no_att.mean()
    mean_att = q_att.mean()
    
    results_no_att.append(mean_no_att)
    results_att.append(mean_att)
    
    ratio = mean_att / mean_no_att if mean_no_att > 0 else 0
    print(f"{z:5.1f} | {mean_no_att:12.2f} | {mean_att:10.2f} | {ratio:10.3f}")


plt.figure(figsize=(10, 4))

plt.plot(z_positions, results_no_att, 'o-', color='blue', label='Без затухания')
plt.plot(z_positions, results_att, 's--', color='red', label=f'С затуханием (L={L} м)')

plt.xlabel('Положение источника по оси Z, м')
plt.ylabel('Средний собранный заряд <Q>')
plt.title('Энергетический отклик детектора от положения источника')
plt.legend()
plt.grid(alpha=0.3)
plt.show()