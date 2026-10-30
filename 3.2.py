import numpy as np
import matplotlib.pyplot as plt
import pymc as pm
import pytensor.tensor as pt

np.random.seed(42)


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
        return np.where(min_dists < self.r, closest, -1), t


class Source:
    def __init__(self, Y: float):
        self.Y = Y

    def generate_event_array(self, E: float) -> np.ndarray:
        N_total = np.random.poisson(E * self.Y)
        if N_total == 0:
            return np.empty((0, 3))
        phi = np.random.uniform(0, 2 * np.pi, size=N_total)
        cos_theta = np.random.uniform(-1, 1, size=N_total)
        sin_theta = np.sqrt(1 - cos_theta ** 2)
        x = sin_theta * np.cos(phi)
        y = sin_theta * np.sin(phi)
        z = cos_theta
        return np.vstack((x, y, z)).T


detector = Detector(R=2.0, N=40, r=0.1, a=0.5)
source = Source(Y=10000.0)

E_true = 1.5
pos_true = np.array([0.5, -0.2, 1.0])

dirs = source.generate_event_array(E_true)
hit_idx, _ = detector.trace_photons(pos_true, dirs)

true_pe_counts = np.zeros(detector.N, dtype=int)
for idx in hit_idx[hit_idx >= 0]:
    true_pe_counts[idx] += np.random.poisson(detector.a)

print(f"Истинная энергия: {E_true} МэВ")
print(f"Истинная позиция: {pos_true}")
print(f"Всего фотоэлектронов: {true_pe_counts.sum()}")

pmt_positions_np = detector.pmt_positions

with pm.Model() as reco_model:
    E = pm.Uniform('E', lower=0.1, upper=3.0)
    x = pm.Uniform('x', lower=-2.0, upper=2.0)
    y = pm.Uniform('y', lower=-2.0, upper=2.0)
    z = pm.Uniform('z', lower=-2.0, upper=2.0)

    source_pos = pt.stack([x, y, z])
    vecs_to_pmt = pmt_positions_np - source_pos
    dists_sq = pt.sum(vecs_to_pmt ** 2, axis=1)

    geom_factor = (detector.r ** 2) / (4 * dists_sq + 1e-9)
    mu_pmt = detector.a * source.Y * E * geom_factor

    obs = pm.Poisson('obs', mu=mu_pmt, observed=true_pe_counts)

    trace = pm.sample(2000, tune=1000, chains=2, return_inferencedata=True, progressbar=True)

print("\n--- Результаты реконструкции ---")
true_vals = {'E': E_true, 'x': pos_true[0], 'y': pos_true[1], 'z': pos_true[2]}
print(f"{'Параметр':8s} | {'Истина':>8s} | {'Среднее':>8s} | {'Стд.откл':>8s}")
print("-" * 45)
for var in ['E', 'x', 'y', 'z']:
    s = trace.posterior[var].values.flatten()
    print(f"{var:8s} | {true_vals[var]:8.3f} | {s.mean():8.3f} | {s.std():8.3f}")

fig, axes = plt.subplots(1, 4, figsize=(16, 3))

for i, var in enumerate(['E', 'x', 'y', 'z']):
    samples = trace.posterior[var].values.flatten()
    axes[i].hist(samples, bins=40, color='steelblue', edgecolor='black', alpha=0.7)
    axes[i].axvline(true_vals[var], color='red', linestyle='--', linewidth=2, label='Истина')
    axes[i].set_title(var)
    axes[i].legend()

plt.tight_layout()
plt.show()