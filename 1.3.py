import numpy as np
import matplotlib.pyplot as plt

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

    def is_inside_pmt(self, point: np.ndarray) -> bool:
        norm = np.linalg.norm(point)
        if abs(norm - self.R) > 1e-6:
            point = point / norm * self.R
        distances = np.linalg.norm(self.pmt_positions - point, axis=1)
        return bool(np.any(distances < self.r))

    def trace_photon(self, origin: np.ndarray, direction: np.ndarray) -> int | None:
        b = 2.0 * np.dot(origin, direction)
        c = np.dot(origin, origin) - self.R ** 2
        discr = b ** 2 - 4 * c
        if discr < 0:
            return None
        t = (-b + np.sqrt(discr)) / 2.0
        if t <= 0:
            return None
        hit_point = origin + t * direction
        distances = np.linalg.norm(self.pmt_positions - hit_point, axis=1)
        idx = int(np.argmin(distances))
        return idx if distances[idx] < self.r else None

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

    def visualize(self):
        import k3d
        plot = k3d.plot()
        u, v = np.mgrid[0:2*np.pi:60j, 0:np.pi:30j]
        x = self.R * np.cos(u) * np.sin(v)
        y = self.R * np.sin(u) * np.sin(v)
        z = self.R * np.cos(v)
        sphere = np.vstack((x.flatten(), y.flatten(), z.flatten())).T.astype(np.float32)

        plot += k3d.points(sphere, point_size=0.02, color=0x00ff00, name="Detector")
        plot += k3d.points(self.pmt_positions.astype(np.float32), point_size=0.15, color=0xff0000, name="PMTs")
        plot += k3d.points(np.array([[0, 0, 0]], dtype=np.float32), point_size=0.1, color=0xffffff, name="Origin")
        plot.camera = [5, 5, 5, 0, 0, 0, 0, 0, 1]
        plot.axes = [True, True, True]
        plot.display()


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
E = 1.0

detector = Detector(R=R, N=N_pmt, r=r, a=a)
source = Source(Y=Y)
origin = np.array([0.0, 0.0, 0.0])

Q_expected = a * Y * E * (N_pmt * r**2) / (4 * R**2)
print(f"Q_теор = {Q_expected:.2f}")

N_events = 1000
charges = []
n_photons_list = []
n_hits_list = []

for _ in range(N_events):
    directions = source.generate_event(E)
    n_photons_list.append(len(directions))
    hit_idx = detector.trace_photons(origin, directions)
    hits = hit_idx[hit_idx >= 0]
    n_hits_list.append(len(hits))
    q = sum(detector.generate_photoelectrons() for _ in hits)
    charges.append(q)

charges = np.array(charges)
n_photons_list = np.array(n_photons_list)
n_hits_list = np.array(n_hits_list)

print(f"<N_фотонов> = {n_photons_list.mean():.0f}")
print(f"<N_попаданий> = {n_hits_list.mean():.0f}")
print(f"<Q> = {charges.mean():.2f}")
print(f"std(Q) = {charges.std():.2f}")
print(f"<Q>/Q_теор = {charges.mean() / Q_expected:.3f}")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].hist(charges, bins=40, color='steelblue', edgecolor='black')
axes[0].axvline(Q_expected, color='red', linestyle='--', label=f'Теория = {Q_expected:.1f}')
axes[0].axvline(charges.mean(), color='orange', linestyle='--', label=f'Среднее = {charges.mean():.1f}')
axes[0].set_xlabel('Q')
axes[0].set_ylabel('События')
axes[0].set_title('Заряд')
axes[0].legend()

axes[1].hist(n_photons_list, bins=40, color='green', edgecolor='black')
axes[1].axvline(Y * E, color='red', linestyle='--', label=f'Y·E = {Y*E:.0f}')
axes[1].set_xlabel('N фотонов')
axes[1].set_ylabel('События')
axes[1].set_title('Фотоны')
axes[1].legend()

axes[2].hist(n_hits_list, bins=40, color='purple', edgecolor='black')
axes[2].axvline(n_hits_list.mean(), color='red', linestyle='--', label=f'Среднее = {n_hits_list.mean():.0f}')
axes[2].set_xlabel('N попаданий')
axes[2].set_ylabel('События')
axes[2].set_title('Попадания в ФЭУ')
axes[2].legend()

plt.tight_layout()
plt.show()