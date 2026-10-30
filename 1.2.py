import numpy as np

class Source:
    def __init__(self, Y: float):
        self.Y = Y
    def generate_event(self, E: float) -> np.ndarray:
        N_mean = E * self.Y
        N = np.random.poisson(N_mean)
        if N == 0:
            return np.empty((0, 3))
        phi = np.random.uniform(0, 2 * np.pi, size=N)
        cos_theta = np.random.uniform(-1, 1, size=N)
        sin_theta = np.sqrt(1 - cos_theta**2)
        x = sin_theta * np.cos(phi)
        y = sin_theta * np.sin(phi)
        z = cos_theta
        return np.vstack((x, y, z)).T
source = Source(Y=10000.0)
E = 1.0 
directions = source.generate_event(E)
print(f"Энергия события: {E} МэВ")
print(f"Ожидаемое число фотонов: {E * source.Y}")
print(f"Сгенерировано фотонов (Пуассон): {len(directions)}")
print(f"Форма массива направлений: {directions.shape}")
mean_vec = directions.mean(axis=0)
print(f"Средний вектор направлений (должен быть ~0): {mean_vec}")
norms = np.linalg.norm(directions, axis=1)
print(f"Средняя норма вектора (должна быть 1): {norms.mean():.4f}")
print(f"Макс. отклонение нормы от 1: {np.max(np.abs(norms - 1.0)):.2e}")