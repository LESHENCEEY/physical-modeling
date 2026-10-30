import numpy as np

class Source:
    def __init__(self, Y: float):
        self.Y = Y

    def generate_event(self, E: float):
        """
        Генератор фотонов для события с энергией E.
        Возвращает фотоны порциями по 100 штук (yield).
        """
        N_mean = E * self.Y
        N_total = np.random.poisson(N_mean)
        
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

# Демонстрация работы
source = Source(Y=10000.0)
E = 1.0

print(f"Событие с энергией {E} МэВ:")

total_photons = 0
chunks_received = 0

for chunk in source.generate_event(E):
    chunks_received += 1
    total_photons += len(chunk)
    
    if chunks_received <= 3 or chunks_received > 95:
        print(f"  Получена порция {chunks_received}: форма {chunk.shape}")

print(f"\nВсего порций: {chunks_received}")
print(f"Всего фотонов: {total_photons} (ожидалось ~{E * source.Y:.0f})")

assert chunks_received == np.ceil(total_photons / 100)