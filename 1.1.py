import numpy as np
import k3d

class Detector:
    def __init__(self, R: float, N: int, r: float, a: float):
        self.R = R
        self.N = N
        self.r = r
        self.a = a
        self.pmt_positions = self._generate_pmt_positions()
    
    def _generate_pmt_positions(self) -> np.ndarray:
        indices = np.arange(0, self.N, dtype=float) + 0.5
        phi = np.arccos(1 - 2 * indices / self.N)  # Полярный угол (0 до pi)
        theta = np.pi * (1 + 5**0.5) * indices     # Азимутальный угол (золотое сечение)
        x = np.cos(theta) * np.sin(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(phi)
        positions = np.vstack((x, y, z)).T * self.R
        return positions

    def is_inside_pmt(self, point: np.ndarray) -> bool:
        norm = np.linalg.norm(point)
        if np.abs(norm - self.R) > 1e-6:
            point = point / norm * self.R
        distances = np.linalg.norm(self.pmt_positions - point, axis=1)
        return np.any(distances < self.r)
    
    def visualize(self):
        plot = k3d.plot()
        u, v = np.mgrid[0:2*np.pi:60j, 0:np.pi:30j]
        x = self.R * np.cos(u) * np.sin(v)
        y = self.R * np.sin(u) * np.sin(v)
        z = self.R * np.cos(v)
        sphere_points = np.vstack((x.flatten(), y.flatten(), z.flatten())).T
        
        plot += k3d.points(
            sphere_points, 
            point_size=0.02, 
            color=0x00ff00,  # Зеленый цвет для поверхности
            name="Detector Surface (R=2m)"
        )
        plot += k3d.points(
            self.pmt_positions, 
            point_size=0.15,  # Размер точек ФЭУ
            color=0xff0000,   # Красный цвет для ФЭУ
            name=f"PMTs (N={self.N}, r={self.r}m)"
        )
        plot += k3d.points(
            np.array([[0, 0, 0]]), 
            point_size=0.1, 
            color=0xffffff, 
            name="Origin"
        )
        plot.camera = [5, 5, 5, 0, 0, 0, 0, 0, 1]  # [px, py, pz, cx, cy, cz, ux, uy, uz]
        plot.axes = [True, True, True]
        plot.display()
det = Detector(R=2.0, N=40, r=0.1, a=0.5)
print(f"Создан детектор: R={det.R} м, N={det.N} ФЭУ, r={det.r} м")
print(f"Координаты первых 3 ФЭУ:\n{det.pmt_positions[:3]}")
print(f"Форма массива координат: {det.pmt_positions.shape}")
pmt_center = det.pmt_positions[0]
print(f"\nТочка {pmt_center} принадлежит ФЭУ? {det.is_inside_pmt(pmt_center)}")
point_outside = pmt_center + np.array([0.2, 0, 0])
point_outside = point_outside / np.linalg.norm(point_outside) * det.R  # Проекция на сферу
print(f"Точка {point_outside} принадлежит ФЭУ? {det.is_inside_pmt(point_outside)}")
det.visualize()