import numpy as np
import matplotlib.pyplot as plt
import pymc as pm
import pytensor.tensor as pt

R = 2.0
N_pmt = 40
r = 0.1
a = 0.5
Y = 10000.0
E_fixed = 1.0

Q_mean_analytic = a * Y * E_fixed * (N_pmt * r**2) / (4 * R**2)
print(f"Аналитическое среднее для центра: {Q_mean_analytic:.2f}")

with pm.Model() as detector_model:
    Q = pm.Poisson('Q', mu=Q_mean_analytic, observed=None)
    trace_center = pm.sample(5000, tune=1000, chains=2, return_inferencedata=True, progressbar=False)

q_pymc = trace_center.posterior['Q'].values.flatten()

print(f"PyMC среднее: {q_pymc.mean():.2f}")
print(f"PyMC std: {q_pymc.std():.2f}")

np.random.seed(42)
q_mc = np.random.poisson(Q_mean_analytic, size=len(q_pymc))

plt.figure(figsize=(8, 4))
plt.hist(q_pymc, bins=40, alpha=0.5, color='blue', label='PyMC (Poisson)', density=True)
plt.hist(q_mc, bins=40, alpha=0.5, color='orange', label='Задача 4 (Монте-Карло)', density=True)
plt.axvline(Q_mean_analytic, color='red', linestyle='--', label=f'Теория = {Q_mean_analytic:.1f}')
plt.xlabel('Полный заряд Q')
plt.ylabel('Плотность вероятности')
plt.title('Сравнение отклика детектора: PyMC vs Монте-Карло')
plt.legend()
plt.grid(alpha=0.3)
plt.show()

L_att = 8.0

with pm.Model() as detector_att_model:
    z_source = pm.Uniform('z_source', lower=-1.5, upper=1.5)
    d_avg = pt.sqrt(R**2 + z_source**2)
    survival_prob = pt.exp(-d_avg / L_att)
    Q_mean_att = Q_mean_analytic * survival_prob
    Q_att = pm.Poisson('Q_att', mu=Q_mean_att, observed=None)
    trace_att = pm.sample(5000, tune=1000, chains=2, return_inferencedata=True, progressbar=False)

q_att_pymc = trace_att.posterior['Q_att'].values.flatten()
z_samples = trace_att.posterior['z_source'].values.flatten()

print(f"PyMC (с затуханием) среднее Q: {q_att_pymc.mean():.2f}")
print(f"PyMC (с затуханием) std: {q_att_pymc.std():.2f}")

plt.figure(figsize=(8, 4))
plt.hist(q_pymc, bins=40, alpha=0.5, color='blue', label='Без затухания', density=True)
plt.hist(q_att_pymc, bins=40, alpha=0.5, color='red', label=f'С затуханием (L={L_att} м)', density=True)
plt.xlabel('Полный заряд Q')
plt.ylabel('Плотность вероятности')
plt.title('Влияние затухания на отклик детектора (PyMC)')
plt.legend()
plt.grid(alpha=0.3)
plt.show()

plt.figure(figsize=(6, 3))
plt.hist(z_samples, bins=30, color='green', edgecolor='black')
plt.xlabel('Положение источника по Z, м')
plt.ylabel('Число семплов')
plt.title('Постериор распределения координаты Z')
plt.show()