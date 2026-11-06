"""
Python script converted directly from IDC.ipynb.

The code cells are kept in their original notebook order.
Markdown cells have been omitted.
"""
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent / 'results'
RESULTS_DIR.mkdir(exist_ok=True)


# ====================================================================
# Original notebook code cell 1
# ====================================================================
m = 1
c = 0.5
k = 4
print('Mass:',m)
print('Damping Coefficient:',c)
print('Stiffness:',k)


# ====================================================================
# Original notebook code cell 2
# ====================================================================
import numpy as np

A = np.array([
    [0, 1],
    [-k / m, -c / m]
])

print("State-space matrix A:")
print(A)


# ====================================================================
# Original notebook code cell 3
# ====================================================================
eigenvalues, eigenvectors = np.linalg.eig(A)

print("Eigenvalues:")
print(eigenvalues)


# ====================================================================
# Original notebook code cell 4
# ====================================================================
def dynamics(t, X):
    x1 = X[0]
    x2 = X[1]

    d_x1 = x2
    d_x2 = -(k / m) * x1 - (c / m) * x2

    return d_x1, d_x2


# ====================================================================
# Original notebook code cell 5
# ====================================================================
from scipy.integrate import solve_ivp

x0 = [1.0, 0.0]

t_start = 0.0
t_end = 10.0

t_eval = np.linspace(t_start, t_end, 1000)


# ====================================================================
# Original notebook code cell 6
# ====================================================================
sol = solve_ivp(
    dynamics,
    [t_start, t_end],
    x0,
    t_eval=t_eval
)


# ====================================================================
# Original notebook code cell 7
# ====================================================================
t = sol.t

x = sol.y[0]

d_x = sol.y[1]
print("Number of time points:", len(t))
print("Initial position:", x[0])
print("Final position:", x[-1])
print("Initial velocity:", d_x[0])
print("Final velocity:", d_x[-1])


# ====================================================================
# Original notebook code cell 8
# ====================================================================
import matplotlib.pyplot as plt

plt.subplot(2,1,1)
plt.plot(t, x)
plt.xlabel('Time(s)')
plt.ylabel('Position(m)')
plt.title('Mass-Spring-Damper Response')
plt.grid()


# ====================================================================
# Original notebook code cell 9
# ====================================================================
plt.subplot(2,1,2)
plt.plot(t, d_x)
plt.xlabel('Time')
plt.ylabel('Velocity')
plt.grid()
plt.tight_layout()
plt.savefig(RESULTS_DIR / 'time_response.png', dpi=300, bbox_inches='tight')
plt.savefig(RESULTS_DIR / 'phase_portrait.png', dpi=300, bbox_inches='tight')
plt.savefig(RESULTS_DIR / 'mechanical_energy.png', dpi=300, bbox_inches='tight')
plt.savefig(RESULTS_DIR / 'dataset_relationships.png', dpi=300, bbox_inches='tight')
plt.savefig(RESULTS_DIR / 'actual_vs_predicted_velocity.png', dpi=300, bbox_inches='tight')
plt.savefig(RESULTS_DIR / 'actual_vs_predicted_acceleration.png', dpi=300, bbox_inches='tight')
plt.show()


# ====================================================================
# Original notebook code cell 10
# ====================================================================
plt.plot(x, d_x)
plt.xlabel('Position')
plt.ylabel('Velocity')
plt.grid()
plt.tight_layout()
plt.show()


# ====================================================================
# Original notebook code cell 11
# ====================================================================
KE = 0.5*m*(d_x)**2
PE = 1/2*k*x**2
E = PE + KE

plt.figure(figsize=(10, 10))
plt.plot(t, KE, label='Kinetic Energy', color='#a7d975')
plt.plot(t, PE, label='Potential Energy')
plt.plot(t, E, label='Total energy')

plt.xlabel('Time(s)')
plt.ylabel('Energy(J)')
plt.title('MSD System Energy Analysis')

plt.grid(True, alpha=0.3)
plt.legend()
plt.show()


# ====================================================================
# Original notebook code cell 12
# ====================================================================
# Let a denote acceleration
a = -(k / m) * x - (c / m) * d_x
print("First 5 acceleration values:", a[:5])


# ====================================================================
# Original notebook code cell 13
# ====================================================================
X_data = np.column_stack((x, d_x))
Y_data = np.column_stack((d_x, a))


# ====================================================================
# Original notebook code cell 14
# ====================================================================
print("X_data shape:", X_data.shape)
print("Y_data shape:", Y_data.shape)

print("\nFirst 5 input samples:")
print(X_data[:5])

print("\nFirst 5 target samples:")
print(Y_data[:5])


# ====================================================================
# Original notebook code cell 15
# ====================================================================
plt.figure(figsize=(10, 5))
plt.subplot(1,2,1)
plt.scatter(X_data[:,0], Y_data[:,0], s=10)
plt.xlabel('Position')
plt.ylabel('Velocity')
plt.title('Position and Velocity')
plt.grid(True, alpha=0.3)
plt.subplot(1, 2, 2)
plt.scatter(X_data[:, 0], Y_data[:, 1], s=10)
plt.tight_layout()
plt.show()


# ====================================================================
# Original notebook code cell 16
# ====================================================================
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(
    X_data,
    Y_data,
    test_size=0.2,
    random_state=50
)
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("Y_train shape:", Y_train.shape)
print("Y_test shape:", Y_test.shape)


# ====================================================================
# Original notebook code cell 17
# ====================================================================
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("First 5 scaled training samples:")
print(X_train_scaled[:5])


# ====================================================================
# Original notebook code cell 18
# ====================================================================
from sklearn.linear_model import LinearRegression

model = LinearRegression()


# ====================================================================
# Original notebook code cell 19
# ====================================================================
model.fit(X_train_scaled, Y_train)


# ====================================================================
# Original notebook code cell 20
# ====================================================================
Y_pred = model.predict(X_test_scaled)

print("Y_pred shape:", Y_pred.shape)
print("\nFirst 5 predicted samples:")
print(Y_pred[:5])


# ====================================================================
# Original notebook code cell 21
# ====================================================================
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

mae = mean_absolute_error(Y_test, Y_pred)
mse = mean_squared_error(Y_test, Y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(Y_test, Y_pred)

print("MAE :", mae)
print("MSE :", mse)
print("RMSE:", rmse)
print("R²  :", r2)


# ====================================================================
# Original notebook code cell 22
# ====================================================================
mae_velocity = mean_absolute_error(
    Y_test[:, 0],
    Y_pred[:, 0]
)

mae_acceleration = mean_absolute_error(
    Y_test[:, 1],
    Y_pred[:, 1]
)

print("MAE - Velocity:", mae_velocity)
print("MAE - Acceleration:", mae_acceleration)


# ====================================================================
# Original notebook code cell 23
# ====================================================================
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.scatter(Y_test[:, 0], Y_pred[:, 0], s=10)
plt.xlabel('True Velocity')
plt.ylabel('Predicted Velocity')
plt.title('True vs Predicted Velocity')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.scatter(Y_test[:, 1], Y_pred[:, 1], s=10)
plt.xlabel('True Acceleration')
plt.ylabel('Predicted Acceleration')
plt.title('True vs Predicted Acceleration')
plt.grid(True)

plt.tight_layout()
plt.show()


# ====================================================================
# Original notebook code cell 24
# ====================================================================
print("Learned coefficients:")
print(model.coef_)

print("\nLearned intercept:")
print(model.intercept_)


# ====================================================================
# Original notebook code cell 25
# ====================================================================
physical_coefficients = model.coef_ / scaler.scale_

print("Recovered physical coefficients:")
print(physical_coefficients)


# ====================================================================
# Original notebook code cell 26
# ====================================================================
A_physical = A

A_ml = physical_coefficients

print("Physical coefficient matrix:")
print(A_physical)

print("\nMachine learning coefficient matrix:")
print(A_ml)

print("\nAbsolute difference:")
print(np.abs(A_physical - A_ml))


# ====================================================================
# Original notebook code cell 27
# ====================================================================
# Measuring the maximum error 
coefficient_error = np.max(A_physical - A_ml)


print("Maximum coefficient error:", coefficient_error)


# ====================================================================
# Original notebook code cell 28
# ====================================================================
import pandas as pd

comparison = pd.DataFrame(
    {
        "Physical": A_physical.flatten(),
        "Machine Learning": A_ml.flatten()
    },
    index=[
        "x -> dx/dt",
        "v -> dx/dt",
        "x -> d2x/dt2",
        "v -> d2x/dt2"
    ]
)

comparison["Absolute Error"] = np.abs(
    comparison["Physical"] - comparison["Machine Learning"]
)

print(comparison)
