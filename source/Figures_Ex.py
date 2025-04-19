# Visualization of Temporal Synchrony and Signal Displacement in Dyadic Interactions

# Code in this file is NOT part of our human motion cross-corrleation 
# example. It uses random noise in order to generate explanatory plots
# included in the article that goes along with our code example. It's
# included for completeness.

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline

# Define the lag range for the x-axis
lags = np.linspace(-1.5, 1.5, 100)
num_points = 12  # Number of points to define the spline

# Specify zero-crossing points and edge points to shape the curve
zero_crossings = np.array([-1.2, 0.0, 1.2])
near_edge_points = np.array([-1.5, 1.5])  # Ensure the curve starts and ends near zero

# Generate additional random points between the zero-crossings
random_points = np.sort(np.concatenate((
    zero_crossings,
    near_edge_points,
    np.random.uniform(-1.5, 1.5, num_points - len(zero_crossings) - len(near_edge_points))
)))

# Generate random values for the y-axis
random_values = np.random.rand(num_points) * 0.4 - 0.2
random_values[np.isin(random_points, zero_crossings)] = 0  # Force zero-crossings
random_values[np.isin(random_points, near_edge_points)] = 0  # Keep edges close to zero

# Adjust the mean of the signal to be close to zero
random_values -= np.mean(random_values)

# Create the cubic spline function
cs = CubicSpline(random_points, random_values, bc_type='natural')

# Evaluate the spline across the defined lags
experimental_after_enlarged = cs(lags)

# Plot the spline signal
plt.figure(figsize=(10, 6))
plt.plot(lags, experimental_after_enlarged, color='#e8a8b1')  # pink curve

# Add error bands to simulate confidence intervals
plt.fill_between(lags, experimental_after_enlarged - 0.035, experimental_after_enlarged + 0.035, color='#e8a8b1', alpha=0.3)

# Add labels and title
plt.title('Graphic Example')
plt.xlabel('Lag (seconds)')
plt.ylabel('Pearson Correlation')

# Annotate imitation phases
plt.text(-0.75, 0.17, '"A" imitates "B"', color='black', ha='center', bbox=dict(facecolor='white', edgecolor='black'))
plt.text(0.75, 0.17, '"B" imitates "A"', color='black', ha='center', bbox=dict(facecolor='white', edgecolor='black'))

# Mark immediate coordination
plt.axvline(0, color='black', linestyle='--')
plt.text(0, 0.19, 'Immediate Coordination', color='black', ha='center', bbox=dict(facecolor='white', edgecolor='black'))

# Display legend and grid
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

# ---------- Second plot: Time-shifted signals ----------

# Define time axes
x_a = np.linspace(0, 10, 1000)
x_b = np.linspace(1.5, 11.5, 1000)  # B starts 1.5 seconds later

# Define sinusoidal signals with different offsets
y_a = 0.3 * np.sin(5 * x_a) + 3      # A with smaller amplitude
y_b = 0.3 * np.sin(5 * (x_b - 1.5)) + 1.8  # B is a shifted version of A

# Create the plot
fig, ax = plt.subplots()
ax.set_title('Displacement in Time')

# Plot both signals
ax.plot(x_a, y_a, label='Original "A" signal', color='orange')
ax.plot(x_b, y_b, label='Signal "B" shifted to the right by 1.5 seconds', color='black')

# Highlight the overlapping region
ax.fill_betweenx([0, 4], 1.5, 10, color='#D3D3D3', alpha=0.5)

# Annotate the common signal zone
plt.text((1.5 + 10) / 2, 3.7, 'Common Signal', color='black', ha='center', bbox=dict(facecolor='white', edgecolor='black'))

# Add legend
ax.legend(loc='lower center')

# Add vertical lines to mark boundaries of overlap
ax.axvline(x=1.5, color='black', linestyle='--')
ax.axvline(x=10, color='black', linestyle='--')

# Axis labels and formatting
ax.set_xlabel('Offset in seconds [s]')
ax.set_yticks([])  # Hide y-axis values
ax.set_xlim(0, 11.5)
ax.set_ylim(0.5, 4)

plt.show()
