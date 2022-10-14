"""
Simple linear regression implementation using 
NumPy, Scipy and visualization with Matplotlib.
"""


import numpy as np
import matplotlib.pyplot as plt


def simple_linear_regression(x, y):
  """
  Calculates the slope (B1) and intercept (B0) for a simple linear regression
  using the Ordinary Least Squares (OLS) method with NumPy.

  Args:
    x: A NumPy array of the independent variable values.
    y: A NumPy array of the dependent variable values.

  Returns:
    A tuple containing the intercept (B0) and slope (B1).
  """
  # Ensure x and y are NumPy arrays
  x = np.array(x)
  y = np.array(y)

  # Calculate the means of x and y [1]
  x_mean = np.mean(x)
  y_mean = np.mean(y)

  # Calculate the terms needed for the slope (B1) [1]
  # Numerator: Sum of (x_i - x_mean) * (y_i - y_mean)
  numerator = np.sum((x - x_mean) * (y - y_mean))

  # Denominator: Sum of (x_i - x_mean)^2
  denominator = np.sum((x - x_mean) ** 2)

  # Calculate the slope (B1) [1]
  # Handle potential division by zero if all x values are the same
  if denominator == 0:
      slope = 0 # Or handle as an error/special case
  else:
      slope = numerator / denominator

  # Calculate the intercept (B0) [1]
  intercept = y_mean - slope * x_mean

  return intercept, slope

# --- Example Usage ---

# Sample data (e.g., hours studied vs. test results) [2]
hours_studied = np.array([3, 4, 5, 6, 7, 2, 8, 1, 9, 10])
# Corrected: Added sample data for test_results
test_results = np.array( 3*hours_studied + 2*hours_studied**2 + 2)

# Calculate intercept and slope
intercept, slope = simple_linear_regression(hours_studied, test_results)

print(f"Intercept (B0): {intercept:.4f}")
print(f"Slope (B1): {slope:.4f}")

# --- Optional: Make predictions and visualize ---

# Predict values using the calculated intercept and slope
y_pred = intercept + slope * hours_studied
print(f"Predicted results: {np.round(y_pred, 2)}")

# Plot the original data and the regression line [8]
plt.scatter(hours_studied, test_results, color='blue', label='Actual Data Points')
plt.plot(hours_studied, y_pred, color='red', label='Regression Line')
plt.xlabel("Hours Studied")
plt.ylabel("Test Results")
plt.title("Simple Linear Regression (NumPy from Scratch)")
plt.legend()
plt.grid(True)
plt.show()