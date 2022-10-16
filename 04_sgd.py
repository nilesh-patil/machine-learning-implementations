import numpy as np
import matplotlib.pyplot as plt

def mean_squared_error(X_b, y, theta):
    """
    Calculates the Mean Squared Error (MSE) cost.

    Args:
      X_b: Feature matrix with intercept term (n_samples, n_features + 1).
      y: Target values (n_samples, 1).
      theta: Model parameters (n_features + 1, 1).

    Returns:
      The MSE cost as a float.
    """

    m = len(y)
    predictions = X_b.dot(theta)
    cost = (1/(2*m)) * np.sum(np.square(predictions - y))

    return(cost)

def gradient_descent(X, y, learning_rate=0.01, n_iterations=100):
    """
    Performs Gradient Descent to learn theta by minimizing the cost function.

    Args:
      X: Feature matrix (n_samples, n_features).
      y: Target values (n_samples,).
      theta_initial: Initial guess for parameters (n_features + 1, 1).
      learning_rate: Step size for each iteration.
      n_iterations: Number of iterations to perform.

    Returns:
      A tuple containing:
        - final_theta: The learned parameters (n_features + 1, 1).
        - cost_history: A list of cost values for each iteration.
    """

    # Ensure X and y are NumPy arrays
    X = np.array(X)
    y = np.array(y).reshape(-1, 1) # Reshape y to be a column vector

    # Add intercept term (column of ones) to X
    X_b = np.c_[np.ones((len(X), 1)), X] # Add x0 = 1 to each instance

    m = len(y) # Number of samples
    theta = theta_initial.copy() # Start with initial theta
    cost_history = []

    for iteration in range(n_iterations):
        # Calculate predictions (hypothesis)
        predictions = X_b.dot(theta)

        # Calculate the error
        error = predictions - y

        # Calculate the gradient [1, 2]
        # Gradient = (1/m) * X_b^T * (predictions - y)
        gradients = (1/m) * X_b.T.dot(error)

        # Update theta [1, 2]
        theta = theta - learning_rate * gradients

        # Calculate and store the cost for this iteration [3]
        cost = mean_squared_error(X_b, y, theta)
        cost_history.append(cost)

    return(theta, cost_history)

# --- Example Usage with Simple Linear Regression Data ---

# Sample data (same as Simple Linear Regression example) [4]
hours_studied = np.array([5, 6, 7, 8, 9, 4, 3, 10, 11, 12])
test_results = np.array([60, 65, 70, 75, 80, 55, 50, 85, 90, 95])

# Prepare data for gradient descent
X_train = hours_studied.reshape(-1, 1) # Reshape to 2D array (1 feature)
y_train = test_results

# Set hyperparameters
learning_rate = 0.01
n_iterations = 100
# Initial theta (random or zeros). Size is n_features + 1 (for intercept)
initial_theta = np.zeros((X_train.shape[-1] + 1, 1))

# Run Gradient Descent
final_theta, cost_history = gradient_descent(X_train, y_train, initial_theta, learning_rate, n_iterations)

print(f"Gradient Descent Results:")
print(f"Final Theta (Intercept, Slope): \n{final_theta}")
print(f"Final Cost (MSE): {cost_history[-1]:.4f}")

# Compare with Normal Equation result (from Simple Linear Regression implementation)
# Note: This requires running the simple_linear_regression function from implementation 01
# intercept_ols, slope_ols = simple_linear_regression(hours_studied, test_results)
# print(f"\nOLS Normal Equation Results (for comparison):")
# print(f"Intercept: {intercept_ols:.4f}, Slope: {slope_ols:.4f}")


# --- Optional: Visualize Cost Reduction ---
plt.figure(figsize=(12, 5))

plt.subplot(1,2,1)
plt.plot(range(n_iterations), cost_history, 'b-')
plt.xlabel("Iterations")
plt.ylabel("Cost (MSE)")
plt.title("Gradient Descent: Cost Reduction Over Iterations")
plt.grid(True)

# --- Optional: Visualize the final regression line ---
X_plot = np.array([[min(hours_studied)], [max(hours_studied)]]) # Points for line
X_plot_b = np.c_[np.ones((2, 1)), X_plot] # Add intercept
y_plot = X_plot_b.dot(final_theta)

plt.subplot(1,2,2)
plt.scatter(hours_studied, test_results, color='blue', label='Actual Data Points')
plt.plot(X_plot, y_plot, color='red', label='Regression Line (Gradient Descent)')
plt.xlabel("Hours Studied")
plt.ylabel("Test Results")
plt.title("Simple Linear Regression using Gradient Descent")
plt.legend()
plt.grid(True)


plt.show()