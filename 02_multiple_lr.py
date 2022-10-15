import numpy as np
import matplotlib.pyplot as plt

def multiple_linear_regression(X, y):
    """
    Calculates the coefficients (including intercept) for multiple linear regression
    using the Normal Equation method with NumPy.

    Args:
      X: A NumPy array of the independent variables (features).
         Shape should be (n_samples, n_features).
      y: A NumPy array of the dependent variable values.
         Shape should be (n_samples,).

    Returns:
      A NumPy array containing the intercept and coefficients (theta).
      The first element is the intercept, followed by coefficients for each feature.
      Returns None if the matrix (X^T * X) is singular (non-invertible).
    """
    # Ensure X and y are NumPy arrays
    X = np.array(X)
    y = np.array(y)

    # Add intercept term (column of ones) to X [2, 3]
    # Reshape y to be a column vector
    y = y.reshape(-1, 1)
    X_b = np.c_[np.ones((X.shape[0], 1)), X] # Add x0 = 1 to each instance

    # Calculate theta using the Normal Equation: theta = (X^T * X)^(-1) * X^T * y [1, 2, 3]

    try:
        # Calculate (X^T * X)
        xtx = np.dot(X_b.T, X_b)
        # Calculate the inverse of (X^T * X)
        xtx_inv = np.linalg.inv(xtx)
        # Calculate X^T * y
        xty = np.dot(X_b.T, y)
        # Calculate theta
        theta = np.dot(xtx_inv, xty)
    except np.linalg.LinAlgError:
        # Handle cases where the matrix is singular (cannot be inverted)
        print("Error: Matrix (X^T * X) is singular and cannot be inverted.")
        print("This might happen if features are perfectly correlated or if n_features >= n_samples.")
        theta = None

    return(theta)

# --- Example Usage ---

# Sample data (e.g., predicting house price based on size and number of bedrooms)
# Features: Size (sq ft), Number of Bedrooms
X_features = np.array([
    [2104, 3, 1],
    [1600, 3, 1],
    [2400, 3, 1],
    [1416, 2, 1],
    [3000, 4, 1],
    [1985, 4, 1],
    [1534, 3, 1],
    [1427, 3, 1],
    [1380, 3, 2],
    [1494, 3, 2],
    [2940, 4, 2],
    [2000, 3, 2]
])

# Target: House Price ($1000s)
y_target = np.array([400, 330, 369, 232, 540, 300, 315, 199, 212, 242, 400, 350])

# Calculate coefficients (theta)
theta = multiple_linear_regression(X_features, y_target)

if theta is not None:
    print(f"Calculated Theta (Intercept and Coefficients):\n{theta}")

    # The first value is the intercept (b0)
    # Subsequent values are coefficients for each feature (b1, b2,...)
    intercept = theta[0][0]
    coefficients = theta[1:].flatten() # Flatten to 1D array

    print(f"\nIntercept (b0): {intercept:.4f}")
    for i, coef in enumerate(coefficients):
        print(f"Coefficient for feature {i+1} (b{i+1}): {coef:.4f}")

    # --- Optional: Make predictions ---
    # Add intercept term to the features for prediction
    X_features_b = np.c_[np.ones((X_features.shape[0], 1)), X_features]
    y_pred = np.dot(X_features_b, theta)

    print(f"\nActual Prices: {y_target}")
    print(f"Predicted Prices: {np.round(y_pred.flatten(), 2)}")

    # --- Optional: Visualize (only feasible for 1 or 2 features) ---
    # For 2 features, we could attempt a 3D plot, but it's complex.
    # Here, we'll just plot predicted vs actual for simplicity.
    plt.scatter(y_target, y_pred.flatten(), color='blue', label='Predicted vs Actual')
    plt.plot([min(y_target), max(y_target)], [min(y_target), max(y_target)], color='red', linestyle='--', label='Ideal Fit') # Line y=x
    plt.xlabel("Actual Prices ($1000s)")
    plt.ylabel("Predicted Prices ($1000s)")
    plt.title("Multiple Linear Regression: Actual vs Predicted Prices")
    plt.legend()
    plt.grid(True)
    plt.show()