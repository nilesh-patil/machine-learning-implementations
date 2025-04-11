import numpy as np
import matplotlib.pyplot as plt

def create_polynomial_features(x, degree):
    """
    Generates polynomial features for a given 1D input array.

    Args:
      x: A NumPy array of the independent variable (must be 1D).
      degree: The degree of the polynomial.

    Returns:
      A NumPy array with polynomial features (x, x^2,..., x^degree).
      Shape will be (n_samples, degree).
    """
    x = np.array(x)
    if x.ndim!= 1:
        raise ValueError("Input array x must be 1-dimensional for this simple implementation.")

    X_poly = np.ones((len(x), 1)) # Start with x^0 (intercept handled later)
    for d in range(1, degree + 1):
        X_poly = np.column_stack((X_poly, x**d))

    # Remove the initial column of ones (x^0) as the intercept
    # will be added in the regression function itself.
    return(X_poly[:, 1:]) # Return features from x^1 to x^degree

def polynomial_regression(x, y, degree):
    """
    Calculates the coefficients for polynomial regression using the Normal Equation.

    Args:
      x: A NumPy array of the independent variable (1D).
      y: A NumPy array of the dependent variable values.
      degree: The degree of the polynomial to fit.

    Returns:
      A NumPy array containing the intercept and coefficients (theta).
      The first element is the intercept, followed by coefficients for x, x^2,..., x^degree.
      Returns None if the matrix (X^T * X) is singular (non-invertible).
    """
    # Ensure y is a NumPy array and reshape it to a column vector
    y = np.array(y).reshape(-1, 1)

    # 1. Create Polynomial Features [1]
    X_poly_features = create_polynomial_features(x, degree)

    # 2. Add intercept term (column of ones) to the polynomial features matrix [2]
    X_b = np.c_[np.ones((len(X_poly_features), 1)), X_poly_features] # Add x0 = 1

    # 3. Calculate theta using the Normal Equation: theta = (X^T * X)^(-1) * X^T * y [3, 2]
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
        # Handle cases where the matrix is singular
        print("Error: Matrix (X^T * X) is singular and cannot be inverted.")
        theta = None

    return(theta)


# --- Example Usage ---

# Sample data with a non-linear (quadratic) relationship
np.random.seed(0) # for reproducibility
x_data = np.linspace(-3, 3, 100)
# y = 0.5 * x^2 + x + 2 + noise
y_data = 0.5 * x_data**2 + x_data + 2 + np.random.randn(100) * 1.5

# Choose the degree of the polynomial
poly_degree = 2

# Calculate coefficients (theta)
theta = polynomial_regression(x_data, y_data, poly_degree)

if theta is not None:
    print(f"Calculated Theta (Intercept and Coefficients for degree {poly_degree}):\n{theta}")

    intercept = theta[0][0]
    coefficients = theta[1:].flatten()

    print(f"\nIntercept (b0): {intercept:.4f}")
    for i, coef in enumerate(coefficients):
        print(f"Coefficient for x^{i+1} (b{i+1}): {coef:.4f}")

    # --- Optional: Make predictions and visualize ---

    # To predict, create the polynomial feature matrix including the intercept term
    X_data_poly_features = create_polynomial_features(x_data, poly_degree)
    X_data_b = np.c_[np.ones((len(X_data_poly_features), 1)), X_data_poly_features]

    y_pred = np.dot(X_data_b, theta)

    # Sort values for plotting the curve smoothly
    sort_axis = np.argsort(x_data)
    x_data_sorted = x_data[sort_axis]
    y_pred_sorted = y_pred[sort_axis]

    # Plot the original data and the polynomial regression curve [4]
    plt.scatter(x_data, y_data, color='blue', label='Actual Data Points', s=10)
    plt.plot(x_data_sorted, y_pred_sorted, color='red', label=f'Polynomial Regression (Degree {poly_degree})')
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Polynomial Regression (NumPy from Scratch)")
    plt.legend()
    plt.grid(True)
    plt.show()