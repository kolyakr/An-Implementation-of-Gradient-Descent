# An Implementation of Gradient Descent 📉

This project demonstrates the core machine learning optimization technique, **Gradient Descent**, used to find the best coefficients for a linear regression model by minimizing the Residual Sum of Squares (RSS).

---

## 1. The Problem: Finding Unknown Coefficients

In linear regression, our goal is to model the relationship between a predictor $X$ and a response $y$ using a linear equation.

The **Population Regression Line** (the true relationship) is defined as:
$$y = \beta_{0} + \beta_{1}X + e$$
_Example:_ $$y = 7 - 5X + e$$

| Image 1: Generated Data and True Line |          Image 2: Conceptual Scatter Plot           |
| :-----------------------------------: | :-------------------------------------------------: |
| ![Generated Data](./images/image.png) | ![Population Regression Line](./images/image-1.png) |

In the real world, we only have the data points ($\mathbf{X}$ and $\mathbf{y}$) and **do not know the true coefficients ($\beta_0, \beta_1, \dots$)**. Our task is to **estimate** these coefficients from the data.

---

## 2. The Solution: Minimizing the Cost Function

### The Cost Function: Residual Sum of Squares (RSS)

To measure how good our estimated coefficients ($\hat{\beta}$) are, we use a **Loss (Cost) Function**. Here, we use the **Residual Sum of Squares (RSS)**:

**Model Prediction ($\hat{y}_i$):**
$$\hat{y}_i = \beta_0 + \sum_{j=1}^p \beta_j x_{ij}$$

**Loss (RSS):**

$$
\text{RSS}(\beta) = \sum_{i=1}^n (y_i - \hat{y}_i)^2 = \sum_{i=1}^n \left(y_i - \beta_0 - \sum_{j=1}^p \beta_j x_{ij}\right)^2
$$

The input to $\text{RSS}(\beta)$ is the set of coefficients $\beta$, and our objective is to find the combination of $\beta$ that **minimizes this cost**.

### Finding the Minimum via Derivatives

We can find the minimum by checking where the derivative (or slope) of the cost function is zero:

- A **Negative Slope** ($\beta_a$) means the cost is decreasing; we need to keep moving.
- A **Positive Slope** ($\beta_c$) means the cost is increasing; we've gone too far.
- A **Zero Slope** ($\beta_i$) signals a potential minimum.

![RSS Derivative Example](./images/image-2.png)
_Image 3: 2D RSS Curve illustrating different slopes_

![3D RSS Surface 1](./images/image-5.png)
_Image 4: 3D RSS Surface showing the cost landscape_

![Update formula representation](./images/image-4.png)
_Image 5: Update formula representation_

---

## 3. The Algorithm: Gradient Descent

Since the cost surface is often multi-dimensional (one dimension for each coefficient), we use **Gradient Descent** to systematically navigate to the minimum.

### Step 1: Calculate the Gradient ($\nabla_\beta L(\beta)$)

The **gradient** is a vector of partial derivatives that points in the direction of the **steepest ascent** of the loss function $L(\beta)$.

$$
\nabla_\beta L(\beta) =
\begin{bmatrix}
\frac{\partial L}{\partial\beta_0} \\
\frac{\partial L}{\partial\beta_1} \\
\vdots \\
\frac{\partial L}{\partial\beta_p}
\end{bmatrix}
$$

Since we want to **minimize** the cost, we move in the opposite direction: the **negative gradient** $(-\nabla_\beta L(\beta))$, which points to the steepest decrease.

### Step 2: The Update Rule

We start with initial parameter values ($\beta^{\text{(old)}}$) and iteratively adjust them using the update rule (see **Image 5**):

$$\beta^{\text{(new)}} = \beta^{\text{(old)}} - \eta \cdot \nabla_\beta L(\beta)$$

|           Term            |         Name           | Role and Significance                                         |
| :-----------------------: | :--------------------: | :------------------------------------------------------------ |
|  $\beta^{\text{(new)}}$   |  **New Parameters**    | The coefficients after the update step.                       |
|          $\eta$           |   **Learning Rate**    | A hyperparameter controlling the **step size**.               |
| $-\nabla_\beta L(\beta)$  | **Negative Gradient**  | The direction of **steepest descent** (the fastest way down). |

---

### Step 3: Iteration and Convergence

We repeat this process until we meet a **convergence criterion**, such as:

1.  The gradient magnitude approaches zero (we've hit the bottom).
2.  The loss change between steps is negligible.
3.  A maximum number of iterations (`steps`) has been reached.

---

## 4. Code Implementation

The provided Python code implements this Gradient Descent process using `sympy` for symbolic differentiation and `numpy` for numerical evaluation and vector operations.

### `RSS_gradient(X, y)`

This function symbolically computes the **gradient** of the RSS cost function. It uses `sympy` to create the symbolic derivatives $\frac{\partial \text{RSS}}{\partial \beta_i}$ based on the input data $X$ and $y$.

### `compute_gradient(grad, b)`

This function takes the list of **symbolic gradient expressions** (`grad`) and a set of **current numerical coefficients** (`b`), substitutes the numerical values into the expressions, and returns the resulting **numerical gradient vector**.

### `gradient_descent(X, y, steps=1000)`

This is the main driver function. It initializes coefficients, sets the learning rate, and runs the iterative loop:

1.  **Calculates** the numerical gradient (`compute_gradient`).
2.  **Applies** the update rule: `b_values = b_values - computed_gradient * learning_rate`.
3.  **Repeats** until convergence or the step limit is hit, returning the final optimized coefficients.
