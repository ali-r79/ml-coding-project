## Final Project – ML-CC

This repository contains the **final joint project** for the courses **Coding in Chemistry** and **Machine Learning (ML)**.

### Project Objective
The goal of this project is to demonstrate the capability of **Neural Networks (NNs)** to solve physical problems **without explicit simulations or analytical assumptions**, relying solely on **input–output data**.

### Problem Description
- The dataset consists of **1000 samples**.
- The outer radius of the circular domain (`re`) is **fixed and equal to 10**.
- A **circular hole** exists inside the domain:
  - Its radius (`ri`) is **variable**.
  - The hole moves inside the main circle.
  - The hole position is defined by (`x_center`, `y_center`).
- The **outputs** are the first **four eigenvalues (energy levels)** corresponding to:
  
  \[
  n = [1, 2, 3, 4]
  \]

### Model Usage
A trained neural network model is stored as a serialized file (`best_model.sav`) and can be loaded using `pickle`.

#### Steps to Run the Model
1. Clone the repository.
2. Load the trained model.
3. Prepare the input DataFrame with the correct column order.
4. Run prediction.

```python
import pickle

# Load the trained model
loaded_model = pickle.load(open('./best_model.sav', 'rb'))

# IMPORTANT:
# The input DataFrame X must have the following column order:
# ["x_center", "y_center", "ri"]

Y_pred = loaded_model.predict(X)
