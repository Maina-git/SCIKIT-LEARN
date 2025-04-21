
# Step 1: Install all the required Python libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Step 2: Create the dataset
data = {
    "SquareFeet": [600, 800, 1000, 1200, 1400, 1600, 1800, 2000],
    "Price": [150000, 180000, 210000, 240000, 270000, 300000, 330000, 360000]
}
# converts that Python dictionary into a pandas table
df = pd.DataFrame(data)

# Step 3: Preview and debug
print("Column names:", df.columns.tolist())
print("Data preview:\n", df.head())

# Step 4: Split features (X) and target (y)
x = df[["SquareFeet"]]  # Capital F!
y = df["Price"]         # Single bracket is fine

# Step 5: Train-Test Split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Step 6: Train the model
model = LinearRegression()
model.fit(x_train, y_train)

# Step 7: Make predictions
predictions = model.predict(x_test)

# Step 8: Visualize the regression
plt.scatter(x, y, color="blue", label="Actual")
plt.plot(x, model.predict(x), color="red", label="Regression Line")
plt.xlabel("Square Feet")
plt.ylabel("Price")
plt.title("Linear Regression - House Prices")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Step 9: Show model details
print("\nModel Coefficient (slope):", model.coef_)
print("Model Intercept (bias):", model.intercept_)


