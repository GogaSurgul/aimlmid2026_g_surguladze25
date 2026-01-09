import numpy as np
import matplotlib.pyplot as plt

# Data collected from the blue points on the graph
x = np.array([
    -8.10,
    -5.70,
    -3.30,
     1.40,
     3.90,
     6.40,
     8.90
])

y = np.array([
    -6.50,
    -4.00,
    -1.70,
     2.20,
     4.40,
     6.70,
     8.30
])

# Calculate Pearson correlation coefficient
r = np.corrcoef(x, y)[0, 1]
print("Pearson correlation coefficient:", r)

# Scatter plot
plt.scatter(x, y)
plt.xlabel("X values")
plt.ylabel("Y values")
plt.title("Correlation Analysis")
plt.grid(True)
plt.show()
