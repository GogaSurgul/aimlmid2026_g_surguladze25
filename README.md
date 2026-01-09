Task 1 – Correlation Analysis

I collected the (x, y) coordinates of the blue points by hovering over the points on the provided online graph.

Then I calculated Pearson’s correlation coefficient using NumPy (`np.corrcoef`) and visualized the data with a scatter plot using Matplotlib.

Result: r ≈ 0.999, which indicates a very strong positive linear correlation between X and Y (as X increases, Y also increases).



<img width="1380" height="794" alt="Screenshot 2026-01-09 at 20 46 23" src="https://github.com/user-attachments/assets/e4d1b6d6-8ec2-4719-ae09-f63aa4d34b0c" />






Task 2 – Spam Detection using Logistic Regression

The provided CSV dataset contains numerical features related to email content, such as number of words, links, capital words, and spam-related words.

A Logistic Regression classifier was trained to detect spam emails. The dataset was split into training and testing subsets.

Model performance was evaluated using accuracy and a classification report. The results demonstrate that Logistic Regression is effective for spam detection on this dataset.


<img width="1364" height="794" alt="image" src="https://github.com/user-attachments/assets/41020594-34ff-4ae5-b6dd-0b58697a450c" />

