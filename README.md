Task 1 – Correlation Analysis

I collected the (x, y) coordinates of the blue points by hovering over the points on the provided online graph.

Then I calculated Pearson’s correlation coefficient using NumPy (`np.corrcoef`) and visualized the data with a scatter plot using Matplotlib.

Result: r ≈ 0.999, which indicates a very strong positive linear correlation between X and Y (as X increases, Y also increases).



<img width="1380" height="794" alt="Screenshot 2026-01-09 at 20 46 23" src="https://github.com/user-attachments/assets/e4d1b6d6-8ec2-4719-ae09-f63aa4d34b0c" />





Task 2 – Spam Email Detection (Logistic Regression)

I used the provided dataset `g_surguladze25_82519.csv` with the following features:
- words, links, capital_words, spam_word_count
and the target label `is_spam` (0 = legitimate, 1 = spam).

I trained a Logistic Regression classifier using a 70/30 train-test split.
Model performance on the test set:
- Accuracy: 0.9533
- Confusion Matrix: [[366, 10], [25, 349]]
A full classification report is printed by the console application.

The program also supports classifying new email text:
1) It extracts the same numeric features from the email text,
2) Uses the trained model to predict SPAM or LEGITIMATE.
I tested two manually written examples (one spam and one legitimate) and the predictions were correct.

<img width="1415" height="831" alt="image" src="https://github.com/user-attachments/assets/c7e693fc-a24d-4254-8580-0cadf33ea0fb" />

<img width="1418" height="840" alt="image" src="https://github.com/user-attachments/assets/0e4ec482-d500-4701-96b1-59effa2bd3b0" />





