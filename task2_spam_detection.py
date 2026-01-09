import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ----------------------------
# Load dataset (Point 1)
# ----------------------------
df = pd.read_csv("g_surguladze25_82519.csv")

# Features and label (Point 2)
X = df[["words", "links", "capital_words", "spam_word_count"]]
y = df["is_spam"]

# Train-test split 70/30 (Point 2 requirement)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

# Logistic Regression model (Point 2)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation (Point 3)
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

print("Accuracy:", accuracy)
print("\nConfusion Matrix (rows=Actual, cols=Predicted; 0=Legit, 1=Spam):\n", cm)
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))


# ==========================================================
# Point 4–7: Console application + examples + visualizations
# ==========================================================

# (Point 4) Email text -> features extraction
SPAM_WORDS = {
    "free", "winner", "win", "prize", "urgent", "limited", "offer", "click",
    "discount", "bonus", "money", "cash", "claim", "congratulations", "exclusive",
    "deal", "promo", "guarantee", "now"
}

LINK_REGEX = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)
WORD_REGEX = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")

def extract_features(email_text: str):
    tokens = WORD_REGEX.findall(email_text)
    words = len(tokens)
    links = len(LINK_REGEX.findall(email_text))
    capital_words = sum(1 for w in tokens if len(w) >= 2 and w.isupper())
    spam_word_count = sum(1 for w in (t.lower() for t in tokens) if w in SPAM_WORDS)
    return words, links, capital_words, spam_word_count

def predict_email_text(email_text: str):
    words, links, capital_words, spam_word_count = extract_features(email_text)
    sample_df = pd.DataFrame(
        [[words, links, capital_words, spam_word_count]],
        columns=["words", "links", "capital_words", "spam_word_count"]
    )
    pred = int(model.predict(sample_df)[0])
    proba = float(model.predict_proba(sample_df)[0][1])

    label = "SPAM" if pred == 1 else "LEGITIMATE"
    return label, proba, {"words": words, "links": links, "capital_words": capital_words, "spam_word_count": spam_word_count}


# (Point 5) Manually written SPAM email example
spam_email = """
CONGRATULATIONS!!! You are a WINNER of a FREE CASH PRIZE!
Click now to claim your exclusive bonus: https://fast-cash-prize.example/win
Limited OFFER, URGENT action required. Get DISCOUNT now!
"""

spam_label, spam_prob, spam_feats = predict_email_text(spam_email)
print("\n--- Manual SPAM Email Test ---")
print("Extracted features:", spam_feats)
print("Prediction:", spam_label, "| spam probability:", round(spam_prob, 4))

# (Point 6) Manually written LEGITIMATE email example
legit_email = """
Hello,
Please find attached the meeting notes and next steps from today.
Let me know if you have any questions.
Best regards,
Goga
"""

legit_label, legit_prob, legit_feats = predict_email_text(legit_email)
print("\n--- Manual LEGITIMATE Email Test ---")
print("Extracted features:", legit_feats)
print("Prediction:", legit_label, "| spam probability:", round(legit_prob, 4))


# (Point 4 continuation) Optional: interactive console mode
print("\n--- Console Mode (type your email text, or press ENTER to skip) ---")
user_text = input("Paste email text here: ").strip()
if user_text:
    user_label, user_prob, user_feats = predict_email_text(user_text)
    print("Extracted features:", user_feats)
    print("Prediction:", user_label, "| spam probability:", round(user_prob, 4))
else:
    print("Skipped console input.")


# (Point 7) Visualizations (save to plots/)
PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

# Visualization 1: class distribution
plt.figure()
counts = y.value_counts().sort_index()
plt.bar(["Legitimate (0)", "Spam (1)"], [counts.get(0, 0), counts.get(1, 0)])
plt.title("Class Distribution: Legitimate vs Spam")
plt.xlabel("Class")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "class_distribution.png"), dpi=200)
plt.close()

# Visualization 2: confusion matrix heatmap
plt.figure()
plt.imshow(cm, interpolation="nearest")
plt.title("Confusion Matrix Heatmap")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.xticks([0, 1], ["Legit (0)", "Spam (1)"])
plt.yticks([0, 1], ["Legit (0)", "Spam (1)"])
for i in range(2):
    for j in range(2):
        plt.text(j, i, str(cm[i, j]), ha="center", va="center")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "confusion_matrix_heatmap.png"), dpi=200)
plt.close()

print("\nSaved plots to ./plots/: class_distribution.png, confusion_matrix_heatmap.png")

