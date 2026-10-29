import re
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report

# 1. Sample Dataset
data = {
	"email_text": [
		"Your bank account is blocked. Click here to verify your password.",
		"Congratulations! You won a prize. Claim it immediately.",
		"Urgent! Verify your account at http://fake-login.xyz",
		"Your account will be suspended. Update your details now.",
		"You have won a lottery. Send your bank details.",
		"Click this suspicious link to claim your free gift.",
		"Reset your password immediately using this link.",
		"Your account has a security problem. Verify now.",
		"Meeting scheduled for tomorrow at 10 AM.",
		"Hello, please find the attached project report.",
		"Your order has been shipped. Thank you for shopping.",
		"The class assignment is due next Monday.",
		"Thank you for attending today's meeting.",
		"Your appointment is confirmed for tomorrow.",
		"Please review the attached invoice.",
		"The college examination timetable is available.",
		"Your monthly bank statement is ready.",
		"Here are the notes from our last meeting.",
		"Your course registration is successful.",
		"Please submit your project before the deadline."
	],
	"label": [
		"Phishing", "Phishing", "Phishing", "Phishing",
		"Phishing", "Phishing", "Phishing", "Phishing",
		"Safe", "Safe", "Safe", "Safe", "Safe",
		"Safe", "Safe", "Safe", "Safe", "Safe",
		"Safe", "Safe"
	]
}

df = pd.DataFrame(data)

# 2. Extract Email and URL Features
def extract_features(email):
	email = str(email).lower()

	urls = re.findall(r'https?://\S+|www\.\S+', email)

	suspicious_words = [
		"urgent", "verify", "password", "blocked",
		"suspended", "prize", "claim", "lottery"
	]

	return {
		"url_count": len(urls),
		"suspicious_words": sum(
			email.count(word) for word in suspicious_words
		),
		"has_ip_url": int(
			bool(re.search(r'https?://\d{1,3}(?:\.\d{1,3}){3}', email))
		),
		"email_length": len(email)
	}

features = df["email_text"].apply(extract_features).apply(pd.Series)

df = pd.concat([df, features], axis=1)

# 3. Select Input Features and Labels
X = df[
	[
		"email_text",
		"url_count",
		"suspicious_words",
		"has_ip_url",
		"email_length"
	]
]

y = df["label"]

# 4. Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
	X,
	y,
	test_size=0.25,
	random_state=42,
	stratify=y
)

# 5. Convert Text into Numerical Features
preprocessor = ColumnTransformer(
	transformers=[
		(
			"text",
			TfidfVectorizer(),
			"email_text"
		),
		(
			"numeric",
			StandardScaler(),
			[
				"url_count",
				"suspicious_words",
				"has_ip_url",
				"email_length"
			]
		)
	]
)

# 6. Build the Machine Learning Model
model = Pipeline([
	("features", preprocessor),
	("classifier", LogisticRegression(max_iter=1000))
])

# 7. Train the Model
model.fit(X_train, y_train)

# 8. Test the Model
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\n===== PHISHING EMAIL DETECTION =====")
print(f"Model Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# 9. Display Confusion Matrix
cm = confusion_matrix(
	y_test,
	y_pred,
	labels=["Phishing", "Safe"]
)

display = ConfusionMatrixDisplay(
	confusion_matrix=cm,
	display_labels=["Phishing", "Safe"]
)

display.plot()
plt.title("Phishing Email Detection - Confusion Matrix")
plt.tight_layout()
plt.show()

# 10. Predict a New Email
def predict_email(email):
	new_features = extract_features(email)

	new_data = pd.DataFrame([{
		"email_text": email,
		**new_features
	}])

	prediction = model.predict(new_data)[0]

	print("\n===== EMAIL ANALYSIS =====")
	print("Email:", email)
	print("Prediction:", prediction)

	if prediction == "Phishing":
		print("Warning: This email may be a phishing email!")
	else:
		print("This email is classified as safe.")

# Example
predict_email(
	"URGENT! Verify your password immediately at http://fake-login.xyz"
)

predict_email(
	"Hello, please find the meeting notes attached."
)
