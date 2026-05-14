import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Step 1: Load dataset
data = pd.read_csv("data.csv")

# Step 2: Display dataset
print("Face Biometric Recognition Dataset")
print(data)

# Step 3: Add recognition status
data["RecognitionStatus"] = data["ActualIdentity"] == data["PredictedIdentity"]

# Step 4: Calculate accuracy
accuracy = accuracy_score(data["ActualIdentity"], data["PredictedIdentity"])

print("\nRecognition Accuracy:")
print(round(accuracy * 100, 2), "%")

# Step 5: Display recognition status
print("\nRecognition Result:")
print(data[["SampleID", "ActualIdentity", "PredictedIdentity", "SimilarityScore", "RecognitionStatus"]])

# Step 6: Confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(data["ActualIdentity"], data["PredictedIdentity"]))

# Step 7: Classification report
print("\nClassification Report:")
print(classification_report(data["ActualIdentity"], data["PredictedIdentity"]))

# Step 8: Average similarity score
average_score = data["SimilarityScore"].mean()
print("\nAverage Similarity Score:")
print(round(average_score, 2))

# Step 9: Interpretation
if accuracy >= 0.90:
    interpretation = "The face biometric recognition system has high accuracy."
elif accuracy >= 0.70:
    interpretation = "The face biometric recognition system has moderate accuracy."
else:
    interpretation = "The face biometric recognition system has low accuracy."

print("\nInterpretation:")
print(interpretation)

# Step 10: Save updated result
data.to_csv("recognition_result.csv", index=False)

# Step 11: Plot similarity score
plt.bar(data["SampleID"], data["SimilarityScore"])
plt.title("Face Recognition Similarity Score")
plt.xlabel("Sample ID")
plt.ylabel("Similarity Score")
plt.ylim(0, 1)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("similarity_score_chart.png")
plt.show()
