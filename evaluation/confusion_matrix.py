import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Illustrative example data (not real guardrail output) to demonstrate the metric.
# True labels: 1 = scientific (should pass), 0 = junk (should be blocked)
y_true = [1, 1, 0, 0, 0, 1]
# Predicted labels (1 = not blocked, 0 = blocked)
y_pred = [1, 1, 0, 1, 0, 0]  # example failures: one junk query passed, one scientific query got blocked

cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Blocked", "Passed"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix: Safety Guardrail")
plt.show()