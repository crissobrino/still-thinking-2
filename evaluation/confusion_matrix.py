from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Etiquetas reales: 1 = Científico (debe pasar), 0 = Basura (debe bloquearse)
y_true = [1, 1, 0, 0, 0, 1] 
# Etiquetas predichas (si NO bloqueó = 1, si bloqueó = 0)
y_pred = [1, 1, 0, 1, 0, 0] # Ejemplo de fallos (un 1 que pasó y un científico que se bloqueó)

cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Blocked", "Passed"])
disp.plot(cmap="Blues")
plt.title("Matriz de Confusión: Guardrail de Seguridad")
plt.show()