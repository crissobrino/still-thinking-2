import os
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevance, context_precision, context_recall
from langchain_google_genai import ChatGoogleGenerativeAI

# 1. CONFIGURACIÓN DEL JUEZ (Gratis en Google AI Studio)
os.environ["GOOGLE_API_KEY"] = "TU_API_KEY_AQUI"
evaluator_llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

# 2. LOS DATOS (Lo que tu sistema generó)
# En un caso real, harías un bucle llamando a tu backend
data = {
    "question": [
        "¿Cómo se comparan las métricas UMass y UCI en LDA?",
        "¿Qué aporta mi idea de usar Neural Topic Models frente al PDF?"
    ],
    "answer": [
        "UMass usa probabilidad logarítmica y UCI usa PMI (Información Mutua Puntual).",
        "El PDF menciona que los modelos neurales son una propuesta actual para explotar modelos de lenguaje contextualizados."
    ],
    "contexts": [
        ["El PDF en la página 45 explica que UMass se basa en recuentos de co-ocurrencia y UCI en una ventana deslizante con PMI."],
        ["Página 4 menciona: Nowadays, Neural Topic Models have been proposed to exploit contextualized language models."]
    ],
    "ground_truth": [
        "UMass se basa en co-ocurrencia de documentos y UCI en PMI sobre una ventana externa.",
        "Los modelos neurales permiten usar modelos de lenguaje contextualizados para mejores temas."
    ]
}

dataset = Dataset.from_dict(data)

# 3. EJECUCIÓN DE LA EVALUACIÓN
print("Evaluando sistema con Ragas...")
result = evaluate(
    dataset=dataset,
    metrics=[
        faithfulness,       # ¿Miente el bot? (Rubric: No alucinaciones)
        answer_relevance,  # ¿Responde a lo que se le pide?
        context_precision  # ¿Es buena la búsqueda en ChromaDB?
    ],
    llm=evaluator_llm
)

# 4. GUARDAR RESULTADOS
df = result.to_pandas()
df.to_csv("evaluation/resultados_finales.csv")
print("¡Evaluación completada! Notas medias:", result)