import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_benchmark(csv_file):
    # 1. Cargar datos
    df = pd.read_csv(csv_file)
    df = df[df['status'] == 'SUCCESS'] # Solo filas exitosas
    
    # 2. Métricas Globales
    print("\n=== MÉTRICAS GLOBALES POR IDIOMA ===")
    global_stats = df.groupby('lang').agg({
        'total_ms': 'mean',
        'llm_ms': 'mean',
        'recall_vs_en': 'mean',
        'context_tokens': 'mean'
    }).round(2)
    print(global_stats)

    # 3. Configuración de Gráficos
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Gráfico 1: Desglose de Tiempos (ANN vs Rerank vs LLM)
    time_cols = ['ann_ms', 'rerank_ms', 'guardrail_ms', 'llm_ms']
    df_melted = df.melt(id_vars='lang', value_vars=time_cols, var_name='Etapa', value_name='ms')
    sns.barplot(data=df_melted, x='lang', y='ms', hue='Etapa', ax=axes[0,0])
    axes[0,0].set_title("Desglose de Tiempos por Idioma")

    # Gráfico 2: Recall por Idioma (Consistencia Multilingüe)
    sns.line_marker = True
    sns.pointplot(data=df, x='lang', y='recall_vs_en', ax=axes[0,1], color='green')
    axes[0,1].set_title("Recall vs Inglés (Consistencia de búsqueda)")
    axes[0,1].set_ylim(0, 1.1)

    # Gráfico 3: Correlación Tokens de Contexto vs Tiempo LLM
    sns.regplot(data=df, x='context_tokens', y='llm_ms', ax=axes[1,0], scatter_kws={'alpha':0.5})
    axes[1,0].set_title("Influencia del Contexto (Tokens) en el Tiempo del LLM")

    # Gráfico 4: Respuesta del LLM (Tokens de Salida vs Tiempo)
    sns.scatterplot(data=df, x='answer_tokens', y='llm_ms', hue='lang', ax=axes[1,1])
    axes[1,1].set_title("Longitud de Respuesta vs Tiempo LLM")

    plt.tight_layout()
    plt.show()

    # 4. Análisis de Correlación
    correlation = df['context_tokens'].corr(df['llm_ms'])
    print(f"\n💡 Coeficiente de correlación (Contexto vs Tiempo): {correlation:.2f}")
    if correlation > 0.7:
        print("⚠️ El tamaño del contexto influye fuertemente en la latencia.")
    else:
        print("✅ El tiempo del LLM parece ser independiente del tamaño del contexto (quizás domina la latencia de red o carga del modelo).")

if __name__ == "__main__":
    analyze_benchmark("full_diagnostic_benchmark3.csv")