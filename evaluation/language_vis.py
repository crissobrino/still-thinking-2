import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_benchmark(csv_file):
    # 1. Load data
    df = pd.read_csv(csv_file)
    df = df[df['status'] == 'SUCCESS']  # only successful rows

    # 2. Global metrics
    print("\n=== GLOBAL METRICS BY LANGUAGE ===")
    global_stats = df.groupby('lang').agg({
        'total_ms': 'mean',
        'llm_ms': 'mean',
        'recall_vs_en': 'mean',
        'context_tokens': 'mean'
    }).round(2)
    print(global_stats)

    # 3. Chart setup
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Chart 1: Time breakdown (ANN vs rerank vs LLM)
    time_cols = ['ann_ms', 'rerank_ms', 'guardrail_ms', 'llm_ms']
    df_melted = df.melt(id_vars='lang', value_vars=time_cols, var_name='Stage', value_name='ms')
    sns.barplot(data=df_melted, x='lang', y='ms', hue='Stage', ax=axes[0,0])
    axes[0,0].set_title("Time Breakdown by Language")

    # Chart 2: Recall by language (multilingual consistency)
    sns.line_marker = True
    sns.pointplot(data=df, x='lang', y='recall_vs_en', ax=axes[0,1], color='green')
    axes[0,1].set_title("Recall vs English (Search Consistency)")
    axes[0,1].set_ylim(0, 1.1)

    # Chart 3: Context tokens vs LLM time correlation
    sns.regplot(data=df, x='context_tokens', y='llm_ms', ax=axes[1,0], scatter_kws={'alpha':0.5})
    axes[1,0].set_title("Context Size (Tokens) Impact on LLM Time")

    # Chart 4: LLM response (output tokens vs time)
    sns.scatterplot(data=df, x='answer_tokens', y='llm_ms', hue='lang', ax=axes[1,1])
    axes[1,1].set_title("Response Length vs LLM Time")

    plt.tight_layout()
    plt.show()

    # 4. Correlation analysis
    correlation = df['context_tokens'].corr(df['llm_ms'])
    print(f"\n💡 Correlation coefficient (context vs time): {correlation:.2f}")
    if correlation > 0.7:
        print("⚠️ Context size has a strong influence on latency.")
    else:
        print("✅ LLM time appears mostly independent of context size (likely dominated by network latency or model load).")

if __name__ == "__main__":
    analyze_benchmark("full_diagnostic_benchmark3.csv")