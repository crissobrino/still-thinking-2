import matplotlib
matplotlib.use('Agg')
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats

base = 'evaluation/llm_as_a_judge/results'

small_faith  = pd.read_csv(f'{base}/small/automated_results_openai_gpt-4o-mini_faithfulness.csv')
small_ar     = pd.read_csv(f'{base}/small/automated_results_manual_openai_gpt-4o-mini_answer_relevancy.csv')
large_faith  = pd.read_csv(f'{base}/large/automated_results_openai_gpt-4o-mini_faithfulness.csv')
large_ar     = pd.read_csv(f'{base}/large/automated_results_manual_openai_gpt-4o-mini_answer_relevancy.csv')

rows = []
for model, fd, ad in [
    ('Small (qwen3:8b)',  small_faith, small_ar),
    ('Large (qwen3:32b)', large_faith, large_ar),
]:
    for v in fd['faithfulness'].dropna():
        rows.append({'Configuration': model, 'Metric': 'Faithfulness',     'Score': v})
    for v in ad['answer_relevancy'].dropna():
        rows.append({'Configuration': model, 'Metric': 'Answer Relevancy', 'Score': v})

df = pd.DataFrame(rows)

def ci95_upper(vals):
    n = len(vals)
    return np.mean(vals) + stats.t.ppf(0.975, df=n-1) * stats.sem(vals)

ci_upper = {}
for (conf, metric), grp in df.groupby(['Configuration', 'Metric']):
    ci_upper[(conf, metric)] = ci95_upper(grp['Score'].values)

palette = {'Small (qwen3:8b)': '#A8C4E0', 'Large (qwen3:32b)': '#1B3B6F'}

sns.set_theme(style='whitegrid', font_scale=1.1)
fig, ax = plt.subplots(figsize=(6, 6))

sns.barplot(
    data=df, x='Metric', y='Score', hue='Configuration',
    hue_order=['Small (qwen3:8b)', 'Large (qwen3:32b)'],
    palette=palette,
    capsize=0.08,
    errorbar=('ci', 95),
    ax=ax,
    width=0.7,
)

ax.set_ylim(0, 1.28)
ax.set_xlabel('')
ax.set_ylabel('Score (0-1)', fontsize=12)
ax.set_title('LLM Generation Quality', fontsize=14, fontweight='bold')

metric_order = ['Faithfulness', 'Answer Relevancy']
configs      = ['Small (qwen3:8b)', 'Large (qwen3:32b)']
for container, conf in zip(ax.containers, configs):
    for patch, metric in zip(container.patches, metric_order):
        mean_val = df[(df['Configuration']==conf) & (df['Metric']==metric)]['Score'].mean()
        upper    = ci_upper[(conf, metric)]
        ax.annotate(
            f'{mean_val:.2f}',
            xy=(patch.get_x() + patch.get_width() / 2, upper + 0.02),
            ha='center', va='bottom', fontsize=10,
        )

ax.legend(title='Configuration', loc='upper right',
          ncol=1, framealpha=0.95, edgecolor='#cccccc',
          fontsize=10, title_fontsize=10)
ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.yaxis.grid(True, linewidth=0.7, color='#e0e0e0')
ax.set_axisbelow(True)

plt.tight_layout()
out = f'{base}/llm_eval_chart_2metrics.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f"Saved to {out}")
