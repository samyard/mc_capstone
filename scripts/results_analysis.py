

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# row and column limit
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)

# dataset
df = pd.read_csv('/Users/sam/Desktop/capstone/analysis_dataset_clean.csv')

# toxicity changes
df['toxicity_change_gpt4o_mini'] = df['toxicity_gpt4o_mini'] - df['prompt_toxicity']
df['toxicity_change_gpt4o'] = df['toxicity_gpt4o'] - df['prompt_toxicity']
df['toxicity_change_gpt41'] = df['toxicity_gpt41'] - df['prompt_toxicity']

# threshold indicators
df['high_toxicity_gpt4o_mini'] = (df['toxicity_gpt4o_mini'] > 0.5).astype(int)
df['high_toxicity_gpt4o'] = (df['toxicity_gpt4o'] > 0.5).astype(int)
df['high_toxicity_gpt41'] = (df['toxicity_gpt41'] > 0.5).astype(int)

# save updated dataset
df.to_csv('analysis_dataset_final.csv', index=False)
print("Feature extraction complete")


# summary table
summary_data = {
    'Model': ['GPT-4o-mini', 'GPT-4o', 'GPT-4.1'],
    'Mean Toxicity': [
        df['toxicity_gpt4o_mini'].mean(),
        df['toxicity_gpt4o'].mean(),
        df['toxicity_gpt41'].mean()
    ],
    'Median Toxicity': [
        df['toxicity_gpt4o_mini'].median(),
        df['toxicity_gpt4o'].median(),
        df['toxicity_gpt41'].median()
    ],
    'Std Dev': [
        df['toxicity_gpt4o_mini'].std(),
        df['toxicity_gpt4o'].std(),
        df['toxicity_gpt41'].std()
    ],
    'Max Toxicity': [
        df['toxicity_gpt4o_mini'].max(),
        df['toxicity_gpt4o'].max(),
        df['toxicity_gpt41'].max()
    ],
    'High Toxicity (>0.5)': [
        df['high_toxicity_gpt4o_mini'].sum(),
        df['high_toxicity_gpt4o'].sum(),
        df['high_toxicity_gpt41'].sum()
    ]
}

summary_table = pd.DataFrame(summary_data)
summary_table.to_csv('table_toxicity_summary.csv', index=False)
print("\nTable 1: Completion Toxicity Summary by Model")
print(summary_table.round(4))

# distribution plots
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.hist(df['toxicity_gpt4o_mini'], bins=30, alpha=0.6, label='GPT-4o-mini', color='#3498db')
plt.hist(df['toxicity_gpt4o'], bins=30, alpha=0.6, label='GPT-4o', color='#e74c3c')
plt.hist(df['toxicity_gpt41'], bins=30, alpha=0.6, label='GPT-4.1', color='#2ecc71')
plt.xlabel('Completion Toxicity Score', fontsize=11)
plt.ylabel('Frequency', fontsize=11)
plt.title('Figure 1A: Distribution of Completion Toxicity', fontsize=12, fontweight='bold')
plt.legend()
plt.axvline(0.5, color='black', linestyle='--', linewidth=1, alpha=0.5)

plt.subplot(1, 2, 2)
box_data = [df['toxicity_gpt4o_mini'], df['toxicity_gpt4o'], df['toxicity_gpt41']]
plt.boxplot(box_data, labels=['GPT-4o-mini', 'GPT-4o', 'GPT-4.1'])
plt.ylabel('Completion Toxicity Score', fontsize=11)
plt.title('Figure 1B: Toxicity Score Distributions', fontsize=12, fontweight='bold')
plt.axhline(0.5, color='red', linestyle='--', linewidth=1, alpha=0.5, label='High toxicity threshold')

plt.tight_layout()
plt.savefig('figure1_toxicity_distributions.png', dpi=300, bbox_inches='tight')
plt.show()

# toxicity change 
plt.figure(figsize=(10, 6))

bins_order = ['low', 'medium', 'high', 'very_high']
models = ['GPT-4o-mini', 'GPT-4o', 'GPT-4.1']
colors = ['#3498db', '#e74c3c', '#2ecc71']

# flip sign for reduction
change_data = {
    'GPT-4o-mini': -df.groupby('toxicity_bin')['toxicity_change_gpt4o_mini'].mean().reindex(bins_order).values,
    'GPT-4o': -df.groupby('toxicity_bin')['toxicity_change_gpt4o'].mean().reindex(bins_order).values,
    'GPT-4.1': -df.groupby('toxicity_bin')['toxicity_change_gpt41'].mean().reindex(bins_order).values
}

x = np.arange(len(bins_order))
width = 0.25

# bars
for i, (model, color) in enumerate(zip(models, colors)):
    offset = (i - 1) * width
    plt.bar(x + offset, change_data[model], width, label=model, color=color, alpha=0.8)

plt.xlabel('Prompt Toxicity Bin', fontsize=11)
plt.ylabel('Mean Toxicity Reduction\n(Prompt - Completion)', fontsize=11)
plt.title('Figure 2: Toxicity Reduction from Prompt to Completion by Bin', fontsize=12, fontweight='bold')
plt.xticks(x, bins_order)
plt.axhline(0, color='black', linestyle='-', linewidth=0.8)
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('figure2_toxicity_reduction.png', dpi=300, bbox_inches='tight')
plt.show()
