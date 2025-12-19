import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# seaborn style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# sample data
PROJECT_ROOT = Path(__file__).parent.parent
sample_df = pd.read_csv(PROJECT_ROOT / 'sampled_prompts.csv')

# table 1: descriptive statistics by bin

table1 = sample_df.groupby('toxicity_bin').agg({
    'prompt_toxicity': ['count', 'mean', 'std', 'min', 'max'],
    'prompt_text': lambda x: x.str.len().mean()
}).round(3)

table1.columns = ['N', 'Mean Toxicity', 'Std Dev', 'Min', 'Max', 'Avg Length (chars)']
print("\nTable 1: Descriptive Statistics by Toxicity Bin")
print(table1)
table1.to_csv('table1_descriptive_stats.csv')

# table 2: sample characteristics

table2 = pd.DataFrame({
    'Statistic': ['Total Prompts', 'Mean Toxicity', 'Median Toxicity', 
                  'Std Dev', 'Min Toxicity', 'Max Toxicity',
                  'Mean Length (chars)', 'Mean Length (words)'],
    'Value': [
        len(sample_df),
        sample_df['prompt_toxicity'].mean(),
        sample_df['prompt_toxicity'].median(),
        sample_df['prompt_toxicity'].std(),
        sample_df['prompt_toxicity'].min(),
        sample_df['prompt_toxicity'].max(),
        sample_df['prompt_text'].str.len().mean(),
        sample_df['prompt_text'].str.split().str.len().mean()
    ]
}).round(3)

print("\nTable 2: Overall Sample Characteristics")
print(table2)
table2.to_csv('table2_overall_stats.csv', index=False)

# figure 1: prompt toxicity distribution

plt.figure(figsize=(10, 6))
plt.hist(sample_df['prompt_toxicity'], bins=50, edgecolor='black', alpha=0.7)
plt.axvline(sample_df['prompt_toxicity'].mean(), color='red', 
            linestyle='--', linewidth=2, label=f'Mean = {sample_df["prompt_toxicity"].mean():.3f}')
plt.axvline(sample_df['prompt_toxicity'].median(), color='green', 
            linestyle='--', linewidth=2, label=f'Median = {sample_df["prompt_toxicity"].median():.3f}')
plt.xlabel('Toxicity Score', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title('Figure 1: Distribution of Prompt Toxicity Scores in Sample', fontsize=14)
plt.legend()
plt.tight_layout()
plt.savefig('figure1_toxicity_distribution.png', dpi=300)
plt.show()

# figure 2: toxicity by bin

plt.figure(figsize=(10, 6))
sns.boxplot(data=sample_df, x='toxicity_bin', y='prompt_toxicity', 
            order=['low', 'medium', 'high', 'very_high'])
plt.xlabel('Toxicity Bin', fontsize=12)
plt.ylabel('Toxicity Score', fontsize=12)
plt.title('Figure 2: Toxicity Score Distribution by Bin', fontsize=14)
plt.tight_layout()
plt.savefig('figure2_toxicity_by_bin.png', dpi=300)
plt.show()

# figure 3: sample distribution across bins

plt.figure(figsize=(10, 6))
bin_counts = sample_df['toxicity_bin'].value_counts().reindex(
    ['low', 'medium', 'high', 'very_high']
)
bin_counts.plot(kind='bar', edgecolor='black')
plt.xlabel('Toxicity Bin', fontsize=12)
plt.ylabel('Number of Prompts', fontsize=12)
plt.title('Figure 3: Sample Distribution Across Toxicity Bins', fontsize=14)
plt.xticks(rotation=0)
for i, v in enumerate(bin_counts):
    plt.text(i, v + 2, str(v), ha='center', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig('figure3_sample_distribution.png', dpi=300)
plt.show()

