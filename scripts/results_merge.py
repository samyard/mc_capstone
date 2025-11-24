import pandas as pd
import numpy as np

# model results
gpt4o_mini = pd.read_csv('scripts/gpt4omini_results.csv')
gpt4o = pd.read_csv('scripts/gpt4o_results.csv')
gpt41 = pd.read_csv('scripts/gpt4.1_results.csv')

# col rename for merging
gpt4o_mini_cols = gpt4o_mini[['prompt_id', 'completion', 'completion_toxicity']].rename(
    columns={'completion': 'completion_gpt4o_mini', 'completion_toxicity': 'toxicity_gpt4o_mini'}
)

gpt4o_cols = gpt4o[['prompt_id', 'completion', 'completion_toxicity']].rename(
    columns={'completion': 'completion_gpt4o', 'completion_toxicity': 'toxicity_gpt4o'}
)

gpt41_cols = gpt41[['prompt_id', 'completion', 'completion_toxicity']].rename(
    columns={'completion': 'completion_gpt41', 'completion_toxicity': 'toxicity_gpt41'}
)

# base dataframe with prompt info
base_df = gpt4o_mini[['prompt_id', 'prompt_text', 'prompt_toxicity', 'toxicity_bin']].copy()

# merge
merged_df = base_df.merge(gpt4o_mini_cols, on='prompt_id', how='left')
merged_df = merged_df.merge(gpt4o_cols, on='prompt_id', how='left')
merged_df = merged_df.merge(gpt41_cols, on='prompt_id', how='left')

print(f"\nMerged dataset: {len(merged_df)} unique prompts")

# missing vals
print("\nMissing values per column:")
print(merged_df.isnull().sum())

# handle missing
incomplete_rows = merged_df[merged_df.isnull().any(axis=1)]
clean_df = merged_df.dropna().copy()

# save and preview
merged_df.to_csv('analysis_dataset_all.csv', index=False)
clean_df.to_csv('analysis_dataset_clean.csv', index=False)

print(clean_df.head())