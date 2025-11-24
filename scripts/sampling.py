
import pandas as pd
import numpy as np
import json

# row and column limit
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)

# data path
data_path = '/Users/sam/Desktop/capstone/data/prompts.jsonl'

def load_prompts(filepath):
    """load prompts"""
    prompts = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            prompts.append(data)
    return pd.DataFrame(prompts)

# data load
df = load_prompts(data_path)

# prompt information
def extract_prompt_info(df):
    """ relevant fields from nested JSON"""
    extracted_data = []
    
    for idx, row in df.iterrows():
        prompt_data = row['prompt']
        extracted_data.append({
            'prompt_id': idx,
            'prompt_text': prompt_data['text'],
            'prompt_toxicity': prompt_data.get('toxicity', None),
        })
    
    return pd.DataFrame(extracted_data)

df_clean = extract_prompt_info(df)

# check unicode decoding
print(df_clean['prompt_text'].iloc[0])

# remove prompts without toxicity scores
df_clean = df_clean.dropna(subset=['prompt_toxicity'])

print(f"\nCleaned dataset: {len(df_clean)} prompts with toxicity scores")
print(f"Toxicity range: {df_clean['prompt_toxicity'].min():.3f} to {df_clean['prompt_toxicity'].max():.3f}")

def stratified_sample_prompts(df, n_samples=350, n_bins=4, random_state=1):
    """
    Create stratified sample across toxicity levels
    random_state = n for reproducibility
    """
    
    # toxicity bins
    df['toxicity_bin'] = pd.cut(
        df['prompt_toxicity'], 
        bins=n_bins,
        labels=['low', 'medium', 'high', 'very_high']
    )
    
    # samples per bin
    samples_per_bin = n_samples // n_bins
    
    # sample from each bin
    sampled_dfs = []
    for bin_name in ['low', 'medium', 'high', 'very_high']:
        bin_df = df[df['toxicity_bin'] == bin_name]
        
        # if a bin has fewer samples than needed, take all available
        n_to_sample = min(samples_per_bin, len(bin_df))
        
        sampled = bin_df.sample(n=n_to_sample, random_state=random_state)
        sampled_dfs.append(sampled)
        print(f"{bin_name}: sampled {n_to_sample} from {len(bin_df)} available")
    
    # Combine all samples
    final_sample = pd.concat(sampled_dfs, ignore_index=True)
    
    print(f"\nFinal sample size: {len(final_sample)}")
    print("\nSample distribution:")
    print(final_sample['toxicity_bin'].value_counts().sort_index())
    
    return final_sample

# sample and save
sample_df = stratified_sample_prompts(df_clean, n_samples=350)

sample_df.to_csv('sampled_prompts.csv', index=False, encoding='utf-8')
print("\nSample saved to 'sampled_prompts.csv'")

# basic statistics
print("\nSample Statistics:")
print(sample_df['prompt_toxicity'].describe())

# examples from each bin
print("\n-Example Prompts\n")
for bin_name in ['low', 'medium', 'high', 'very_high']:
    print(f"\n{bin_name.upper()} toxicity example:")
    example = sample_df[sample_df['toxicity_bin'] == bin_name].iloc[0]
    print(f"Toxicity: {example['prompt_toxicity']:.3f}")
    print(f"Text: {example['prompt_text'][:100]}...")


