# LLM Toxicity Reduction: GPT-4 Family Evaluation

A replication study of [RealToxicityPrompts (2020)](https://arxiv.org/abs/2009.11462) evaluating toxicity mitigation in GPT-4 family models.

## Overview

This capstone project investigates whether modern large language models have made meaningful progress in reducing toxic text generation compared to their predecessors from 2020. By replicating the methodology of the RealToxicityPrompts study, I compare toxicity generation across three GPT-4 family models (GPT-4o-mini, GPT-4o, and GPT-4.1) against historical GPT benchmarks.

### Key Findings

- **30+ percentage point reduction** in toxicity rates from GPT-3, 2020 to GPT-4 family, 2024
- **GPT-4o-mini** achieved the lowest toxicity scores despite being the smallest and most cost-efficient model
- Models demonstrate **adaptive safety mechanisms**: minimal intervention on benign prompts (0.05-0.1 reduction), strong mitigation on toxic inputs (0.5-0.8 reduction)
- **Vulnerabilities remain**: Maximum toxicity scores still reach 0.6-0.9, and adversarial attacks can still compromise safety (external research shows - https://arxiv.org/html/2311.05553v3#bib.bib9)

## Research Question

**Have GPT-4 family models made meaningful progress in reducing toxic text generation compared to legacy models?**

### Hypothesis
Architectural improvements (RLHF, data curation, content filtering) implemented since 2020 have reduced models' likelihood to generate toxic content.

## Methodology

### Dataset
- **Source**: [RealToxicityPrompts](https://allenai.org/data/real-toxicity-prompts) - 100K naturally occurring prompts from web text
- **Sample Size**: 347 prompts via stratified random sampling
- **Toxicity Bins**: Low (0.0-0.25), Medium (0.25-0.5), High (0.5-0.75), Very High (0.75-1.0)
- **Measurement**: Google's Perspective API (0.0-1.0 scale)

### Models Tested
1. **GPT-4o-mini** - Smallest, most efficient model
2. **GPT-4o** - Standard GPT-4 optimized model  
3. **GPT-4.1** - Latest GPT-4 iteration

### Data Collection Process
```
Prompt (n=347) → GPT-4 Model (via OpenAI API) → Completion → Perspective API → Toxicity Score
```

All generation used consistent parameters:
- Temperature: 1.0
- Max tokens: 20

### Toxicity Reduction Patterns
- **Benign prompts**: 0.05-0.1 point reduction (minimal intervention)
- **Toxic prompts**: 0.5-0.8 point reduction (strong mitigation)

This adaptive behavior suggests risk based filtering rather than blanket filtering

## Repository Structure
```
├── data/
│   ├── README.md                       # RealToxicityPrompts markdown
│   └── ...                             # RealToxicityPrompts data
├── scripts/
│   ├── data_collection.py              # data collection pipeline
│   ├── prompt_stats.py                 # prompt descriptive statistics
│   ├── results_analysis.py             # final data analysis
│   ├── results_merge.py                # merging of all GPT response results
│   └── sampling.py                     # stratified sampling prompts
|
├── requirements.txt			# all dependencies for study replication
├── .gitignore				# files / file types for git to ignore
├── CapstonePresentation_Yard.pptx	# final presentation ppt
├── FinalReport_Yard.pdf		# final report pdf
└── README.md				# documentation
```

## Reproducibility

To use the data collection pipeline on your local machine:

**Prerequisites:**
- Python 3.8 or higher
- OpenAI API key (for GPT model access)
- Google Perspective API key (for toxicity scoring)

**Setup:**
1. Clone the repository: `git clone https://github.com/samyard/mc_capstone`
2. Install required packages: `pip install -r requirements.txt`
3. Create a `.env` file in the project root with your API keys and output dir:
```
   OPENAI_API_KEY=your_openai_key_here
   PERSPECTIVE_API_KEY=your_perspective_key_here
   # Optional: Specify a custom results directory (defaults to `results/` in project root):
   RESULTS_DIR=/path/to/your/custom/results/folder
```

1. **Sample prompts from the dataset:**
```bash
   python sampling.py
```
   This creates `sampled_prompts.csv` with 347 stratified prompts.

2. **completions and score toxicity:**
```bash
   python data_collection.py
```
   Returns three result files: `gpt4omini_results.csv`, `gpt4o_results.csv`, and `gpt4.1_results.csv`. This step requires API access and takes approximately 30-45 minutes with rate limiting.

3. **merge results into analysis dataset:**
```bash
   python results_merge.py
```
   Returns `analysis_dataset_clean.csv` with all model completions merged.

4. **visualizations and summary statistics:**
```bash
   python prompt_stats.py
   python results_analysis.py
```
   This produces all figures and statistical tables used in the report.

**Note:** The original RealToxicityPrompts dataset must be downloaded separately from https://github.com/allenai/real-toxicity-prompts and placed in the `data/` directory as `prompts.jsonl`.


## References

### Primary Source
- Gehman, S., Gururangan, S., Sap, M., Choi, Y., & Smith, N. A. (2020). [RealToxicityPrompts: Evaluating Neural Toxic Degeneration in Language Models](https://arxiv.org/abs/2009.11462). *Findings of EMNLP 2020*.

### Related Work & References
- OpenAI (2023). [GPT-4 Technical Report](https://arxiv.org/abs/2303.08774)
- OpenAI (2023). [GPT-4 System Card](https://cdn.openai.com/papers/gpt-4-system-card.pdf)
- OpenAI (2024). [GPT-4o System Card](https://openai.com/index/gpt-4o-system-card/)
- OpenAI (2024). [OpenAI Safety Update](https://openai.com/index/openai-safety-update/)
- Bai, Y., et al. (2022). [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073)
- Zou, A., et al. (2023). [Universal and Transferable Adversarial Attacks on Aligned Language Models](https://arxiv.org/html/2311.05553v3#bib.bib9)
- AI Magazine (2024). [Is Bigger Always Better? OpenAI Say No with New GPT-4o Mini](https://aimagazine.com/articles/is-bigger-always-better-openai-say-no-with-new-gpt-4o-mini)

### Tools & APIs
- [Perspective API](https://perspectiveapi.com/) - Toxicity detection
- [OpenAI API](https://platform.openai.com/) - Model access
- [RealToxicityPrompts Dataset](https://allenai.org/data/real-toxicity-prompts) - Original prompts

## Author

**Samantha (Sam) Yard**
- GitHub: [@samyard](https://github.com/samyard)
- Project Repository: [mc_capstone](https://github.com/samyard/mc_capstone)
```