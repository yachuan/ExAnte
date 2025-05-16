# Wikipedia Pipeline README

### 0. Data Prepartion
Put `exante_wiki.csv` in root.
### 1. Extract Wikipedia Content
Run: 
```bash
python main.py --model claude --method zero_shot --extract
```

Extracts content from Wikipedia pages with fixed cutoff dates

Content will be extracted to `./cutoff_content/{topic_name}/content/` as `{topic_name}_{year}_content.txt`

### 2. Generate Model Predictions and Results
For different models (GPT-4, Claude, Gemini) and different prompting methods (zero shot, few shot, instruction based, chain of thought, generate and validate, run:
```bash
python main.py --model model_name --method method_name
```

Results saved to `./result_with_cutoff/{model_name}/{method_name}`