import pandas as pd
import numpy as np

file_path = "final-multi-prompt(corresponds to phrase language).csv"
df = pd.read_csv(file_path)

def parse_logits(logit_str):
    return np.array([float(x) for x in logit_str.split(',')])

languages = ['polish', 'english', 'hindi']
models = ['roberta', 'bert']
sample_size = 500

compiled_data = []

for lang in languages:
    for model in models:
        conf_col = f'{model}_conf_{lang}'
        pred_col = f'{model}_pred_{lang}'
        true_labels = df['label'].values
        pred_labels = df[pred_col].values
        logits = df[conf_col].apply(parse_logits)

        predicted_logits = [logit[int(pred)] for logit, pred in zip(logits, pred_labels)]
        correctness = pred_labels == true_labels

        indices = np.arange(len(predicted_logits))
        sampled_indices = np.random.choice(indices, sample_size, replace=False) if len(indices) > sample_size else indices

        for idx in sampled_indices:
            compiled_data.append({
                'language': lang,
                'model': model.upper(),
                'predicted_logit': predicted_logits[idx],
                'correct': correctness[idx]
            })

compiled_df = pd.DataFrame(compiled_data)
output_path = "sampled_logit_probabilities.csv"
compiled_df.to_csv(output_path, index=False)

output_path
