import pandas as pd
import os
from nltk.metrics.agreement import AnnotationTask
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")
base_dir = "Final Results/Merged_data"
dataset_paths = {
    "Hindi": os.path.join(base_dir, "Merged_data_Hindi", "Actual"),
    "Polish": os.path.join(base_dir, "Merged_data_Polish", "Actual")
}

all_results = []

for dataset_name, path in dataset_paths.items():
    files = ["english.csv", "hindi.csv", "polish.csv"]
    lang_dfs = {f.replace(".csv", ""): pd.read_csv(os.path.join(path, f)) for f in files}

    for df in lang_dfs.values():
        df.columns = df.columns.str.strip().str.lower()

    model_sets = [set([col.replace(f"_pred_{lang}", "") for col in df.columns if col.endswith(f"_pred_{lang}")])
                  for lang, df in lang_dfs.items()]
    common_models = set.intersection(*model_sets)

    for model in common_models:
        triples = []
        for idx in range(len(lang_dfs['english'].index)):
            item = str(idx)
            for lang, df in lang_dfs.items():
                print(lang)
                col = f"{model}_pred_{lang}"
                if col in df.columns:
                    print(idx)
                    label = df.iloc[idx][col]
                    label = "hate" if label == 1 else "non-hate"
                    triples.append((lang, item, label))
        try:
            kappa = AnnotationTask(data=triples).multi_kappa()
            all_results.append({"model": model, "dataset": dataset_name, "kappa": kappa})
        except (ZeroDivisionError, StopIteration):
            all_results.append({"model": model, "dataset": dataset_name, "kappa": None})

res_df = pd.DataFrame(all_results).dropna()
graph_output_dir = "Final Results/graphs/FINALPLOT/MultiKappa"
os.makedirs(graph_output_dir, exist_ok=True)
plt.figure(figsize=(10, 6))
models = sorted(res_df["model"].unique())
x = range(len(models))
bar_width = 0.35

for i, dataset in enumerate(res_df["dataset"].unique()):
    subset = res_df[res_df["dataset"] == dataset]
    kappas = [subset[subset["model"] == model]["kappa"].values[0] if model in subset["model"].values else 0 for model in models]
    plt.bar([pos + i * bar_width for pos in x], kappas, width=bar_width, label=dataset)

plt.xticks([pos + bar_width / 2 for pos in x], models, rotation=45)
plt.ylabel("Multi Kappa")
plt.title("Inter-Language Agreement per Model by Dataset")
plt.legend()
plt.tight_layout()
plt.grid(axis='y')
# Plotting simple bars for single dataset (Polish)
# plt.figure(figsize=(6, 5))
# models = sorted(res_df["model"].unique())
# x = range(len(models))
# kappas = [res_df[res_df["model"] == model]["kappa"].values[0] for model in models]
#
# plt.bar(x, kappas, width=0.4, color='tab:blue')
# plt.xticks(x, models, rotation=45)
# plt.ylabel("Multi Kappa")
# plt.title("Inter-Language Agreement per Model")
# plt.tight_layout()
# plt.grid(axis='y')

# Save the plot
plot_path = os.path.join(graph_output_dir, "Multi_kappa_allData.png")
plt.savefig(plot_path, dpi=300, bbox_inches='tight')

