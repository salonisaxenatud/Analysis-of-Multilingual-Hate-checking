import pandas as pd
import os
from nltk.metrics.agreement import AnnotationTask
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")
# Define dataset locations
base_dir = "Final Results/Merged_data"
dataset_paths = {
    "Hindi": os.path.join(base_dir, "Merged_data_Hindi", "Actual"),
    "Polish": os.path.join(base_dir, "Merged_data_Polish", "Actual")
}

# Collect results for each model in each dataset
all_results = []

for dataset_name, path in dataset_paths.items():
    # Load language-specific files
    files = ["english.csv", "hindi.csv", "polish.csv"]
    lang_dfs = {f.replace(".csv", ""): pd.read_csv(os.path.join(path, f)) for f in files}

    # Normalize column names
    for df in lang_dfs.values():
        df.columns = df.columns.str.strip().str.lower()

    # Find common models
    model_sets = [set([col.replace(f"_pred_{lang}", "") for col in df.columns if col.endswith(f"_pred_{lang}")])
                  for lang, df in lang_dfs.items()]
    common_models = set.intersection(*model_sets)

    # Compute inter-language agreement for each model
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
# res_df = res_df[res_df["model"] == "bloomz_finetuned"]
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

# import pandas as pd
# import os
# from nltk.metrics.agreement import AnnotationTask
# import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.use("TkAgg")
#
# # Define dataset paths for Cultural, Non-Cultural, and HateSpeech
# base_dir = "Final Results/Merged_data"
# conditions = {
#     "Toxic": {"Polish": os.path.join(base_dir, "Merged_data_Polish", "Toxic")},
#     "Non-Toxic": {"Polish": os.path.join(base_dir, "Merged_data_Polish", "Non-toxic")},
#     "HateSpeech": {"Polish": os.path.join(base_dir, "Merged_data_Polish", "Def")}
# }
#
# model_target = "bloomz_finetuned"
# all_results = []
#
# for condition, datasets in conditions.items():
#     for dataset_name, path in datasets.items():
#         try:
#             lang_dfs = {
#                 lang: pd.read_csv(os.path.join(path, f"{lang}.csv")).rename(columns=str.lower)
#                 for lang in ["english", "hindi", "polish"]
#             }
#         except Exception:
#             continue
#
#         # Normalize and clean up Roberta if present
#         for df in lang_dfs.values():
#             df.columns = df.columns.str.strip()
#             df.drop(columns=[col for col in df.columns if col.startswith("roberta_pred_")], inplace=True, errors='ignore')
#
#         model_cols = [f"{model_target}_pred_{lang}" for lang in lang_dfs]
#         if not all(col in df.columns for col, df in zip(model_cols, lang_dfs.values())):
#             continue
#
#         # Use the smallest shared length across languages
#         min_len = min(len(df) for df in lang_dfs.values())
#         triples = []
#         for i in range(min_len):
#             item = str(i)
#             for lang, df in lang_dfs.items():
#                 col = f"{model_target}_pred_{lang}"
#                 label = df.iloc[i][col]
#                 label = "hate" if label == 1 else "non-hate"
#                 triples.append((lang, item, label))
#
#         try:
#             kappa = AnnotationTask(data=triples).multi_kappa()
#             all_results.append({
#                 "model": model_target,
#                 "dataset": dataset_name,
#                 "condition": condition,
#                 "kappa": kappa
#             })
#         except (ZeroDivisionError, StopIteration):
#             continue
#
# # Prepare and plot results
# res_df = pd.DataFrame(all_results).dropna()
# plt.figure(figsize=(8, 6))
# conditions_order = ["Toxic", "Non-Toxic", "HateSpeech"]
# x = range(len(conditions_order))
#
# bar_width = 0.4
# kappas = [
#     res_df[(res_df["dataset"] == "Polish") & (res_df["condition"] == cond)]["kappa"].values[0]
#     if cond in res_df["condition"].values else 0
#     for cond in conditions_order
# ]
#
# plt.bar(x, kappas, width=bar_width, color='tab:blue', label="Polish")
# plt.xticks(x, conditions_order)
# plt.ylabel("Multi Kappa")
# plt.title(f"Inter-Language Agreement for {model_target}")
# plt.legend(title="Dataset")
# plt.tight_layout()
# plt.grid(axis='y')
#
# # Save the plot
# graph_output_dir = "Final Results/graphs/Final_multiKappa/Multi_Kappa_Results"
# os.makedirs(graph_output_dir, exist_ok=True)
# plot_path = os.path.join(graph_output_dir, f"multi_kappa_{model_target}_toxic_non-toxic_polish.png")
# plt.savefig(plot_path, dpi=300, bbox_inches='tight')
