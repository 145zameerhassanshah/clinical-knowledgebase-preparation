
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "evaluation_results"

df = pd.read_csv(OUT / "similarity_scores.csv")

lines = [
    "# Medical Embedding Model Evaluation",
    "",
    "## Models",
    "- General: all-MiniLM-L6-v2 (384 dimensions)",
    "- Biomedical: NeuML/biomedbert-base-embeddings (768 dimensions)",
    "",
    f"Total sentence pairs: {len(df)}",
    "",
    "## Category Results",
    "",
    "| Category | General | Biomedical |",
    "|---|---:|---:|",
]

for category, group in df.groupby("category"):
    g = group["general_cosine"].mean()
    b = group["biomedical_cosine"].mean()
    lines.append(f"| {category} | {g:.4f} | {b:.4f} |")

lines += [
    "",
    "## Equivalent vs Different Separation",
    "",
]

for name in ["general", "biomedical"]:
    col = f"{name}_cosine"
    positive = df.loc[df.expected_equivalent == 1, col]
    negative = df.loc[df.expected_equivalent == 0, col]

    lines += [
        f"### {name.title()}",
        f"- Equivalent mean: {positive.mean():.4f}",
        f"- Different mean: {negative.mean():.4f}",
        f"- Separation: {positive.mean() - negative.mean():.4f}",
        "",
    ]

lines += [
    "## Interpretation",
    "",
    "Higher similarity is desirable for equivalent pairs.",
    "For negation and numerical differences, very high similarity",
    "may indicate difficulty distinguishing clinically different statements.",
    "",
    "These results are exploratory and do not establish clinical safety.",
    "",
    "## Visualizations",
    "",
    "![General PCA](pca_general.png)",
    "",
    "![Biomedical PCA](pca_biomedical.png)",
]

path = OUT / "evaluation_report.md"
path.write_text("\n".join(lines), encoding="utf-8")

print("Report created:", path)
