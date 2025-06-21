from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from supabase import create_client
from dotenv import load_dotenv
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
import os
import uuid
import io

# Load environment variables
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Upload image from memory to Supabase
def upload_image_bytes_to_supabase(image_bytes, filename, bucket="plots"):
    supabase.storage.from_(bucket).upload(filename, image_bytes.read(), {
    "content-type": "image/png",
    "x-upsert": "true"
})

    return f"{SUPABASE_URL}/storage/v1/object/public/{bucket}/{filename}"

app = FastAPI()

@app.post("/analyze/")
async def analyze_csv(file: UploadFile = File(...)):
    df = pd.read_csv(file.file)

    response = {}

    # 1. Dtypes
    response["dtypes"] = {col: str(dtype) for col, dtype in df.dtypes.items()}

    # 2. Shape
    response["shape"] = {"rows": df.shape[0], "columns": df.shape[1]}

    # 3. Non-null counts
    response["non_null_counts"] = df.count().to_dict()

    # 4. Impute missing values
    imputer = SimpleImputer(strategy="most_frequent")
    df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    response["post_impute_counts"] = {col: int(df_imputed[col].count()) for col in df_imputed.columns}

    # 5. Medians
    numeric_columns = df.select_dtypes(include='number').columns.tolist()
    response["medians"] = df[numeric_columns].median().to_dict()

    plot_id = str(uuid.uuid4())

    # 6. Boxplot (memory only)
    boxplot_buffer = io.BytesIO()
    plt.figure(figsize=(8, 6))
    for i, col in enumerate(numeric_columns):
        plt.subplot((len(numeric_columns) + 1) // 2, 2, i + 1)
        sns.boxplot(y=df[col])
        plt.title(f"Boxplot of {col.capitalize()}")
    plt.tight_layout()
    plt.savefig(boxplot_buffer, format="png")
    plt.close()
    boxplot_buffer.seek(0)
    boxplot_filename = f"boxplot_{plot_id}.png"
    boxplot_url = upload_image_bytes_to_supabase(boxplot_buffer, boxplot_filename)

    # 7. Heatmap (memory only)
    heatmap_buffer = io.BytesIO()
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(heatmap_buffer, format="png")
    plt.close()
    heatmap_buffer.seek(0)
    heatmap_filename = f"heatmap_{plot_id}.png"
    heatmap_url = upload_image_bytes_to_supabase(heatmap_buffer, heatmap_filename)

    response["boxplot_url"] = boxplot_url
    response["heatmap_url"] = heatmap_url

    # 8. Top 5 correlation pairs
    corr = df.corr(numeric_only=True)
    corr_pairs = corr.unstack()
    corr_pairs = corr_pairs[corr_pairs.index.get_level_values(0) != corr_pairs.index.get_level_values(1)]
    corr_pairs = corr_pairs.drop_duplicates().sort_values(key=abs, ascending=False).head(5)

    top_corr_list = [
        {"feature1": i, "feature2": j, "correlation": round(corr_pairs[(i, j)], 4)}
        for i, j in corr_pairs.index
    ]
    response["top_correlations"] = top_corr_list

    # Get all numeric columns
    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    # If there are at least 2 numeric columns, compute the correlation between the first two
    if len(numeric_cols) >= 2:
        selected_cols = numeric_cols[:2]  # First two numeric columns
        corr_matrix = df[selected_cols].corr()
        response["correlation_matrix"] = corr_matrix.to_dict()
    else:
        response["correlation_matrix"] = {}  # Not enough numeric columns

    return JSONResponse(content=response)
