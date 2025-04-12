import os
import re
import pickle
import torch
import uvicorn
import openai
import numpy as np
import pandas as pd
from collections import Counter
from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from concurrent.futures import ThreadPoolExecutor
import asyncio
from collections import Counter

# Global variable for cluster mapping (from cluster ID to GPT-generated label)
global_cluster_mapping = {}

# ------------------------------------------------
# Set your OpenAI API key from the environment.
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    print("Warning: OPENAI_API_KEY is not set in your environment.")
else:
    print("✅ OpenAI API key loaded successfully.")
    print("[🔍 DEBUG] OPENAI_API_KEY from env:", openai.api_key[:20] + "...")

# ------------------------------------------------
# Create FastAPI app and mount static/template directories.
app = FastAPI(title="Unified Product Analysis API (0.1.0)")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Create a global executor for running blocking functions.
executor = ThreadPoolExecutor(max_workers=4)

# ------------------------------------------------
# Global DataFrame (updated by CSV upload)
global_df = pd.DataFrame()

# ------------------------------------------------
# 1. Classification Model Setup (BERT-based)
CLASS_MODEL_PATH = "my_bert_model"  # path to your saved classifier model folder
try:
    from transformers import BertForSequenceClassification, BertTokenizerFast
    classifier_model = BertForSequenceClassification.from_pretrained(CLASS_MODEL_PATH)
    classifier_tokenizer = BertTokenizerFast.from_pretrained(CLASS_MODEL_PATH)
except Exception as e:
    print("Error loading classifier model:", e)
    classifier_model, classifier_tokenizer = None, None

label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}

def classify_review(review_text: str) -> str:
    if classifier_model is None or classifier_tokenizer is None:
        raise HTTPException(status_code=500, detail="Classifier model not loaded.")
    inputs = classifier_tokenizer(review_text, return_tensors="pt", truncation=True, padding=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifier_model.to(device)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = classifier_model(**inputs)
    pred_idx = outputs.logits.argmax(dim=-1).item()
    return label_map.get(pred_idx, "Unknown")

# ------------------------------------------------
# 2. Summarization Model Setup (Using ChatGPT via OpenAI API)
def create_detailed_summary_prompt(category: str, top3: list, worst: str, df: pd.DataFrame) -> str:
    def get_reviews(product_name):
        reviews = df[df['name'] == product_name]['reviews.text'].dropna().astype(str).head(5).tolist()
        return reviews if reviews else ["No reviews available."]
    prompt = f"""You are an experienced product analyst writing a professional review article.
You have customer feedback and ratings for the product category "{category}".

Write a narrative-style blog article (around 200 words) that tells a story based on customer experiences.

- Begin with an introduction about the category and its popularity.
- Introduce the top 3 products with a storytelling tone—explain how customers feel and what makes them stand out.
- Seamlessly transition into discussing any customer complaints or issues.
- Conclude with a caution regarding the worst-performing product.

Do not use bullet points, headings, or numbered lists—write it as a continuous article in a professional tone.

Here are some review snippets for context:
- Great Tablet for the casual user
- Love it! Easy to use and fun
"""
    for prod in top3:
        reviews = get_reviews(prod)
        prompt += f"\nProduct: {prod}\nReviews:\n"
        for review in reviews:
            prompt += f"- {review}\n"
    prompt += f"\nWorst Product: {worst}\nReviews:\n"
    worst_reviews = get_reviews(worst)
    for review in worst_reviews:
        prompt += f"- {review}\n"
    prompt += "\nNow write the article in a natural, engaging tone."
    return prompt

def generate_summary_chatgpt_sync(prompt: str, max_tokens: int = 512) -> str:
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an experienced product analyst that writes detailed, professional summaries."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=0.7,
            top_p=1.0
        )
        return response["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print("Error during ChatGPT summarization:", e)
        return "Error generating summary."

async def generate_summary_chatgpt(prompt: str, max_tokens: int = 512) -> str:
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(executor, generate_summary_chatgpt_sync, prompt, max_tokens)
    return result

# ------------------------------------------------
# 3. Clustering Model Setup (SentenceTransformer and HDBSCAN)
SENT_TRANSFORMER_PATH = "sentence_transformer_model (AZ)"
try:
    from sentence_transformers import SentenceTransformer
    sent_transformer = SentenceTransformer(SENT_TRANSFORMER_PATH)
except Exception as e:
    print("Error loading sentence transformer:", e)
    sent_transformer = None

HDBSCAN_MODEL_PATH = "hdbscan_model (AZ).pkl"
try:
    with open(HDBSCAN_MODEL_PATH, "rb") as f:
        hdbscan_model = pickle.load(f)
except Exception as e:
    print("Error loading HDBSCAN model:", e)
    hdbscan_model = None

# ------------------------------------------------
# 4. CSV File Upload Endpoint
@app.post("/upload_csv")
async def upload_csv(file: UploadFile = File(...)):
    global global_df
    try:
        contents = await file.read()
        df = pd.read_csv(pd.io.common.BytesIO(contents), low_memory=False)
        print("[DEBUG] Uploaded CSV columns:", df.columns.tolist())
        print("[DEBUG] Sample data:\n", df.head(3))
        if 'categories' not in df.columns or 'name' not in df.columns:
            raise HTTPException(status_code=400, detail="CSV must contain 'categories' and 'name' columns.")
        df['main_category'] = df['categories'].apply(
            lambda x: x.split(',')[0].strip() if pd.notnull(x) else 'Unknown'
        )
        if 'cleaned_combined' not in df.columns:
            df['cleaned_combined'] = df['categories'].fillna("")
        global_df = df
        print("[✅ DEBUG] CSV uploaded. Unique main categories:", df['main_category'].unique().tolist())
        return {"detail": "CSV file uploaded successfully.", "num_rows": len(df)}
    except Exception as e:
        print("[❌ ERROR] Failed to process CSV:", str(e))
        raise HTTPException(status_code=500, detail=f"Error processing CSV: {str(e)}")

@app.get("/sentiment_stats")
async def get_sentiment_chart_data():
    if global_df.empty:
        raise HTTPException(status_code=400, detail="Data not loaded.")
    comments = global_df["reviews.text"].dropna().astype(str)
    sentiment_map = {"Positive": [], "Neutral": [], "Negative": []}
    for comment in comments:
        text = comment.lower()
        if "bad" in text or "poor" in text:
            sentiment_map["Negative"].append(comment)
        elif "okay" in text or "average" in text:
            sentiment_map["Neutral"].append(comment)
        else:
            sentiment_map["Positive"].append(comment)
    return {
        "labels": list(sentiment_map.keys()),
        "data": [round(np.log1p(len(sentiment_map[k])), 2) for k in sentiment_map],
        "samples": {k: v[:2] for k, v in sentiment_map.items()}
    }

@app.get("/category_distribution")
async def get_category_distribution():
    if global_df.empty:
        raise HTTPException(status_code=400, detail="Data not loaded.")
    dist = global_df["main_category"].value_counts(normalize=True).head(5) * 100
    return {
        "labels": dist.index.tolist(),
        "data": [round(v, 2) for v in dist.values]
    }

# ------------------------------------------------
# Frontend Root Endpoint
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# ------------------------------------------------
# API Endpoints
class ReviewInput(BaseModel):
    review: str

@app.post("/classify")
async def classify_endpoint(input: ReviewInput):
    try:
        result = classify_review(input.review)
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/categories")
async def get_cluster_categories():
    if hdbscan_model is None or sent_transformer is None:
        raise HTTPException(status_code=500, detail="Clustering model not loaded.")
    if global_df.empty:
        raise HTTPException(status_code=500, detail="CSV not uploaded.")
    if "cluster" not in global_df.columns:
        texts = global_df["cleaned_combined"].tolist()
        embeddings = sent_transformer.encode(texts, show_progress_bar=True)
        labels = hdbscan_model.fit_predict(embeddings)
        global_df["cluster"] = labels
    # Get top clusters (excluding noise)
    top_clusters = global_df[global_df["cluster"] != -1]["cluster"].value_counts().head(10).index.tolist()
    def generate_clean_gpt_label(keywords: list[str], cluster_id: int) -> str:
        prompt = f"""You are a data scientist specializing in product reviews.
You are given the following keywords extracted from customer reviews:
{keywords}
Choose the single best category from the following list that describes these reviews: Electronics, Entertainment, Smart, Ereaders, Tablets.
If none clearly match, choose 'Other'.
Return only the chosen word.
"""
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You name product review clusters using one clear, concise word from the given options."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=10,
                temperature=0.3
            )
            label = response["choices"][0]["message"]["content"].strip()
            allowed = {"electronics", "entertainment", "smart", "ereaders", "tablets", "other"}
            if label.lower() not in allowed:
                label = "Other"
            return label.capitalize()
        except Exception as e:
            print("GPT naming error:", e)
            return "Other"
    cluster_names = {}
    seen_norms = {}
    for cluster_id in top_clusters:
        cluster_texts = global_df[global_df["cluster"] == cluster_id]["cleaned_combined"].dropna().tolist()
        words = " ".join(cluster_texts).lower().split()
        filtered_words = [w for w in words if w.isalpha() and len(w) > 3]
        top_keywords = [word for word, _ in Counter(filtered_words).most_common(10)]
        label = generate_clean_gpt_label(top_keywords, cluster_id)
        norm_label = ''.join(ch for ch in label.lower() if ch.isalnum())
        if norm_label in seen_norms:
            continue
        seen_norms[norm_label] = (cluster_id, label)
    distinct = list(seen_norms.values())
    if len(distinct) < 3:
        raise HTTPException(status_code=500, detail="Not enough distinct categories available.")
    distinct = distinct[:6]  # Limit to maximum 6
    final_labels = [label for cid, label in distinct]
    final_clusters = [str(cid) for cid, label in distinct]
    global global_cluster_mapping
    global_cluster_mapping = {str(cid): label for cid, label in zip(final_clusters, final_labels)}
    return {"categories": final_labels, "clusters": final_clusters}

class SummarizeInput(BaseModel):
    category: str

@app.post("/summarize")
async def summarize_endpoint(input: SummarizeInput):
    if global_df.empty:
        raise HTTPException(status_code=500, detail="Data not loaded. Please upload a CSV file.")
    try:
        cluster_num = int(input.category)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid cluster ID")
    cat_df = global_df[global_df["cluster"] == cluster_num]
    if cat_df.empty:
        raise HTTPException(status_code=404, detail="Cluster not found in data.")
    for col in ['reviews.rating', 'name', 'reviews.text']:
        if col not in cat_df.columns:
            raise HTTPException(status_code=500, detail=f"CSV is missing required column: {col}")
    product_avg = cat_df.groupby('name')['reviews.rating'].mean().sort_values(ascending=False)
    if len(product_avg) == 0:
        raise HTTPException(status_code=404, detail="No products found in this cluster.")
    if len(product_avg) < 2:
        only_product = product_avg.index.tolist()[0]
        top3_products = [only_product, only_product, only_product]
        worst_product = only_product
    else:
        top3_products = product_avg.head(3).index.tolist()
        worst_product = product_avg.tail(1).index.tolist()[0]
    # Use the GPT-generated label from global mapping if available
    category_label = global_cluster_mapping.get(str(cluster_num), f"Cluster {cluster_num}")
    prompt = create_detailed_summary_prompt(category_label, top3_products, worst_product, cat_df)
    summary_text = await generate_summary_chatgpt(prompt, max_tokens=512)
    if summary_text == "Error generating summary.":
        raise HTTPException(status_code=500, detail="Failed to generate summary with ChatGPT.")
    return {
        "category": category_label,
        "summary": summary_text,
        "top3_products": top3_products,
        "ratings": [round(product_avg[prod], 2) for prod in top3_products]
    }

@app.get("/clusters")
async def clusters_endpoint():
    if hdbscan_model is None or sent_transformer is None:
        raise HTTPException(status_code=500, detail="Clustering model not loaded.")
    if global_df.empty:
        raise HTTPException(status_code=500, detail="CSV data not uploaded yet.")
    if 'cluster' not in global_df.columns:
        texts = global_df['cleaned_combined'].tolist()
        embeddings = sent_transformer.encode(texts, show_progress_bar=False)
        cluster_labels = hdbscan_model.fit_predict(embeddings)
        global_df['cluster'] = cluster_labels
    clusters = global_df['cluster'].value_counts().to_dict()
    return {"clusters": clusters}

@app.get("/cluster_names")
async def cluster_names_endpoint():
    if global_df.empty or 'cluster' not in global_df.columns:
        raise HTTPException(status_code=500, detail="Clustering has not been performed yet.")
    cluster_names = {}
    for cluster in sorted(global_df['cluster'].unique()):
        cluster_data = global_df[global_df['cluster'] == cluster]
        texts = cluster_data['cleaned_combined'] if 'cleaned_combined' in cluster_data.columns else cluster_data['categories'].fillna("")
        all_words = " ".join(texts.tolist()).lower().split()
        if not all_words:
            cluster_names[str(cluster)] = f"Cluster {cluster}"
        else:
            top_word = Counter(all_words).most_common(1)[0][0]
            cluster_names[str(cluster)] = f"{top_word.capitalize()} ({cluster})"
    return {"cluster_names": cluster_names}

if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
