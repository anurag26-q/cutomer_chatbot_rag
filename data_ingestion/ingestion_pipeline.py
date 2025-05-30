import os
import sys
import faiss
import numpy as np
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from typing import List
from dotenv import load_dotenv
from utils.model_loader import ModelLoader
from config.config_loader import load_config
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.docstore.in_memory import InMemoryDocstore



class DataIngestion:
    def __init__(self):
        print("Initializing DataIngestion pipeline...")
        self.model_loader = ModelLoader()
        self._load_env_variables()
        self.csv_path = self._get_csv_paths()
        self.product_data = self._load_csv()
        self.config = load_config()
        self.faiss_index_path = os.path.abspath(os.getenv("FAISS_INDEX_PATH", "faiss_index"))




    def _load_env_variables(self):
        load_dotenv()
        required_vars = ["GOOGLE_API_KEY", "ASTRA_DB_API_ENDPOINT", "ASTRA_DB_APPLICATION_TOKEN", "ASTRA_DB_KEYSPACE"]
        missing_vars = [var for var in required_vars if os.getenv(var) is None]
        if missing_vars:
            raise EnvironmentError(f"Missing environment variables: {missing_vars}")

    def _get_csv_paths(self):
        current_dir = os.getcwd()
        data_dir = os.path.join(current_dir, 'data/processed_data')
        
        # List of all CSVs to include
        filenames = [
            # "flipkart_product_review.csv",    
            "headphones_data_cleaned.csv",
            "mobiles_data_cleaned.csv",
        ]

        paths = [os.path.join(data_dir, fname) for fname in filenames]
        for path in paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"CSV file not found at: {path}")
        return paths

    def _load_csv(self):
        dfs = []
        for path in self.csv_path:  # Use self.csv_path instead of calling _get_csv_paths again
            df = pd.read_csv(path)
            expected_columns = {'Title', 'Rating', 'Price', 'Reviews'}
            if not expected_columns.issubset(set(df.columns)):
                raise ValueError(f"CSV at {path} must contain columns: {expected_columns}")

            # Just use the file name without "_data.csv" to determine the category
            category = os.path.basename(path).replace("_data.csv", "")
            df["category"] = category
            dfs.append(df)

        combined_df = pd.concat(dfs, ignore_index=True)
        print('combined_df :',combined_df)
        return combined_df


    def transform_data(self) -> List[Document]:
        documents = []
        clean_data = self.product_data.dropna(subset=["Reviews"])

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=50,
            length_function=len,
            separators=[";",]
        )

        for _, row in clean_data.iterrows():
            review_texts = str(row["Reviews"]).split(";")  # Split multiple reviews
            metadata = {
                "product_name": row["Title"],
                "product_rating": row["Rating"],
                "product_price": row["Price"],
                "product_url": row.get("URL", ""),
                "category": row.get("Category", "unknown")
            }

            for single_review in review_texts:
                single_review = single_review.strip()
                if not single_review:
                    continue

                if len(single_review) > 512:
                    chunks = text_splitter.split_text(single_review)
                    for chunk in chunks:
                        documents.append(Document(page_content=chunk.strip(), metadata=metadata))
                else:
                    documents.append(Document(page_content=single_review, metadata=metadata))

        print(f"Transformed {len(documents)} documents.")
        print(f'documet : {documents[0]}')
        return documents






    def store_in_vector_db(self, documents: List[Document]):
        embeddings = self.model_loader.load_embeddings()
        print("[INFO] Converting documents to vectors for FAISS training...")

        texts = [doc.page_content for doc in documents]

        # Cache file path
        cache_file = "cached_embeddings.npy"

        # Check if cached file exists
        if os.path.exists(cache_file):
            print("[INFO] Loading cached embeddings...")
            vectors_np = np.load(cache_file)
        else:
            print("[INFO] Generating new embeddings...")
            vectors = embeddings.embed_documents(texts)
            vectors_np = np.array(vectors).astype("float32")
            np.save(cache_file, vectors_np)
            print("[INFO] Embeddings cached to:", cache_file)

        dim = vectors_np.shape[1]
        quantizer = faiss.IndexFlatL2(dim)
        index = faiss.IndexIVFPQ(quantizer, dim, 40, 8, 8)

        if not index.is_trained:
            print("[INFO] Training FAISS index with quantized IVF+PQ...")
            index.train(vectors_np)

        index.add(vectors_np)

        docstore = InMemoryDocstore({str(i): doc for i, doc in enumerate(documents)})
        index_to_docstore_id = {i: str(i) for i in range(len(documents))}

        vstore = FAISS(
            embedding_function=embeddings,
            index=index,
            docstore=docstore,
            index_to_docstore_id=index_to_docstore_id
        )

        vstore.save_local(self.faiss_index_path)
        print(f"[INFO] Successfully saved {len(documents)} documents to FAISS index.")
        return vstore

    def load_faiss_index(self):
        embeddings = self.model_loader.load_embeddings()
        vstore = FAISS.load_local(self.faiss_index_path, embeddings, allow_dangerous_deserialization=True)
        return vstore

    def log_review_statistics(self):
        print("\n[INFO] ====== Review Data Summary ======")
        print("Total rows loaded per category:")
        print(self.product_data['category'].value_counts(), end="\n\n")

        clean_data = self.product_data.dropna(subset=["Reviews"])
        clean_data = clean_data[clean_data["Reviews"].str.strip().astype(bool)]  # Remove empty strings

        print("Rows with valid (non-empty) reviews per category:")
        print(clean_data['category'].value_counts(), end="\n\n")

        total_valid_reviews = 0
        for category in clean_data['category'].unique():
            cat_df = clean_data[clean_data['category'] == category]
            review_count = cat_df['Reviews'].apply(lambda x: len(str(x).split(';'))).sum()
            print(f"Estimated review count for '{category}': {review_count}")
            total_valid_reviews += review_count

        print(f"\nTotal estimated individual reviews across all categories: {total_valid_reviews}")
        print("[INFO] ===================================\n")

def detect_category_from_query(query: str) -> str:
        query = query.lower()
        if "headphone" in query or "earphone" in query or "headset" in query:
            return "headphones"
        elif "mobile" in query or "smartphone" in query or "phone" in query:
            return "mobiles"
        elif "watch" in query:
            return "smart_watches"
        elif "tv" in query or "television" in query:
            return "tv"
        else:
            return "unknown"
        
if __name__ == "__main__":
    ingestion = DataIngestion()
    documents = ingestion.transform_data()
    vstore = ingestion.store_in_vector_db(documents)
    vstore = ingestion.load_faiss_index()

    config = load_config()
    top_k = config.get("retriever", {}).get("top_k", 10)

    query = "Can you suggest phones under ₹20000?"
    detected_category = detect_category_from_query(query)

    print(f"[INFO] Detected category from query: {detected_category}")

    results = vstore.similarity_search(query, k=top_k)

    if detected_category == "unknown":
        print("⚠️ Could not determine product category from the query. Showing top results without category filter.")
        for idx, res in enumerate(results, 1):
            print(f"Result {idx}: {res.page_content}\nMetadata: {res.metadata}\n")
    else:
        filtered_results = [res for res in results if res.metadata.get("category") == detected_category]
        if not filtered_results:
            print(f"⚠️ No relevant documents found for category '{detected_category}'. Try rephrasing your query.")
        else:
            for idx, res in enumerate(filtered_results, 1):
                print(f"Result {idx}: {res.page_content}\nMetadata: {res.metadata}\n")

'''if __name__ == "__main__":
    ingestion = DataIngestion()
    documents = ingestion.transform_data()
    vstore = ingestion.store_in_vector_db(documents)
    vstore = ingestion.load_faiss_index()

    config = load_config()
    top_k = config.get("retriever", {}).get("top_k", 3)

    query = "Can you tell me the low budget smart phones?"
    results = vstore.similarity_search(query, k=top_k)
    for idx, res in enumerate(results, 1):
        print(f"Result {idx}: {res.page_content}\nMetadata: {res.metadata}\n")

    ingestion.log_review_statistics()
    documents = ingestion.transform_data()

'''