import os
import sys
from typing import List

from dotenv import load_dotenv
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.model_loader import ModelLoader
from utils.config_loader import load_config
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS



class Retriever:
    def __init__(self):
        self.model_loader = ModelLoader()
        self.config = load_config()
        self._load_env_variables()
        self.vstore = None
        self.retriever = None
        self.faiss_index_path = os.path.abspath(os.getenv("FAISS_INDEX_PATH", "faiss_index"))


    def _load_env_variables(self):
        load_dotenv()
        required_vars = ["GOOGLE_API_KEY", "ASTRA_DB_API_ENDPOINT", "ASTRA_DB_APPLICATION_TOKEN", "ASTRA_DB_KEYSPACE"]
        missing_vars = [var for var in required_vars if os.getenv(var) is None]
        if missing_vars:
            raise EnvironmentError(f"Missing environment variables: {missing_vars}")

    def load_retriever(self):
        if not self.vstore:
            embeddings = self.model_loader.load_embeddings()
            self.vstore = FAISS.load_local(self.faiss_index_path, embeddings, allow_dangerous_deserialization=True)
            print("FAISS index loaded successfully.")

        if not self.retriever:
            top_k = self.config.get("retriever", {}).get("top_k", 50)
            self.retriever = self.vstore.as_retriever(search_kwargs={"k": top_k})
            print("Retriever initialized successfully.")

        return self.retriever

    def rerank_documents(self, query: str, docs: List[Document], llm) -> List[Document]:
        """
        Re-rank retrieved documents based on relevance to the query using the LLM.
        """
        scored_docs = []
        for i, doc in enumerate(docs):
            try:
                if not isinstance(doc, Document):
                    print(f"[WARN] Skipping non-Document at index {i}")
                    continue

                prompt = f"""Rate the relevance of the following review to the query "{query}" on a scale of 1 to 10.
                    Review: {doc.page_content}
                    Respond with only the number."""

                score_response = llm.invoke(prompt)
                score_str = score_response.content.strip()
                score = int(score_str)

                scored_docs.append((doc, score))
            except Exception as e:
                print(f"[WARN] Skipping doc due to error: {e}")

        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in scored_docs]

    def call_retriever(self, query: str) -> List[Document]:
        retriever = self.load_retriever()
        return retriever.invoke(query)

def detect_category_from_query(query: str) -> str:
    query = query.lower()
    if "headphone" in query or "earphone" in query or "headset" in query:
        return "headphones"
    elif "mobile" in query or "smartphone" in query or "phone" in query:
        return "mobiles"
    elif "watch" in query or 'smartwatch' in query:
        return "smart_watches"
    elif "tv" in query or "television" in query:
        return "tv"
    else:
        return "unknown"

if __name__ == '__main__':
    retriever_obj = Retriever()
    user_query = "Can you suggest good budget smart watch?"

    detected_category = detect_category_from_query(user_query)
    print(f"\n[INFO] Detected category: {detected_category}")

    # Step 1: Basic retrieval
    results = retriever_obj.call_retriever(user_query)

    # Step 2: Filter by category
    if detected_category != "unknown":
        category_filtered_results = [doc for doc in results if doc.metadata.get("category") == detected_category]
    else:
        category_filtered_results = results

    if not category_filtered_results:
        print(f"⚠️ No results found for category '{detected_category}'. Falling back to general results.")
        category_filtered_results = results
    else:
        print(f"✅ Found {len(category_filtered_results)} results for category '{detected_category}'.")

    print("\n🔎 [Filtered Results]")
    for idx, doc in enumerate(category_filtered_results, 1):
        print(f"Result {idx}: {doc.page_content}\nMetadata: {doc.metadata}\n")

    # Step 3: Validate Document types
    valid_docs = [doc for doc in category_filtered_results if isinstance(doc, Document)]

    # Step 4: Initialize LLM
    llm = ModelLoader().load_llm()

    # Step 5: Rerank
    reranked = retriever_obj.rerank_documents(user_query, valid_docs, llm)

    print("\n🏅 [After Reranking]")
    for idx, doc in enumerate(reranked, 1):
        print(f"Reranked Result {idx}: {doc.page_content}\nMetadata: {doc.metadata}\n")
