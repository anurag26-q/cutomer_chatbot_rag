import os
import faiss

def verify_faiss_quantization(index_path: str = "faiss_index/index.faiss"):
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"FAISS index file not found at: {index_path}")

    # Load FAISS index from file
    index = faiss.read_index(index_path)

    print("🔍 FAISS Index Verification Report")
    print("-" * 50)
    print(f"Index Type             : {type(index)}")
    print(f"Is Trained?            : {index.is_trained}")
    print(f"Total Vectors Indexed  : {index.ntotal}")

    # Check if quantizer exists
    if hasattr(index, 'quantizer'):
        print(f"Quantizer Type         : {type(index.quantizer)}")

    # Additional checks for PQ-specific attributes
    if isinstance(index, faiss.IndexIVFPQ):
        print("✅ Quantization Confirmed (IndexIVFPQ)")
        try:
            print(f"-> nlist (centroids)   : {getattr(index, 'nlist', 'N/A')}")
            print(f"-> M (subquantizers)   : {getattr(index, 'pq').M}")
            print(f"-> nbits per subvector : {getattr(index, 'pq').nbits}")
        except AttributeError as e:
            print("⚠️  Could not access PQ parameters:", e)
    else:
        print("❌ This index is not using IVF+PQ quantization.")

if __name__ == "__main__":
    verify_faiss_quantization()
