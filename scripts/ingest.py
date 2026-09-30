"""
RIPTIDE Ingestion Script: Enron Corpus Streaming & Dynamic Chunking
===================================================================
Acceptance Criteria:
  - Process AT LEAST 10,000 emails
  - Dynamic structure-aware chunking (NO fixed-size chunks)
  - Rich metadata per chunk: email_id, chunk_id, chunk_type, chunk_reason, source, text
  - Output to data/chunks.jsonl
  - Robust progress logging and metrics reporting
"""

import os
import sys
import time
import json
from collections import Counter
from pathlib import Path
from typing import Iterator, Dict, Any, Tuple

# Add workspace root to sys.path so backend imports work seamlessly
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT))

from backend.chunker import chunk_email

HF_DATASET_PRIMARY = "parameterlab/scaling_mia_the_pile_00_Enron_Emails"
HF_DATASET_FALLBACK = "haritzpuerto/the_pile_00_Enron_Emails"
PARQUET_DIRECT_URL = (
    "https://huggingface.co/datasets/parameterlab/scaling_mia_the_pile_00_Enron_Emails/"
    "resolve/main/data/train-00000-of-00001.parquet"
)

TARGET_MIN_EMAILS = 10000
OUTPUT_DIR = WORKSPACE_ROOT / "data"
OUTPUT_FILE = OUTPUT_DIR / "chunks.jsonl"
SUMMARY_FILE = OUTPUT_DIR / "ingest_summary.json"


def get_email_stream() -> Tuple[Iterator[Dict[str, Any]], str, float]:
    """
    Attempts to stream the Enron email dataset using HuggingFace datasets,
    with direct pyarrow streaming fallback.
    Returns: (email_iterator, method_name, load_latency_seconds)
    """
    t_start = time.perf_counter()

    # Strategy 1: HuggingFace datasets library (streaming=True)
    try:
        from datasets import load_dataset

        print(f"[Ingest] Attempting streaming load via datasets: {HF_DATASET_PRIMARY}...")
        ds = load_dataset(HF_DATASET_PRIMARY, split="train", streaming=True)
        t_load = time.perf_counter() - t_start
        print(f"[Ingest] Successfully initialized HF streaming generator in {t_load:.2f}s")
        return iter(ds), "hf_datasets_streaming", t_load
    except Exception as e1:
        print(f"[Ingest] datasets.load_dataset primary failed: {e1}")

    try:
        from datasets import load_dataset

        print(f"[Ingest] Trying fallback dataset name: {HF_DATASET_FALLBACK}...")
        ds = load_dataset(HF_DATASET_FALLBACK, split="train", streaming=True)
        t_load = time.perf_counter() - t_start
        print(f"[Ingest] Successfully initialized fallback HF stream in {t_load:.2f}s")
        return iter(ds), "hf_datasets_fallback", t_load
    except Exception as e2:
        print(f"[Ingest] datasets.load_dataset fallback failed: {e2}")

    # Strategy 2: Direct Parquet streaming via pyarrow and requests
    try:
        import urllib.request
        import pyarrow.parquet as pq
        import io

        print(f"[Ingest] Downloading parquet file directly for streaming: {PARQUET_DIRECT_URL}")
        cache_parquet = OUTPUT_DIR / "enron_train.parquet"
        if not cache_parquet.exists():
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            print("[Ingest] Streaming parquet file to local cache...")
            urllib.request.urlretrieve(PARQUET_DIRECT_URL, cache_parquet)
        
        t_load = time.perf_counter() - t_start
        parquet_file = pq.ParquetFile(cache_parquet)

        def parquet_generator():
            for batch in parquet_file.iter_batches(batch_size=1000):
                df = batch.to_pandas()
                for _, row in df.iterrows():
                    yield {"text": row.get("text", "")}

        print(f"[Ingest] Successfully initialized Parquet iterator in {t_load:.2f}s")
        return parquet_generator(), "pyarrow_parquet_stream", t_load
    except Exception as e3:
        raise RuntimeError(f"All dataset loading methods failed: {e3}")


def run_ingestion(target_count: int = TARGET_MIN_EMAILS):
    """Executes the dynamic chunking and ingestion pipeline for at least target_count emails."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print(" RIPTIDE INGESTION: DYNAMIC STRUCTURE-AWARE CHUNKING")
    print(f" Target minimum emails: {target_count:,}")
    print(f" Output file: {OUTPUT_FILE}")
    print("=" * 65)

    stream, method, load_time = get_email_stream()

    emails_processed = 0
    chunks_generated = 0
    skipped_count = 0
    type_distribution = Counter()
    reason_distribution = Counter()

    t_chunk_start = time.perf_counter()
    last_log_time = time.time()

    with open(OUTPUT_FILE, "w", encoding="utf-8") as out_f:
        for item in stream:
            if emails_processed >= target_count:
                break

            raw_text = item.get("text") or item.get("email") or ""
            if not isinstance(raw_text, str) or len(raw_text.strip()) < 15:
                skipped_count += 1
                continue

            email_id = f"enron_{emails_processed + 1:06d}"
            chunks = chunk_email(raw_text, email_id=email_id, source="enron_pile_corpus")

            if not chunks:
                skipped_count += 1
                continue

            for chk in chunks:
                out_f.write(json.dumps(chk, ensure_ascii=False) + "\n")
                type_distribution[chk["chunk_type"]] += 1
                reason_distribution[chk["chunk_reason"]] += 1
                chunks_generated += 1

            emails_processed += 1

            # Progress logging every 1,000 emails or every 3 seconds
            now = time.time()
            if emails_processed % 1000 == 0 or (now - last_log_time) >= 3.0:
                elapsed = time.perf_counter() - t_chunk_start
                rate = emails_processed / elapsed if elapsed > 0 else 0
                avg_chunks = chunks_generated / emails_processed if emails_processed > 0 else 0
                print(
                    f"[{emails_processed:6d}/{target_count:,}] "
                    f"Chunks: {chunks_generated:7d} ({avg_chunks:.1f}/email) | "
                    f"Rate: {rate:5.1f} emails/s | Elapsed: {elapsed:5.1f}s"
                )
                last_log_time = now

    t_chunk_end = time.perf_counter()
    chunking_time = t_chunk_end - t_chunk_start
    file_size_bytes = OUTPUT_FILE.stat().st_size
    file_size_mb = file_size_bytes / (1024 * 1024)

    summary = {
        "status": "COMPLETED",
        "target_emails": target_count,
        "emails_processed": emails_processed,
        "chunks_generated": chunks_generated,
        "avg_chunks_per_email": round(chunks_generated / max(1, emails_processed), 2),
        "skipped_emails": skipped_count,
        "dataset_loading_method": method,
        "dataset_loading_time_seconds": round(load_time, 2),
        "chunking_time_seconds": round(chunking_time, 2),
        "emails_per_second": round(emails_processed / max(0.001, chunking_time), 1),
        "output_file": str(OUTPUT_FILE),
        "output_file_size_bytes": file_size_bytes,
        "output_file_size_mb": round(file_size_mb, 2),
        "chunk_type_distribution": dict(type_distribution),
        "chunk_reason_distribution": dict(reason_distribution.most_common(10)),
        "is_10k_requirement_satisfied": emails_processed >= TARGET_MIN_EMAILS,
    }

    with open(SUMMARY_FILE, "w", encoding="utf-8") as sum_f:
        json.dump(summary, sum_f, indent=2)

    print("\n" + "=" * 65)
    print(" INGESTION COMPLETE - SUMMARY METRICS")
    print("=" * 65)
    print(f" Emails Successfully Processed: {emails_processed:,}")
    print(f" Total Chunks Generated:        {chunks_generated:,}")
    print(f" Average Chunks per Email:      {summary['avg_chunks_per_email']}")
    print(f" Chunk Type Distribution:")
    for ctype, count in type_distribution.items():
        pct = (count / chunks_generated * 100) if chunks_generated else 0
        print(f"   - {ctype:<22}: {count:6d} ({pct:5.1f}%)")
    print(f" Dataset Loading Time:          {load_time:.2f} s")
    print(f" Chunking Time:                 {chunking_time:.2f} s ({summary['emails_per_second']} emails/s)")
    print(f" Output File Size:              {file_size_mb:.2f} MB ({OUTPUT_FILE})")
    print(f" Skipped / Invalid Emails:      {skipped_count}")
    print(f" 10,000-Email Requirement Met:  {'YES - SATISFIED' if summary['is_10k_requirement_satisfied'] else 'NO - FAILED'}")
    print("=" * 65)

    return summary


if __name__ == "__main__":
    count = TARGET_MIN_EMAILS
    if len(sys.argv) > 1:
        try:
            count = int(sys.argv[1])
        except ValueError:
            pass
    run_ingestion(target_count=count)
