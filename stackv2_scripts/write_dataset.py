import os
import boto3
import threading
from botocore import UNSIGNED
from botocore.client import Config
from smart_open import open
from datasets import Dataset
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

s3 = boto3.client('s3', region_name='me-central-1', config=Config(signature_version=UNSIGNED))

def download_and_save(r, save_dir):
    blob_id, src_encoding, extension = r["blob_id"], r["src_encoding"], r["extension"]
    s3_url = f"s3://softwareheritage/content/{blob_id}"

    try:
        with open(s3_url, "rb", compression=".gz", transport_params={"client": s3}) as s3bucket:
            content = s3bucket.read().decode(src_encoding)

        save_path = os.path.join(save_dir, f"{blob_id}.{extension}")

        with open(save_path, "w", encoding=src_encoding) as f:
            f.write(content)

        return True

    except Exception as e:
        print(f"Failed to download {blob_id}: {e}")
        return False

def main(save_dir, dataset_path, num_threads=8):
    dataset = Dataset.from_parquet(dataset_path)
    os.makedirs(save_dir, exist_ok=True)

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []
        progress = tqdm(desc="Starting downloads", total=len(dataset), unit="file", dynamic_ncols=True)

        for r in dataset:
            future = executor.submit(download_and_save, r, save_dir)
            futures.append(future)
            progress.update(1)
        progress.close()

        progress = tqdm(desc="Waiting downloads", total=len(futures), unit="file", dynamic_ncols=True)
        for future in as_completed(futures):
            future.result()
            progress.update(1)
        progress.close()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Multi-threaded S3 file downloader")
    parser.add_argument("--save-dir", type=str, required=True, help="Directory to save downloaded files")
    parser.add_argument("--dataset-path", type=str, required=True, help="Path to local parquet file")
    parser.add_argument("--num-threads", type=int, default=8, help="Number of threads to use")

    args = parser.parse_args()
    main(args.save_dir, args.dataset_path, args.num_threads)
