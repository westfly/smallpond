import sys
import duckdb
import argparse

sys.path.append(".")

from smallpond.logical.dataset import ParquetDataSet

from smallpond.common import DEFAULT_ROW_GROUP_SIZE, MB
from smallpond.utility import ConcurrentIter


def parse_args():
    parser = argparse.ArgumentParser(
        description="Read Parquet files from S3 using smallpond"
    )
    parser.add_argument(
        "--s3-filename",
        type=str,
        default="s3://your-bucket/path/to/data.parquet",
        help="S3 path to Parquet file(s)",
    )
    parser.add_argument(
        "--s3-dir-path",
        type=str,
        default="",
        help="S3 path to Parquet file(s)",
    )
    parser.add_argument(
        "--region", type=str, default=None, help="AWS region (e.g., us-west-2)"
    )
    parser.add_argument("--number", default=10, type=int)
    parser.add_argument("--repeat", default=3, type=int)
    parser.add_argument(
        "--endpoint",
        type=str,
        default=None,
        help="Custom S3 endpoint for non-AWS S3-compatible storage",
    )
    parser.add_argument(
        "--recursive", action="store_true", help="Recursively search for Parquet files"
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    return parser.parse_args()


def get_file_list(args):
    def parse_s3_url(s3_url):
        from urllib.parse import urlparse

        r = urlparse(s3_url, allow_fragments=False)
        if r.scheme not in ("s3", "s3a"):
            message = "invalid s3 url: %r" % (s3_url,)
            raise ValueError(message)
        path = r.path.lstrip("/")
        return r.netloc, path

    def parse_s3_dir_url(s3_url):
        bucket, path = parse_s3_url(s3_url)
        if not path.endswith("/"):
            path += "/"
        return bucket, path

    import boto3

    if not args.s3_dir_path:
        return [args.s3_filename]

    bucket, path = parse_s3_dir_url(args.s3_dir_path)
    s3_client = boto3.client("s3", endpoint_url=args.endpoint, region_name=args.region)
    objects = s3_client.list_objects_v2(
        Bucket=bucket,
        Prefix=f"{path}",
    )
    if "Contents" not in objects:
        return []
    filelist = []
    for obj in objects["Contents"]:
        print(obj)
        file = obj["Key"]
        if file.endswith("parquet"):
            filelist.append(f"s3://{bucket}/{file}")
    return filelist


def read_file_check(args):
    dataset = ParquetDataSet(["tests/data/arrow/*.parquet"])
    print(f"{dataset.num_rows}")
    dataset = ParquetDataSet(get_file_list(args))
    print(f"{dataset.num_rows}")


# https://stackoverflow.com/questions/74789412/aws-role-vs-iam-credential-in-duckdb-httpfs-call
def load_credentials_into_duckdb(args):
    memdb = duckdb.connect(
        database=":memory:", config={"arrow_large_buffer_size": "true"}
    )
    conn = memdb
    # conn = duckdb.connect()
    conn.execute("INSTALL httpfs;")
    conn.execute("LOAD httpfs;")

    conn.execute(
        """CREATE SECRET secret1(
         TYPE S3,
         PROVIDER CREDENTIAL_CHAIN,
         CHAIN 'instance',
         REGION 'us-east-1',
         ENDPOINT 's3.us-east-1.amazonaws.com'
         )
    """
    )
    query = conn.sql("CALL load_aws_credentials()")
    print(f"{query.loaded_access_key_id}")
    return conn


def read_from_dataset(dataset_path, dataset, conn):
    to_batches = dataset.to_arrow_table(max_workers=1, conn=conn).to_batches(
        max_chunksize=DEFAULT_ROW_GROUP_SIZE * 2
    )
    batch_reader = dataset.to_batch_reader(
        batch_size=DEFAULT_ROW_GROUP_SIZE * 2, conn=conn
    )
    with ConcurrentIter(batch_reader, max_buffer_size=2) as batch_reader:
        for batch_iter in (to_batches, batch_reader):
            total_num_rows = 0
            for batch in batch_iter:
                print(
                    f"batch.num_rows {batch.num_rows}, max_batch_row_size {DEFAULT_ROW_GROUP_SIZE*2}"
                )
                # self.assertLessEqual(batch.num_rows, DEFAULT_ROW_GROUP_SIZE * 2)
                total_num_rows += batch.num_rows
            print(f"{dataset_path}: total_num_rows {total_num_rows}")
            # self.assertEqual(total_num_rows, dataset.num_rows)


def benchmark_read_file(args):
    import timeit

    conn = load_credentials_into_duckdb(args)
    dataset_path = get_file_list(args)
    dataset = ParquetDataSet(dataset_path)
    times = timeit.repeat(
        lambda: read_from_dataset(dataset_path, dataset, conn),
        repeat=args.repeat,
        number=args.number,
    )
    print("Time: {}".format(times))


if __name__ == "__main__":
    args = parse_args()
    read_file_check(args)
    benchmark_read_file(args)
