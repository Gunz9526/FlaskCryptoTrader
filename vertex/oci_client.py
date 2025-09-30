# import os
# import oci
# import logging
# from pathlib import Path

# def _get_client_and_ns():
#     config_file = Path(__file__).with_name(".oci").joinpath("config")
#     profile = os.getenv("OCI_PROFILE", "DEFAULT")
#     config = oci.config.from_file(file_location=config_file, profile_name=profile)
#     client = oci.object_storage.ObjectStorageClient(config)
#     namespace = os.getenv("OCI_NAMESPACE") or client.get_namespace().data
#     return client, namespace

# def check_oci_connection(bucket: str) -> bool:
#     try:
#         client, ns = _get_client_and_ns()
#         logging.info(f"[OCI] Namespace resolved: {ns}")
#         resp = client.get_bucket(namespace_name=ns, bucket_name=bucket)
#         logging.info(f"[OCI] Bucket '{bucket}' found. Created at: {resp.data.time_created}")
#         return True
#     except Exception as e:
#         logging.error(f"[OCI] Connection check failed: {e}", exc_info=True)
#         return False

# def upload_object(bucket: str, object_name: str, file_path: str):
#     client, namespace = _get_client_and_ns()
#     with open(file_path, "rb") as f:
#         client.put_object(
#             namespace_name=namespace,
#             bucket_name=bucket,
#             object_name=object_name,
#             put_object_body=f
#         )
#     return f"oci://{namespace}/{bucket}/{object_name}"