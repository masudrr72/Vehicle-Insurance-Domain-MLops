import os
import sys
import dill

from azure.core.exceptions import ResourceNotFoundError

from src.configuration.azure_connection import AzureBlobStorageCient
from src.exception import MyException
from src.logger import logging


class AzureStorageService:
    """
    Azure Blob Storage helper class.
    Responsible for uploading, downloading and checking models.
    """

    def __init__(self):
        try:
            azure_client = AzureBlobStorageCient()

            self.blob_service_client = (azure_client.get_blob_service_client())

        except Exception as e:
            raise MyException(e, sys)
        

    def get_container_client(self, container_name: str):
        """
        Returns container client.
        """
        try:
            return self.blob_service_client.get_container_client(container_name)

        except Exception as e:
            raise MyException(e, sys)
        

    def blob_exists(self,container_name: str,blob_name: str) -> bool:

        try:

            blob_client = self.blob_service_client.get_blob_client(container=container_name,blob=blob_name)

            return blob_client.exists()

        except Exception as e:
            raise MyException(e, sys)
        


    def upload_file(self,source_file_path: str,container_name: str,blob_name: str, remove: bool = False):

        try:

            blob_client = self.blob_service_client.get_blob_client(container=container_name,blob=blob_name)

            with open(source_file_path, "rb") as data:

                blob_client.upload_blob(data, overwrite=True)

            logging.info(f"Uploaded {blob_name} successfully.")

            if remove:
                os.remove(source_file_path)

        except Exception as e:
            raise MyException(e, sys)
        

    def download_model(self, container_name: str, blob_name: str):
        try:
            blob_client = self.blob_service_client.get_blob_client(container= container_name, blob = blob_name)
            downloaded = blob_client.download_blob()

            model = dill.loads(downloaded.readall())

            logging.info(
                "Production model downloaded successfully.")

            return model

        except ResourceNotFoundError:
            return None

        except Exception as e:
            raise MyException(e, sys)
        

    def download_file(self,container_name: str,blob_name: str,destination_path: str):

        try:

            blob_client = self.blob_service_client.get_blob_client(
                container=container_name,
                blob=blob_name
            )

            os.makedirs(
                os.path.dirname(destination_path),
                exist_ok=True
            )

            with open(destination_path, "wb") as file:

                data = blob_client.download_blob()

                file.write(data.readall())

        except Exception as e:
            raise MyException(e, sys)
        

    def delete_blob(
        self,
        container_name: str,
        blob_name: str
    ):

        try:

            blob_client = self.blob_service_client.get_blob_client(
                container=container_name,
                blob=blob_name
            )

            blob_client.delete_blob()

            logging.info(
                f"{blob_name} deleted."
            )

        except Exception as e:
            raise MyException(e, sys)
        

    def list_blobs(
        self,
        container_name: str,
        prefix: str = None
    ):

        try:

            container = self.get_container_client(
                container_name
            )

            return list(
                container.list_blobs(
                    name_starts_with=prefix
                )
            )

        except Exception as e:
            raise MyException(e, sys)