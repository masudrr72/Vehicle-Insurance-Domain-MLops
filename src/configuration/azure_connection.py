import os
from azure.storage.blob import BlobServiceClient

from dotenv import load_dotenv

load_dotenv()


class AzureBlobStorageCient:
    """
    Creates and manages the Azure Blob Storage client.
    """
    def __init__(self):
        self.connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")

        if not self.connection_string:
            raise ValueError("AZURE_STORAGE_CONNECTION_STRING is not set in the environment.")
        
        self.blob_service_client = BlobServiceClient.from_connection_string(self.connection_string)


    def get_blob_service_client(self):
        """
        Returns the initialized BlobServiceClient.
        """

        return self.blob_service_client
