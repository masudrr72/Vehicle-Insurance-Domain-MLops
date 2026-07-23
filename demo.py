
# from src.logger import logging
# logging.debug("This is a debug message")
# logging.info("This is a info message")
# logging.warning("This is a warning message")
# logging.error("This is a error message")
# logging.critical("This is a critical message")

# ---------------------------------------------------------

# from src.logger import logging
# from src.exception import MyException
# import sys

# try:
#     a = 1+'Z'
# except Exception as e:
#     logging.info(e)
#     raise MyException(e, sys) from e
# ------------------------------------------------------

# for connection test( in the origin file)

# if __name__ == "__main__":
#     try:
#         mongodb = MongoDBClient()

#         # Force a connection to MongoDB
#         mongodb.client.admin.command("ping")

#         print("✅ MongoDB connection successful!")
#         print(f"Database Name: {mongodb.database_name}")

#     except Exception as e:
#         print(f"❌ Connection Failed: {e}")

# ---------------------------------------------------------------

# for test dataframe converting

# vehicle_data = VehicleData()
# df = vehicle_data.export_collection_as_dataframe("vehicle_project_data")
# print(df.head())
# print(df.shape)

# ------------------------------------------------------------------

# from src.components.data_ingestion import DataIngestion
# from src.exception import MyException

# if __name__ == "__main__":
#     try:
#         data_ingestion = DataIngestion()
#         artifact = data_ingestion.initiate_data_ingestion()

#         print("\nData Ingestion completed successfully!")
#         print(artifact)

#     except Exception as e:
#         raise MyException(e)



# -------------------------------------------------

# from src.configuration.azure_connection import AzureBlobStorageCient

# client = AzureBlobStorageCient()

# blob_client = client.get_blob_service_client()

# print(type(blob_client))

# -------------------------

from src.pipeline.training_pipeline import TrainPipeline

pipeline = TrainPipeline()
pipeline.run_pipeline()
print("Completed")

