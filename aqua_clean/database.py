import io
import os
from datetime import datetime
import pymongo
from bson.binary import Binary

class MongoHandler:
    def __init__(self, uri="mongodb://localhost:27017/", db_name="AquaCleanDB", collection_name="detections"):
        self.collection = None
        try:
            self.client = pymongo.MongoClient(uri, serverSelectionTimeoutMS=2000)
            self.client.server_info() # Test connection
            self.db = self.client[db_name]
            self.collection = self.db[collection_name]
            print("Successfully connected to MongoDB.")
        except pymongo.errors.ServerSelectionTimeoutError:
            print("MongoDB not running. Please start MongoDB service.")

    def is_connected(self):
        return self.collection is not None

    def pil_to_binary(self, pil_img):
        """Converts a PIL image to MongoDB Binary format."""
        if pil_img is None: return None
        img_byte_arr = io.BytesIO()
        pil_img.save(img_byte_arr, format='JPEG') 
        return Binary(img_byte_arr.getvalue())

    def save_detection(self, filepath, metrics, orig_pil, res_pil):
        """Saves the data and images to the database."""
        if not self.is_connected():
            raise ConnectionError("MongoDB is not connected.")

        document = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "filename": os.path.basename(filepath),
            "objects_found": metrics['count'],
            "classes": metrics['classes'],
            "input_uiqm": metrics['in_uiqm'],
            "enhanced_uiqm": metrics['out_uiqm'],
            "psnr": metrics['psnr'],
            "processing_time": metrics['time'],
            "images": {
                "input_image": self.pil_to_binary(orig_pil),
                "detected_image": self.pil_to_binary(res_pil)
            }
        }
        self.collection.insert_one(document)