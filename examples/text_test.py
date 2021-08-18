import json
from ogm.utils import text_data_preprocess

# Load JSON file as parameter for text_data_preprocess
with open("text_test.json", "r") as infile:
    preprocess_config = json.load(infile)

# Run preprocessor -- saves results to CSV file
text_data_preprocess(preprocess_config)
