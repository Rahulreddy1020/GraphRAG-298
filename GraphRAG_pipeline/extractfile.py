import os
import re
import json
from helper import Helper
from kgopenai import KGOpenai




# extract metadata from text files
class ProcessFile():
    def __init__(self):
        helper = Helper()
        helper.clear_log_file()
        self.log = helper.get_logger()

        self.model = KGOpenai()
    

    def extract_metadata(self, file_content):
        try:
            metadata = self.model.model_prediction(file_content, whatfor='metadata')
            return metadata
        except Exception as e:
            self.log.error(f"Error extracting metadata: {e}")
            return None

    def extract_dict_content(self, text):
        pattern = r'\{[^{}]*\}'
        match = re.search(pattern, text)

        if match:
            dict_content = match.group()
            self.log.info(f"Successfully extracted dictionary content: {dict_content}")
            return dict_content


    def process_article(self, file_name, file_content):
        metadata = None
        attempt_count = 0
        max_attempts = 3

        while not metadata and attempt_count < max_attempts:
            llm_output = self.extract_metadata(file_content)
            self.log.info(f"llm_output: {llm_output}")
            attempt_count += 1
            try:
                dict_content = self.extract_dict_content(llm_output)
                if dict_content:
                    metadata = json.loads(dict_content)
                    if not isinstance(metadata, dict):
                        metadata = None
                        print(f"Attempt {attempt_count}: Extracted content is not a valid dictionary. Retrying...")
                else:
                    print(f"Attempt {attempt_count}: No dictionary content found in the LLM output. Retrying...")
            except json.JSONDecodeError:
                print(f"Attempt {attempt_count}: Extracted content is not a valid JSON string. Retrying...")

        if metadata:
            self.log.info(f"Successfully extracted metadata for {file_name}: {metadata}")

            dict_data = metadata
            dict_data['text'] = file_content
            dict_data['article'] = file_name
            
            with open(f'rpapers_json/{file_name}.json', 'w') as f:
                json.dump(dict_data, f)
                self.log.info(f"Successfully saved metadata to {file_name}.json")
        else:
            self.log.error(f"Failed to extract metadata for {file_name}")
        
    

    def process_files(self, directory):
        for file in os.listdir(directory):
            file_name = file.split('.')[0]
            if file_name == "paper1":
                continue
            if file.endswith(".txt"):
                with open(f"{directory}/{file}", 'r') as f:
                    file_content = f.read()
                    self.process_article(file_name, file_content)


    def change_keys(self, directory):
        for file in os.listdir(directory):
            if file.endswith(".json"):
                with open(f"{directory}/{file}", 'r') as f:
                    data = json.load(f)
                    new_data = {}
                    for k, v in data.items():
                        new_key = k.lower()
                        if new_key == 'title':
                            new_key = 'source'
                        new_data[new_key] = v
                    with open(f"{directory}/{file}", 'w') as f:
                        json.dump(new_data, f)
                        self.log.info(f"Successfully changed keys in {file}")


if __name__ == "__main__":
    pf = ProcessFile()
    pf.process_files('rpapers')
    pf.change_keys('rpapers_json')
    

