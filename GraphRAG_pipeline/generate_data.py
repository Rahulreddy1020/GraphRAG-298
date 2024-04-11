import os
import re
import pandas as pd
from helper import Helper
# from model import GeneratorModel
from docutils import Document
from kgopenai import KGOpenai


class GenerateData:
    def __init__(self):
        helper = Helper()
        helper.clear_log_file()
        self.log = helper.get_logger()

        # self.model = GeneratorModel()
        self.model = KGOpenai()
        self.doc = Document()
    

    def save_to_dataframe(self, data_list, df, base_name, output_dir):
        try:
            n = len(df)

            rows = []
            for i, item in enumerate(data_list):
                item_dict = {'id': n+i, 'input': item['input'], 'output': item['output']}
                rows.append(item_dict)
            
            new_rows = pd.DataFrame(rows)
            self.log.info(f"New Rows: {new_rows}")
            df = pd.concat([df, new_rows], ignore_index=True)
            df.to_csv(f'{output_dir}/{base_name}.csv', index=False)
            self.log.info("Successfully saved to dataframe!")
            return True
        except Exception as e:
            self.log.error(f"Error saving to dataframe: {e}")
            return False
        
    
    
    def post_process_data(self, data):
        pattern = r"<sample>(.*?)</sample>"
        matches = re.findall(pattern, data, re.DOTALL)
        samples = []
    
        for match in matches:
            # match = match.strip().strip('\n')
            match = match.replace('\n', '')
            sample_dict = eval(match)
            samples.append(sample_dict)
        return samples 
        

    def generate_predictions(self, text, base_name, whatfor = 'process', output_dir="extracted_er"):
        try:
            if len(text.split()) > 4000:
                chunks = self.doc.chunk_documents(text)
                processed_output = []
                n = len(chunks)
                for i, chunk in enumerate(chunks):
                    df = self.get_dataframe(base_name, output_dir)
                    self.log.info(f"Processing chunks {i+1}/{n}")
                    chunk_claude = self.model.model_prediction(chunk, whatfor)
                    processed_output = self.post_process_data(chunk_claude)
                    self.save_to_dataframe(processed_output, df, base_name, output_dir)
            else:
                df = self.get_dataframe(base_name, output_dir)
                self.log.info(f"Processing text")
                claude = self.model.model_prediction(text, whatfor)
                processed_output = self.post_process_data(claude)
                self.save_to_dataframe(processed_output, df, base_name, output_dir)

            return True
        
        except Exception as e:
            self.log.error(f"Error generating predictions: {e}")
            return False

    def get_txt_files(self, directory):
        return [f for f in os.listdir(directory) if f.endswith('.txt')]


    def read_data(self, file):
        with open(file, 'r') as f:
            data = f.read()
        return data
    
    def get_dataframe(self, base_name, output_dir):
        if os.path.exists(f'{output_dir}/{base_name}.csv'):
            return pd.read_csv(f'{output_dir}/{base_name}.csv')
        else:
            return pd.DataFrame(columns=['id', 'input', 'output'])

    def generate_training_data(self, directory, output_dir):
        for i, file in enumerate(self.get_txt_files(directory)):
            base_name = file.split('.')[0]
            data = self.read_data(f"{directory}/{file}")
            self.log.info(f"Processing File: {i+1}")
            was_success = self.generate_predictions(data, base_name, whatfor = 'process', output_dir="extracted_er")
            if was_success:
                self.log.info(f"Successfully generated training data for file {i}!")
            else:
                self.log.error("Error generating training data!")

if __name__ == '__main__':
    generate_data = GenerateData()
    input_dir = "rpapers/more_data"
    output_dir = "extracted_er"
    generate_data.generate_training_data(input_dir, output_dir)