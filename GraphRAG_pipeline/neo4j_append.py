from helper import Helper
from PyPDF2 import PdfReader
from io import BytesIO
from docutils import Document
import re
from datetime import datetime
import pandas as pd
import os
from kgopenai import KGOpenai
from neo4j import GraphDatabase


class Neo4jAppend:
    def __init__(self):
        hlp = Helper()
        # hlp.clear_log_file()
        self.log = hlp.get_logger()
        self.log.info("Neo4jAppend initialized")

        self.doc = Document()

        self.model = KGOpenai()

        uri = os.getenv("NEO4J_URI")
        user = os.getenv("NEO4J_USERNAME")
        password = os.getenv("NEO4J_PASSWORD")
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
    

    def pdf_to_text(self, pdf_file):
        self.log.info("Converting PDF to text")
        pdf_reader = PdfReader(BytesIO(pdf_file.read()))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    

    def post_process_data(self, data):
        pattern = r"<output>(.*?)</output>"
        matches = re.findall(pattern, data, re.DOTALL)
        samples = []
    
        for match in matches:
            # match = match.strip().strip('\n')
            match = match.replace('\n', '')
            sample_dict = eval(match)
            samples.append(sample_dict)
        return samples
    

    def get_base_name(self):
        timestamp = datetime.now()
        base_name = "file" + "_" + timestamp.strftime("%Y%m%d%H%M%S")
        return base_name

    def get_dataframe(self, base_name, output_dir):
        if os.path.exists(f'{output_dir}/{base_name}.csv'):
            return pd.read_csv(f'{output_dir}/{base_name}.csv')
        else:
            return pd.DataFrame(columns=['id', 'input', 'output'])
        

    def save_to_dataframe(self, data_list, df):
        try:
            n = len(df)

            rows = []
            for i, item in enumerate(data_list):
                item_dict = {'id': n+i, 'input': item['input'], 'output': item['output']}
                rows.append(item_dict)
            
            new_rows = pd.DataFrame(rows)
            self.log.info(f"New Rows: {new_rows}")
            df = pd.concat([df, new_rows], ignore_index=True)
            
            return df
        except Exception as e:
            self.log.error(f"Error saving to dataframe: {e}")
            return df
        


    def generate_predictions(self, text, base_name, whatfor = 'process', output_dir="streamlit_process"):
        try:
            df = pd.DataFrame(columns=['id', 'input', 'output'])
            if len(text.split()) > 4000:
                chunks = self.doc.chunk_documents(text)
                output = []
                n = len(chunks)
                for i, chunk in enumerate(chunks):
                    self.log.info(f"Processing chunks {i+1}/{n}")
                    chunk_claude = self.model.model_prediction(chunk, whatfor)
                    processed_output = self.post_process_data(chunk_claude)
                    output.append(processed_output)
            else:
                self.log.info(f"Processing text")
                claude = self.model.model_prediction(text, whatfor)
                output = self.post_process_data(claude)
            
            output = sum(output, [])
            return output
        
        except Exception as e:
            self.log.error(f"Error generating predictions: {e}")


    def close(self):
        if self.driver is not None:
            self.driver.close()
    


    def create_graph(self, entity1, entity2, relationship):
        with self.driver.session() as session:
            session.write_transaction(self.__create_and_link_entities, entity1, entity2, relationship)
    


    @staticmethod
    def __create_and_link_entities(tx, entity1, entity2, relationship):
        relationship = relationship.upper().replace("-", "_")
        query = (
            "MERGE (e1:Entity {name: $entity1}) "
            "MERGE (e2:Entity {name: $entity2}) "
            "MERGE (e1)-[r:" + relationship.upper().replace(" ", "_") + "]->(e2)"
        )
        tx.run(query, entity1=entity1, entity2=entity2)

    

    def add_to_neo4j(self, output_list):  
        try:  
            for record in output_list:
                entity1 = record['entity1']
                relationship = record['relationship']
                entity2 = record['entity2']
                self.create_graph(entity1, entity2, relationship)
            return True
        except Exception as e:
            self.log.error(f"Error adding to Neo4j: {e}")
            return False



    def process_file(self, pdf_file):
        output_dir = "streamlit_process"
        text = self.pdf_to_text(pdf_file)
        base_name = self.get_base_name()
        output_list = self.generate_predictions(text, base_name, output_dir=output_dir)
        was_success = self.add_to_neo4j(output_list)
        if was_success:
            print(output_list)
            self.log.info(f"Successfully processed file")
            return True



    # def __init__(self, uri, user, password):
    #     self.__uri = uri
    #     self.__user = user
    #     self.__password = password
    #     self.__driver = GraphDatabase.driver(self.__uri, auth=(self.__user, self.__password))
    #     self.__session = self.__driver.session()


    # def append(self, query):
    #     self.__session.run(query)


    # def close(self):
    #     self.__session.close()
    #     self.__driver.close()


