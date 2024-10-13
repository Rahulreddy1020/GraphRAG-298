import os
import pandas as pd
import json
from neo4j import GraphDatabase

class Neo4jConnection:
    
    def __init__(self, uri, user, password):
        self.uri = uri
        self.user = user
        self.password = password
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
    
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

def main(csv_file_path, uri, user, password):
    conn = Neo4jConnection(uri, user, password)

    df = pd.read_csv(csv_file_path)

    # Iterate over the DataFrame and populate the Neo4j database
    for index, row in df.iterrows():
        relationships_json = row['output']
        relationships = json.loads(relationships_json.replace("'", '"'))
        for record in relationships:
            entity1 = record['entity1']
            relationship = record['relationship']
            entity2 = record['entity2']
            # Populate Neo4j with the extracted data
            conn.create_graph(entity1, entity2, relationship)

    # Close the Neo4j connection
    conn.close()

if __name__ == "__main__":
    # Define the CSV file path and Neo4j credentials
    csv_file_path = 'full_data.csv'
    uri = "neo4j://localhost:7687"
    user = "neo4j"
    password = "qwerty_102030" 

    main(csv_file_path, uri, user, password)
