import os
from helper import Helper
# Common data processing
import json
import textwrap

from dotenv import load_dotenv 

# Langchain
from langchain_community.graphs import Neo4jGraph
from langchain_community.vectorstores import Neo4jVector
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_openai import ChatOpenAI

from langchain_community.embeddings import HuggingFaceBgeEmbeddings

from neo4j import GraphDatabase





class NEO4J_KG():
    def __init__(self):
        load_dotenv('.env', override=True)
        hlp = Helper()
        hlp.clear_log_file()  #to clear history
        self.log = hlp.get_logger()

        NEO4J_URI = os.getenv('NEO4J_URI')
        NEO4J_USERNAME = os.getenv('NEO4J_USERNAME')
        NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')
        NEO4J_DATABASE = os.getenv('NEO4J_DATABASE') or 'neo4j'
        
        self.kg = Neo4jGraph(url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD, database=NEO4J_DATABASE)

        self.OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
        self.OPENAI_ENDPOINT = os.getenv('OPENAI_BASE_URL') + '/embeddings'

        self.VECTOR_INDEX_NAME = 'article_chunk'
        self.VECTOR_NODE_LABEL = 'Chunk'
        self.VECTOR_SOURCE_PROPERTY = 'text'
        self.VECTOR_EMBEDDING_PROPERTY = 'textEmbedding'



    def chunk_text(self, text):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 2000,
            chunk_overlap  = 200,
            length_function = len,
            is_separator_regex = False,)
        
        chunks = text_splitter.split_text(text)
        return chunks
    
    
    def create_chunks(self, json_data, filename, chunk_seq_id=0):
        id = filename[-1]
        chunks_with_metadata = []
        chunks = self.chunk_text(json_data['text'])
        self.log.info(f"Text split into {len(chunks)} chunks")
        for i, chunk in enumerate(chunks):
            json_data['text'] = chunk
            json_data['chunkSeqId'] = i
            chunk_id = f"{filename}_{chunk_seq_id}"
            json_data['chunkId'] = chunk_id
            json_data['about'] = 'Inflammation'
            chunks_with_metadata.append(json_data.copy())
            chunk_seq_id += 1
        return chunks_with_metadata
    

    def show_index(self):
        index = self.kg.query("SHOW INDEXES")
        self.log.info(f"Index: {index}")
        return index



    def create_graph_nodes(self, chunk):
        merge_chunk_node_query = """
                                MERGE(mergedChunk:Chunk {chunkId: $chunkParam.chunkId})
                                ON CREATE SET
                                    mergedChunk.source = $chunkParam.source,
                                    mergedChunk.authors = $chunkParam.authors,
                                    mergedChunk.journal = $chunkParam.journal,
                                    mergedChunk.publicationdate = $chunkParam.publicationdate,
                                    mergedChunk.summary = $chunkParam.summary,
                                    mergedChunk.text = $chunkParam.text,
                                    mergedChunk.chunkSeqId = $chunkParam.chunkSeqId,
                                    mergedChunk.article = $chunkParam.article,
                                    mergedChunk.about = $chunkParam.about
                                RETURN mergedChunk
                                """
        
        try:
            self.kg.query(merge_chunk_node_query, params={"chunkParam": chunk})
            self.kg.query("""CREATE CONSTRAINT unique_chunk IF NOT EXISTS 
                          FOR (c:Chunk) REQUIRE c.chunkId IS UNIQUE""")
            # self.log.info(f"Created graph nodes")

        except Exception as e:
            self.log.error(f"An error occurred: {e}")
            raise e

    
    
    
    def create_constraints(self):
        try:
            self.kg.query("""CREATE CONSTRAINT unique_chunk IF NOT EXISTS 
                            FOR (c:Chunk) REQUIRE c.chunkId IS UNIQUE""")
            # self.log.info(f"Created constraint")
        except Exception as e:
            self.log.error(f"Index not created: {e}")
            raise e
    


    def get_number_of_nodes(self):
        node_count_query = """
                            MATCH (n:Chunk)
                            RETURN count(n) as nodeCount
                            """
        node_count = self.kg.query(node_count_query)[0]['nodeCount']
        self.log.info(f"Number of nodes: {node_count}")
        return node_count


    def create_vector_index(self):
        try:
            self.kg.query("""
            CREATE VECTOR INDEX `article_chunk` IF NOT EXISTS
            FOR (c:Chunk) ON (c.textEmbedding) 
            OPTIONS { indexConfig: {
                `vector.dimensions`: 1536,
                `vector.similarity_function`: 'cosine'    
            }}
            """)
            # self.log.info(f"Created vector index")
        except Exception as e:
            self.log.error(f"Vector index not created: {e}")
            raise e

    
    def calculate_embeddings(self):
        try:
            # Create an instance of OpenAIEmbeddings
            embeddings = OpenAIEmbeddings(openai_api_key=self.OPENAI_API_KEY)

            # Retrieve all Chunk nodes without embeddings
            chunks = self.kg.query("""
                MATCH (chunk:Chunk) WHERE chunk.textEmbedding IS NULL
                RETURN chunk
            """)
            for chunk in chunks:
                # Calculate the embedding for the chunk text
                embedding = embeddings.embed_query(chunk['chunk']['text'])

                # Update the Chunk node with the calculated embedding
                self.kg.query("""
                    MATCH (chunk:Chunk {chunkId: $chunkId})
                    SET chunk.textEmbedding = $embedding
                """, params={"chunkId": chunk['chunk']['chunkId'], "embedding": embedding})

            # self.log.info(f"Calculated embeddings")
        except Exception as e:
            self.log.error(f"Embeddings not calculated: {e}")
            raise e


    def create_article_info(self, article):
        try:
            cypher = """
                    MATCH (anyChunk:Chunk) 
                    WHERE anyChunk.article = $article
                    WITH anyChunk LIMIT 1
                    RETURN anyChunk { .source, .authors, .journal, .publicationdate, .summary, .article, .about} as article_info
                    """
            article_info = self.kg.query(cypher, params={'article': article})
            self.log.info(f"Article info created: {article_info}")
            return article_info[0]["article_info"]
        except Exception as e:
            self.log.error(f"Article info not created: {e}")
            raise e
    
    def merge_article_node(self, article_info):
        try:
            cypher = """
                MERGE (f:Article {article: $ArticleInfoParam.article })
                ON CREATE 
                    SET f.authors = $ArticleInfoParam.authors
                    SET f.source = $ArticleInfoParam.source
                    SET f.journal = $ArticleInfoParam.journal
                    SET f.publicationdate = $ArticleInfoParam.publicationdate
                    SET f.summary = $ArticleInfoParam.summary
                    SET f.about = $ArticleInfoParam.about
            """

            self.kg.query(cypher, params={'ArticleInfoParam': article_info})
            self.log.info(f"Article node merged")
        except Exception as e:
            self.log.error(f"Article node not merged: {e}")
            raise e

    def create_relationship(self, article_info):
        try:
            cypher = """
                    MATCH (from_same_article:Chunk)
                    WHERE from_same_article.article = $articleIdParam
                    WITH from_same_article
                        ORDER BY from_same_article.chunkSeqId ASC
                    WITH collect(from_same_article) as section_chunk_list
                        CALL apoc.nodes.link(
                            section_chunk_list, 
                            "NEXT", 
                            {avoidDuplicates: true}
                        )
                    RETURN size(section_chunk_list)
                    """
            self.kg.query(cypher, params={'articleIdParam': article_info['article']})
            self.log.info(f"Relationship created")
        except Exception as e:
            self.log.error(f"Relationship not created: {e}")
            raise e
    

    def connext_to_parent(self):
        try:
            cypher = """
                    MATCH (c:Chunk), (f:Article)
                        WHERE c.article = f.article
                    MERGE (c)-[newRelationship:PART_OF]->(f)
                    RETURN count(newRelationship)
                """
            self.kg.query(cypher)
            self.log.info(f"Connected to parent")
        except Exception as e:
            self.log.error(f"Connection to parent not created: {e}")
            raise e



    def connect_all_article_bidirectional(self):
        try:
            # Cypher query to match and connect all forms bidirectionally
            cypher_query = """
                MATCH (a:Article), (b:Article)
                WHERE a <> b AND a.about = b.about
                MERGE (a)-[:CONNECTED_TO]->(b)
                MERGE (b)-[:CONNECTED_TO]->(a)
                RETURN count(*) AS connections
            """
            result = self.kg.query(cypher_query)
            connections_count = result[0]['connections']
            self.log.info(f"Established bidirectional connections between all forms: {connections_count}")
        except Exception as e:
            self.log.error(f"Error while establishing bidirectional connections: {e}")
            raise e



    def process(self, directory):
        files = os.listdir(directory)
        
        for file in files:
            file_path = os.path.join(directory, file)
            json_data = self.get_json_data(file_path)
            file_name = os.path.basename(file_path.split('.')[-2])
            self.log.info(f"Processing file: {file_name}")
            chunked_with_metadata = self.create_chunks(json_data, file_name)
            # self.log.info(f"Chunked data: {chunked_with_metadata}")

            self.create_vector_index()
            for chunk in chunked_with_metadata:
                self.create_graph_nodes(chunk)
                self.calculate_embeddings()
            article_info = self.create_article_info(chunk["article"])
            self.merge_article_node(article_info)
            self.create_relationship(article_info)
            self.connext_to_parent()

        self.connect_all_article_bidirectional()
        self.kg.refresh_schema()
        print(self.kg.schema)
        total_node_count = self.get_number_of_nodes()
        self.log.info(f"Total number of nodes: {total_node_count}")
    


    def get_json_data(self, file_path):
        try:
            with open(file_path, 'r') as file:
                json_data = json.load(file)
                return json_data
        except FileNotFoundError:
            self.log.error("File not found")
            raise FileNotFoundError
        except Exception as e:
            self.log.error(f"An error occurred: {e}")
            raise e




if __name__ == '__main__':
    kg = NEO4J_KG()
    file_path = 'rpapers_json'
    kg.process(file_path)
    

# from neo4j import GraphDatabase

# uri = "bolt://localhost:7687"
# driver = GraphDatabase.driver(uri, auth=("neo4j", "qwerty_102030"))

# def print_greeting(tx, message):
#     result = tx.run("CREATE (a:Greeting) "
#                     "SET a.message = $message "
#                     "RETURN a.message + ', from node ' + id(a)", message=message)
#     print(result.single()[0])

# with driver.session() as session:
#     session.write_transaction(print_greeting, "hello, world")
# driver.close()
