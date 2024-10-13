import os
from helper import Helper
# Common data processing
import json
import textwrap
import pandas as pd

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
import openai

from eval import Evaluation
import time



class GraphRAG():
    def __init__(self) -> None:

        load_dotenv('.env', override=True)
        hlp = Helper()
        hlp.clear_log_file()
        self.log = hlp.get_logger()
        
        self.NEO4J_URI = os.getenv('NEO4J_URI')
        self.NEO4J_USERNAME = os.getenv('NEO4J_USERNAME')
        self.NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')
        self.NEO4J_DATABASE = os.getenv('NEO4J_DATABASE') or 'neo4j'

        self.OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
        self.OPENAI_ENDPOINT = os.getenv('OPENAI_BASE_URL') + '/embeddings'

        self.VECTOR_INDEX_NAME = 'article_chunk'
        self.VECTOR_NODE_LABEL = 'Chunk'
        self.VECTOR_SOURCE_PROPERTY = 'text'
        self.VECTOR_EMBEDDING_PROPERTY = 'textEmbedding'

        self.evaluation = Evaluation()

    
    def get_retriever(self):
        try: 
            neo4j_vector_store = Neo4jVector.from_existing_graph(
                embedding = OpenAIEmbeddings(api_key=self.OPENAI_API_KEY),
                url=self.NEO4J_URI,
                username=self.NEO4J_USERNAME,
                password=self.NEO4J_PASSWORD,
                database=self.NEO4J_DATABASE,
                index_name=self.VECTOR_INDEX_NAME,
                node_label=self.VECTOR_NODE_LABEL,
                embedding_node_property=self.VECTOR_EMBEDDING_PROPERTY,
                text_node_properties=[self.VECTOR_SOURCE_PROPERTY]
            )
            retriever = neo4j_vector_store.as_retriever()
            self.log.info('Retriever created successfully')
            return retriever
        except Exception as e:
            self.log.error(f'Error creating retriever: {e}')
            raise e



    def get_chain(self):
        try:
            retriever = self.get_retriever()
            
            if retriever:
                chain = RetrievalQAWithSourcesChain.from_chain_type(
                    ChatOpenAI(temperature=0.5),
                    chain_type="stuff",
                    retriever=retriever, return_source_documents=True)

                self.log.info('Chain created successfully')
                return chain
            else:
                self.log.error('Retriever not created')
                return None
        except Exception as e:
            self.log.error(f'Error creating chain: {e}')
            raise e
    

    def get_response(self, question):
        chain = self.get_chain()
        try:
            """Pretty print the chain's response to a question"""
            start_time = time.time()
            response = chain.invoke({"question": question})

            end_time = time.time()  # Store the end time
            elapsed_time = end_time - start_time

            sources = response.get('source_documents', '')
            source_content = [doc.page_content for doc in sources]
            return textwrap.fill(response['answer'], 60), source_content, elapsed_time
        except Exception as e:
            self.log.error(f'Error pretty printing chain: {e}')
            raise e


    def get_evaluated(self, question, source, answer):

            evaluation = Evaluation()
            embeddings_model = OpenAIEmbeddings(api_key=self.OPENAI_API_KEY)
            # similarity_score = self.evaluation.evaluate_similarity(
            #     embeddings_model,
            #     answer,
            #     source
            # )
            groundedness_score = self.evaluation.groundedness(question, source, answer)
            context_relevancy_score = self.evaluation.context_relevancy(question, source, answer)
            score, reason = self.evaluation.evaluate(question, source, answer)
            # coherence_score, coherence_reason = evaluation.evaluate_coherence(question, answer)
            faithfulness_score, faithfulness_reason = evaluation.evaluate_faithfulness(question, answer, source)
            # #contextual_precision_score, contextual_precision_reason = evaluation.evaluate_contextual_precision(question, answer, source)
            # #contextual_recall_score, contextual_recall_reason = evaluation.evaluate_contextual_recall(question, answer, source)
            hallucination_score, hallucination_reason = evaluation.evaluate_hallucination(question, answer, source)
            # toxicity_score, toxicity_reason = evaluation.evaluate_toxicity(question, answer)
            # bias_score, bias_reason = evaluation.evaluate_bias(question, answer)
            # #ragas_score = evaluation.evaluate_ragas(question, answer, source)

            # return (similarity_score, score, reason, coherence_score, coherence_reason, faithfulness_score,
            #         faithfulness_reason, hallucination_score, hallucination_reason,
            #         toxicity_score, toxicity_reason, bias_score, bias_reason)
    

    def save_to_csv(self, question, source, answer, scores):

        data = {
            'Question': question,
            'Source': source,
            'Answer': answer,
            'Similarity Score': scores[0],
            'Relevancy Score': scores[1],
            'Reason': scores[2],
            'Coherence Score': scores[3],
            'Coherence Reason': scores[4],
            'Faithfulness Score': scores[5],
            'Faithfulness Reason': scores[6],
            'Hallucination Score': scores[7],
            'Hallucination Reason': scores[8],
            'Toxicity Score': scores[9],
            'Toxicity Reason': scores[10],
            'Bias Score': scores[11],
            'Bias Reason': scores[12]
        }

        df = pd.DataFrame([data])

        file_exists = os.path.isfile('er_metrics.csv')

        if file_exists:
            df.to_csv('er_metrics.csv', mode='a', header=False, index=False)
        else:
            df.to_csv('er_metrics.csv', index=False)

        print("Scores saved successfully.")




if __name__ == '__main__':
    graph = GraphRAG()
    questions = ["What role do mast cells play in initiating and directing the host's immune response?",
    "Where do mast cells primarily reside in the body?",
    "How do activated mast cells respond to environmental cues during acute and chronic inflammation?",
    "Name some of the immune mediators released by activated mast cells via rapid degranulation.",
    "What are some of the immune mediators released by mast cells during long-term de novo expression?",
    "What happens to the density of mast cell unique receptors like MRGPRX2 during chronic mast cell activation?",
    "How can the presence of mast cell-related molecules be used to target sites of active inflammation?",
    "What are the two main enzymes released by mast cells upon activation?",
    "What is the role of β-hexosaminidase (β-hex) in mast cell activation?",
    "What percentage of β-hex stored in lung mast cell secretory granules is released minutes after activation?"]
    total_elapsed_time = 0
    question_count = 0
    for question in questions:
        answer, source, elapsed_time = graph.get_response(question)
        print(f"A: {answer}")
        scores = graph.get_evaluated(question, source, answer)
        graph.save_to_csv(question, source, answer, scores)

        total_elapsed_time += elapsed_time
        question_count += 1

    average_time = total_elapsed_time / question_count
    print(f'Average time per question: {average_time} seconds')
