from langchain.text_splitter import RecursiveCharacterTextSplitter
from llama_index.core.node_parser import TokenTextSplitter
from helper import Helper



class Document():
    def __init__(self, chunk_size=1000, chunk_overlap=100):
        hlp = Helper()
        hlp.clear_log_file()
        self.log = hlp.get_logger()
        self.text_splitter = TokenTextSplitter(separator="\n", chunk_size=chunk_size, chunk_overlap=chunk_overlap)


    def chunk_documents(self, text):
        try: 
            chunked_text = self.text_splitter.split_text(text)
            self.log.info("Successfully chunked the text!")
            return chunked_text
        except Exception as e:
            self.log.error("Error chunking the text!")
            return None
    

    def count_tokens(self, text):
        words = text.split()
        num_tokens = len(words)
        return num_tokens
                                                           
