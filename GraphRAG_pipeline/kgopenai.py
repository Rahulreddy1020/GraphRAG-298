import re
from helper import Helper
from docutils import Document
from openai import OpenAI
import tiktoken



class KGOpenai():
    def __init__(self):
        hlp = Helper()
        #hlp.clear_log_file()  #to clear history
        self.log = hlp.get_logger()

        try:
            self.openai_client = OpenAI()
            self.log.info("Successfully connected to the OPENAI API")
        except Exception as e:
            self.log.error(f"Error connecting to the Gemini API {e}")
            raise e

        self.encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")


    def truncate_text(self, text, max_tokens=15000):
        tokens = self.encoding.encode(text)
        if len(tokens) > max_tokens:
            tokens = tokens[:max_tokens]
            text = self.encoding.decode(tokens)
            self.log.info(f"Text truncated to {max_tokens} tokens")
        return text
        


    def get_prompt(self, whatfor='metadata'):
        try:
            with open(f"prompt/{whatfor}.txt", 'r') as file:
                prompt = file.read()
            return prompt
        except FileNotFoundError:
            raise FileNotFoundError
        except Exception as e:
            self.log.error(f"An error occurred: {e}")
            raise e

    
    def model_prediction(self, text, whatfor='metadata'):
        text = self.truncate_text(text)
        prompt = self.get_prompt(whatfor)

        
        if whatfor == 'metadata':
            input = f"""{prompt}
                        Research article text:

                        {text}
                        """
        else:
            input = f"""{prompt}
                        {text}
                        """
        
        try:
            output = self.openai_client.chat.completions.create(
                            model="gpt-3.5-turbo",
                            temperature=0.2,
                            messages=
                            [
                                {"role": "user",
                                "content": input
                                }
                            ]
                            ).choices[0].message.content
            # self.log.info(f"PREDICTION: {output}")
            return output
        except Exception as e:
            self.log.error(f"Error generating Prediction: {e}")
            raise e