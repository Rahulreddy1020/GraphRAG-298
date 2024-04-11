import anthropic
from helper import Helper


class GeneratorModel:
    def __init__(self):
        helper = Helper()
        # helper.clear_log_file()
        self.log = helper.get_logger()

        try:
            API_KEY = helper.get_environ_key('ANTHROPIC_API_KEY')
            self.client = anthropic.Client(api_key=API_KEY)
        except Exception as e:
            self.log.error(f"Error connecting to the anthropic API {e}")
            raise e
    
    def get_completion(self, prompt, model='claude-2.1'):
        return self.client.completions.create(prompt=prompt, max_tokens_to_sample=4000, model=model).completion
    

    def get_prompt(self, whatfor='process'):
        try:
            with open(f"prompt/{whatfor}.txt", 'r') as file:
                prompt = file.read()
            return prompt
        except FileNotFoundError:
            raise FileNotFoundError
        except Exception as e:
            self.log.error(f"An error occurred: {e}")
            raise e

    
    def model_prediction(self, text, whatfor='process'):
        prompt = self.get_prompt(whatfor)
        
        input = f"""\n\nHuman: {prompt}
    
                        <article>{text}</article>
                        
                        Assistant:"""
        
        try:
            completion = self.get_completion(input)
            # self.log.info(f"PREDICTION: {completion}")
            return completion
        except Exception as e:
            self.log.error(f"Error generating Prediction: {e}")
            raise e