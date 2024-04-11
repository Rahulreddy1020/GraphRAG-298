import pandas as pd
import matplotlib.pyplot as plt


Models = [ "LLAMA-2 Fine-tuned", "MISTRAL Fine-tuned"]

perplexity =  [1.89, 2.04]

# save it in a pandas dataframe
df = pd.DataFrame({'Models': Models, 'Perplexity': perplexity})
df.to_csv('perplexity.csv', index=False)

