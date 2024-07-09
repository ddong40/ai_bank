# tf274gpu #

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI

import os 

os.environ['OPENAI_API_KEY'] = 'my APi-kEY'

# Define your OpenAI LLM instance
llm = ChatOpenAI(temperature=0.7, model="gpt-3.5-turbo")

# Define your prompt template
prompt = PromptTemplate(
    input_variables=["input_text"],
    template="You are a helpful assistant. Please respond to the following input: {input_text}"
)

# Create an LLMChain with the LLM and the prompt
chain = LLMChain(llm=llm, prompt=prompt)

# Function to execute the chain
def execute_chain(input_text):
    response = chain.run(input_text=input_text)
    return response

# Example usage
if __name__ == "__main__":
    user_input = "What is the capital of France?"
    response = execute_chain(user_input)
    print('user_input : ', user_input)
    print("Response:", response)