from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnableParallel

# api key
GOOGLE_API_KEY = 'AIzaSyA1JZtSuvWkn6S5UP0zgKVi698H2h2S0ww'
# create gemini-pro chat model 
model = ChatGoogleGenerativeAI(model='gemini-pro', google_api_key=GOOGLE_API_KEY)


# create base prompt 
prompt = PromptTemplate(
    input_variables=['product'],
    template=""" You are helpful reviewer assistant you job to list the features of product: <<<{product}>>>/
    your output must be list.
"""
)

# create pros prompt 
def analysis_pros_freatures(features):
    pros_prompt = PromptTemplate(
        input_variables=['features'],
        template=""" You are helpful reviewer assistant you job to list ths pros of the featuers <<<{features}>>>/
        your output must be list.
    """
    )
    return pros_prompt.format(features=features)

# create cons prompt 
def analysis_cons_freatures(features):
    cons_prompt = PromptTemplate(
        input_variables=['features'],
        template=""" You are helpful reviewer assistant you job to list ths cons of the featuers  <<<{features}>>>/
        your output must be list.
    """
    )
    return cons_prompt.format(features=features)

# create sequence chain for each brach 
pros_chain_branch = RunnableLambda(
    lambda x: analysis_pros_freatures(x) | model| StrOutputParser()
    )

cons_chain_branch = RunnableLambda(
    lambda x: analysis_cons_freatures(x) | model| StrOutputParser()
    )

# create function to handle and combine 2 branches outputs 
def combine_outputs(pros, cons):
    return f'Pros:\n{pros} \n\nCons:\n{cons}'


# create our full chain 
chain = ( 
    prompt
    |model
    |StrOutputParser() 
    |RunnableParallel(branches={'Pros': pros_chain_branch, 'Cons': cons_chain_branch}) 
    |RunnableLambda(lambda x : combine_outputs(pros=x['branches']['Pros'], cons=x['branches']['Cons']))
)

# get response
response = chain.invoke({'product': 'smasunge A21s'})

print(f'Model Response:\n{response}')

"""
                       +---------------------+
                       |    Input Product    |
                       | (e.g., Samsung A21s)|
                       +----------+----------+
                                  |
                                  |
                                  v
                       +---------------------+
                       |   Base Prompt       |
                       |  (List features of  |
                       |    the product)     |
                       +----------+----------+
                                  |
                                  |
                                  v
                       +---------------------+
                       |    Model (Chat      |
                       |  Google Generative   |
                       |          AI)        |
                       +----------+----------+
                                  |
                                  |
                                  v
                       +---------------------+
                       |   Features Output   |
                       +----------+----------+
                                  |
            +---------------------+---------------------+
            |                                           |
            v                                           v
   +---------------------+                       +---------------------+
   |   Pros Analysis     |                       |   Cons Analysis     |
   |  (List pros of     |                       |  (List cons of     |
   |   features)        |                       |   features)        |
   +----------+----------+                       +----------+----------+
              |                                           |
              |                                           |
              v                                           v
   +---------------------+                       +---------------------+
   |   Model (Chat      |                       |   Model (Chat      |
   |  Google Generative   |                       |  Google Generative   |
   |          AI)        |                       |          AI)        |
   +----------+----------+                       +----------+----------+
              |                                           |
              |                                           |
              v                                           v
   +---------------------+                       +---------------------+
   |    Pros Output      |                       |    Cons Output      |
   +----------+----------+                       +----------+----------+
              |                                           |
              +---------------------+---------------------+
                                    |
                                    v
                       +---------------------+
                       |   Combine Outputs    |
                       |   (Format response)  |
                       +----------+----------+
                                  |
                                  v
                       +---------------------+
                       |   Final Response    |
                       +---------------------+

"""
