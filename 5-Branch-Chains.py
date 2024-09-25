from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnableBranch

# api key
GOOGLE_API_KEY = 'AIzaSyA1JZtSuvWkn6S5UP0zgKVi698H2h2S0ww'
# create gemini-pro chat model 
model = ChatGoogleGenerativeAI(model='gemini-pro', google_api_key=GOOGLE_API_KEY)

#create main prompt 
prompt = PromptTemplate(
    input_variables=['feedback'], 
    template=""" you are classifier your role to clssify this feedback: {feedback}/
    to positive or negative or neutral
""" )

#create positive feadback prompt
positive_feadback_prompt = PromptTemplate(
    input_variables=['feedback'], 
    template=""" you are helpful assistant generate response for this positive feedback: {feedback}
""" )

#create negative feadback prompt
negative_feadback_prompt = PromptTemplate(
    input_variables=['feedback'], 
    template=""" you are helpful assistant generate response for this negative feedback: {feedback}
""" )

#create neutral feadback prompt
neutral_feadback_prompt = PromptTemplate(
    input_variables=['feedback'], 
    template=""" you are helpful assistant generate response for this neutral feedback: {feedback}
""" )


# create branches 
braches = RunnableBranch(
    (
        RunnableLambda(lambda x: 'positive' in x, 
                       positive_feadback_prompt | model | StrOutputParser())
    ),

    (
        RunnableLambda(lambda x: 'negative' in x, 
                       negative_feadback_prompt | model | StrOutputParser())
    ),

    (
        RunnableLambda(lambda x: 'neutral' in x, 
                       neutral_feadback_prompt | model | StrOutputParser())
    )
) 

# create chain 
classification_chain = prompt | model | StrOutputParser()

chain = classification_chain | braches

feedback = 'this is a great product'

response = chain.invoke({'feedback': feedback})

print(f'Model Response:\n{response}')
