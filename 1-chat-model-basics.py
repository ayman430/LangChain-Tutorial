from langchain_google_genai import ChatGoogleGenerativeAI

# api key
GOOGLE_API_KEY = 'AIzaSyA1JZtSuvWkn6S5UP0zgKVi698H2h2S0ww'

# create gemini-pro chat model 
model = ChatGoogleGenerativeAI(model='gemini-pro', google_api_key=GOOGLE_API_KEY)

# ask model and get response
response = model.invoke('in one line define llms', )

# print response
print(f'Model full response:\n{response}')
# content only 
print('#' * 100)
print(f'Response Content:\n{response.content}')

