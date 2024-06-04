from openai import OpenAI
client = OpenAI()

response = client.embeddings.create(
  model="text-embedding-ada-002",
  input="The food was delicious and the waiter..."
)

print(response)
