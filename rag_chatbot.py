import os
from langchain_community.llms import LlamaCpp
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
 
model_path = "phi-2.Q2_K.gguf"
llm = LlamaCpp(
    model_path=model_path,
    temperature=0.5,
    max_tokens=256,
    n_ctx=1024,
    n_batch=256
)
 
Data = """
#Weightloss
Weight loss depends on creating a **Caloric Deficit**
Eat fewer calories than you burn (reduce unnecessary carbohydrates)
Focus on high protein and good fat (helps you feel full longer)
Exercise regularly, both weight training and cardio
Drink plenty of water, reduce sugar, reduce fried foods
 
# Eat healthy
Protein: Chicken breast, eggs, nuts, fish
Good carbohydrates: Brown rice, sweet potatoes, oats
Good fats: Avocado, almonds, olive oil
Vegetables and fruits: Green leafy vegetables, berries, bananas, apples
Drink at least 2-3 liters of water/day
 
# How many days per week should you exercise?
3-5 days/week for the average person
5-6 days/week for those who want faster results
2-3 days/week for beginners (let your body adjust)
 
# How many hours of sleep is best?
7-9 hours for adults
8-10 hours for teenagers
Less than 6 hours/night may increase the risk of obesity, stress, and heart disease
"""
with open("data.txt", "w", encoding="utf-8") as f:
    f.write(Data)
 
loader = TextLoader("data.txt", encoding="utf-8")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=20)
docs = text_splitter.split_documents(documents)
 
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
vectorstore = FAISS.from_documents(docs, embeddings)
 
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
 
print("Type a health and fitness question or 'Exit' to exit.")
while True:
    query = input("You : ")
    if query.lower() == "exit":
        print("Thank you for using.")
        break
    result = qa.run(query)
    print(f"Chatbot: {result}")