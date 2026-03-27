from langchain_ollama.chat_models import ChatOllama
from langchain_core.messages import  SystemMessage

def format_retrieved_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def get_model():
    return ChatOllama(
            model="llama3.1:8b",      # Ollama 模型名称
            temperature=0, 
            validate_model_on_init=True  # 若模型不存在则报错
        )
    

if __name__ == "__main__":
    model = get_model()
    messages = [SystemMessage(content=f"You are helpful assistant in Ovide Clinic, dental care center in California (United States).\nAs reference)")]
    response = model.invoke(messages)
    print(response.content)