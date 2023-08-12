from langchain import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
import chainlit as cl

DB_FAISS_PATH = "vectorstores/db_faiss" # Get the embeddings path


custom_prompt_template = """Use the following information to answer the users question, if you dont know the answer
just say " I don't know the answer". DO NOT make up answers that are not based on facts. Explain with detailed answers
that are easy to understand

Context: {context}
Question: {question}

Only return the useful aspects of the answer below and nothing else.
Helpful answer:
"""

## Code for prompt, retrieval and bot 
def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vector store, we also pass in context and question.
    """
    prompt = PromptTemplate(template= custom_prompt_template, input_variables=['context','question'])
    return prompt

def load_llm():
    """
    Loading the llama2 model we have installed using CTransformers
    """
    llm = CTransformers(
        model= "llama-2-7b-chat.ggmlv3.q8_0.bin",
        model_type= "llama",
        max_new_tokens = 512,
        temperature = 0.5
    )
    return llm

def retrieval_qa_chain(llm,prompt,db):
    """
    Setting up a retrieval-based question-answering chain,
    and returning response
    """
    qa_chain = RetrievalQA.from_chain_type(llm = llm,
                            chain_type = 'stuff',
                            retriever = db.as_retriever(search_kwargs = {'k': 2}),
                            return_source_documents = True,
                            chain_type_kwargs = {'prompt':prompt}
                        )
    
    return qa_chain

def qa_bot():
    """
    Loading the db and using it in retrieval_qa_chain
    """
    embeddings = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2', model_kwargs = {'device':'cuda'})
    
    db = FAISS.load_local(DB_FAISS_PATH, embeddings)
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm,qa_prompt,db)
    return qa


def final_result(query):
    qa_result = qa_bot()
    response = qa_result({'query':query})
    return response



## Code for chainlit

@cl.on_chat_start
async def start():
    chain = qa_bot()
    msg = cl.Message(content="Starting the bot...")
    await msg.send()
    msg.content = "Hi, Welcome to Encyclopaedia Britannica 11t Edition, Japan to Jeveros . What is your question?"
    await msg.update()

    cl.user_session.set("chain", chain)

@cl.on_message
async def main(message):
    chain = cl.user_session.get("chain") 
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    cb.answer_reached = True
    res = await chain.acall(message, callbacks=[cb])
    answer = res["result"]
    sources = res["source_documents"]

    if sources:
        answer = f"\nSources:" + str(sources)
    else:
        answer = "\nNo sources found"

    await cl.Message(content=answer).send()
    