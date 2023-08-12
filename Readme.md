# LangChain QA Chatbot
This project demonstrates the creation of a retrieval-based question-answering chatbot using LangChain, a library for Natural Language Processing (NLP) tasks. The chatbot leverages a pre-trained language model, text embeddings, and efficient vector storage for answering questions based on a given context.

## General Steps
1. Download a Quantized Model: Begin by downloading a quantized version of the LLama 2 chat model. These quantized models are smaller, consume less power, and can be fine-tuned on custom datasets. We're utilizing the quantized version of 7B LLama 2 from TheBloke on Hugging Face.

2. Load the Model: Utilize the ctransformers library to load the downloaded quantized model. This library provides Python bindings for efficient transformer model implementations in C/C++.

3. Choose the Data: Insert the PDF you want to use as data in the data folder.

4. Create Embeddings: Generate text embeddings using the sentence-transformers library. These embeddings convert text data into a dense vector space, allowing for efficient semantic analysis. We use the "all-MiniLM-L6-v2" model from Hugging Face.

5. Create Vector Store: Use the faiss library to establish a vector store for saving the generated text embeddings. FAISS provides an efficient way to search for similar vectors.

6. Retrieval-Based QA: Deploy the LLama 2 model to answer questions based on prompts and utilize FAISS to retrieve relevant answers from the document.

## How to Run
1. Run ingest.py: Execute the command python ingest.py to create embeddings from PDF documents and save them locally.

2. Run ChainLit: Employ ChainLit to run the chatbot application. Execute chainlit run app.py to start the chatbot.

### ingest.py
In the ingest.py script, a vector dataset is created from PDF documents using the LangChain library. This involves converting PDFs into text chunks, further splitting the text, generating text embeddings, and saving them using the FAISS vector store.

### app.py
The app.py script defines a retrieval-based question-answering chatbot using LangChain components. It loads the LLama 2 model, sets up a QA chain with a custom prompt template, and uses FAISS for retrieving relevant answers.

## Code Explanation
* create_vector_db(): Creates a vector database from the PDF data.
* set_custom_prompt(): Defines a custom prompt template for QA retrieval, including context and question placeholders.
* load_llm(): Loads the quantized LLama 2 model using ctransformers.
* retrieval_qa_chain(): Sets up a retrieval-based question-answering chain using the LLama 2 model and FAISS.
* qa_bot(): Combines the embedding, LLama model, and retrieval chain to create the chatbot.
* final_result(query): Calls the chatbot to get a response for a given query.
# ChainLit Implementation
The script uses ChainLit to handle user interactions and execute the chatbot functionality.

## Usage
* On chat start, the bot initializes the QA chain.
* Users can ask questions, and the bot responds with relevant answers.
* If available, sources of information are provided along with the answer.
## Other
The PDF used is "The Project Gutenberg EBook of Encyclopaedia Britannica, 11th Edition, Volume 15, Slice 3": https://www.gutenberg.org/ebooks/41156

This project is based on another GitHub project from AIAnytime and is meant as a learning project: https://github.com/AIAnytime/Llama2-Medical-Chatbot
