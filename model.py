# PyPDFLoader is used for loading and extracting text from PDF files.
# DirectoryLoader is used for loading documents from a directory in the file system.
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader

# PromptTemplate is likely a utility for creating and managing prompt templates for language models.
# This can be useful for designing queries or interactions with language models in a structured way.
from langchain.prompts import PromptTemplate

# This class is used for generating embeddings (vector representations) for text using models from Hugging Face's Transformers library.
from langchain_community.embeddings import HuggingFaceEmbeddings

# FAISS is a library for efficient similarity search and clustering of dense vectors. It is used here to store and search through embeddings.
from langchain_community.vectorstores import FAISS

# CTransformers likely refers to a class that interacts with transformer models for language tasks, possibly with some customization or specific implementation details.
from langchain_community.llms import CTransformers

# RetrievalQA is probably a class for building a retrieval-based question-answering system, where questions are answered by retrieving relevant information from a database or corpus.
from langchain.chains import RetrievalQA

import chainlit as cl


DB_FAISS_PATH = 'vectorstore/db_faiss'

custom_prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    # Create an instance of the PromptTemplate class.
    # The 'template' parameter is set to the 'custom_prompt_template' string defined earlier.
    # This string contains placeholders for 'context' and 'question', which will be replaced with actual values at runtime.
    # The 'input_variables' parameter specifies the names of the placeholders in the template. In this case, they are 'context' and 'question'.
    prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=['context', 'question'])

    # Return the prompt instance. This prompt can be used later to generate a full prompt string with specific context and question.
    return prompt


#Loading the model
def load_llm():
    # Load the locally downloaded model here
    llm = CTransformers(
        model = "llama-2-7b-chat.ggmlv3.q8_0.bin",
        model_type="llama",
        max_new_tokens = 512,
        temperature = 0.5
    )
    return llm

# Define a function to set up a retrieval-based question-answering (QA) chain.
def retrieval_qa_chain(llm, prompt, db):
    # Create a RetrievalQA object. This object orchestrates the retrieval of documents relevant to a question
    # and then generates an answer based on those documents using a language model (LLM).
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,  # The language model to use for generating answers.
        chain_type='stuff',  # Placeholder for the type of chain. This should be replaced with the actual chain type.
        retriever=db.as_retriever(search_kwargs={'k': 2}),  # Configures the database to act as a retriever. The 'k': 2 argument specifies that the top 2 most relevant documents should be retrieved for each query.
        return_source_documents=True,  # Indicates that the source documents used to generate the answer should also be returned along with the answer itself.
        chain_type_kwargs={'prompt': prompt}  # Additional arguments specific to the chain type. Here, it includes the custom prompt template to be used in generating answers.
    )

    # Return the configured RetrievalQA chain.
    return qa_chain


# Define a function to initialize and return a question-answering bot.
def qa_bot():
    # Initialize the HuggingFaceEmbeddings object with a pre-trained Sentence Transformers model.
    # This model, "sentence-transformers/all-MiniLM-L6-v2", is used to generate dense vector representations of text.
    # The embeddings are generated on the CPU as specified by 'model_kwargs'.
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': 'cpu'})

    # Load a FAISS vector store from a local file specified by DB_FAISS_PATH.
    # The FAISS database is used for efficient similarity search among the embeddings.
    # The embeddings object is passed to ensure that the database can properly interact with the generated embeddings.
    db = FAISS.load_local(DB_FAISS_PATH, embeddings)

    # Load a language model using the previously defined 'load_llm' function.
    # This model is used for generating answers based on the context and information retrieved.
    llm = load_llm()

    # Initialize a custom prompt template for guiding the language model's answer generation.
    # This is achieved using the 'set_custom_prompt' function, which sets up a specific format for the QA interaction.
    qa_prompt = set_custom_prompt()

    # Create a retrieval-based QA chain that integrates the language model, database, and prompt template.
    # The 'retrieval_qa_chain' function sets up the workflow for processing queries: retrieving relevant information and generating answers.
    qa = retrieval_qa_chain(llm, qa_prompt, db)

    # Return the fully configured QA bot.
    return qa


#output function
def final_result(query):
    qa_result = qa_bot()
    response = qa_result({'query': query})
    return response

#chainlit code
@cl.on_chat_start
async def start():
    chain = qa_bot()
    msg = cl.Message(content="Starting the bot...")
    await msg.send()
    msg.content = "Hello there! I'm Llama-Healer, your virtual health assistant. 😊 How can I assist you with your health queries today?"
    await msg.update()

    cl.user_session.set("chain", chain)

@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain") 
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    cb.answer_reached = True
    res = await chain.ainvoke(message.content, callbacks=[cb])
    answer = res["result"]
    sources = res["source_documents"]

    if sources:
        answer += f"\nSources:" + str(sources)
    else:
        answer += "\nNo sources found"

    await cl.Message(content=answer).send()

