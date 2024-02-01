# Import the HuggingFaceEmbeddings class from the langchain_community.embeddings module.
# This class likely wraps around Hugging Face's transformer models to produce embeddings (vector representations) for text.
from langchain_community.embeddings import HuggingFaceEmbeddings

# Import the FAISS class from the langchain_community.vectorstores module.
# FAISS is a library for efficient similarity search and clustering of dense vectors. The FAISS class would be used to store and search through embeddings, typically in a high-dimensional space.
from langchain_community.vectorstores import FAISS

# Import PyPDFLoader and DirectoryLoader from the langchain_community.document_loaders module.
# PyPDFLoader is likely used for loading and extracting text from PDF files.
# DirectoryLoader may be used for loading documents from a directory in the file system.
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader

# Import the RecursiveCharacterTextSplitter class from the langchain.text_splitter module.
# This class is probably used to split text into smaller chunks recursively based on character count, which is useful for processing large texts that exceed the maximum token limit of many NLP models.
from langchain.text_splitter import RecursiveCharacterTextSplitter 

DATA_PATH = 'data/'
DB_FAISS_PATH = 'vectorstore/db_faiss'

# Define a function to create a vector database.
def create_vector_db():
    # Initialize a DirectoryLoader object that will load PDF files from the specified DATA_PATH.
    # It uses a glob pattern to select all files ending with '.pdf'.
    # The loader_cls parameter specifies that the PyPDFLoader class should be used to read the PDF files.
    loader = DirectoryLoader(DATA_PATH,
                             glob='*.pdf',
                             loader_cls=PyPDFLoader)

    # Use the loader to load the documents from the directory.
    documents = loader.load()

    # Create a RecursiveCharacterTextSplitter object, which will split the text from documents into chunks.
    # Each chunk will be 500 characters long with a 50 character overlap with the next chunk.
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,
                                                   chunk_overlap=50)

    # Split the loaded documents into chunks of text using the text splitter.
    texts = text_splitter.split_documents(documents)

    # Initialize HuggingFaceEmbeddings with a specified transformer model to generate embeddings.
    # The embeddings are generated on the CPU.
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                       model_kwargs={'device': 'cpu'})

    # Create a FAISS vector store from the chunks of text using the embeddings.
    # This will index the embeddings for efficient similarity search and retrieval.
    db = FAISS.from_documents(texts, embeddings)

    # Save the FAISS vector store to a local path defined by DB_FAISS_PATH.
    db.save_local(DB_FAISS_PATH)


if __name__ == "__main__":
    create_vector_db()

