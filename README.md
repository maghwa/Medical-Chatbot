# Llama-Healer: A Medical Assistance Bot 🤖🩺:

Welcome to the Llama-Healer repository! Llama-Healer is an intelligent medical chatbot designed to provide medical information and answer health-related queries. It is built using the Llama2 model and Sentence Transformers, powered by Langchain and Chainlit technologies. This bot is designed to run on a CPU machine with at least 16GB of RAM to ensure smooth performance.

## Requirements

To install and run Llama-Healer, you'll need to have the following packages installed:

- `pypdf`
- `langchain`
- `torch`
- `ctransformers`
- `sentence_transformers`
- `faiss_cpu`
- `chainlit`
- `huggingface_hub`

Make sure your Python environment meets these requirements.

## Installation

Clone the repository and navigate to the project directory:

```bash
git clone https://github.com/maghwa/Medical-Chatbot.git
cd Medical-Chatbot
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

To start using the Llama2 Medical Bot, follow these steps:

1. **Start the Bot**:
   - Run the application or use the provided Python script with the command:
     ```bash
     chainlit run model.py -w
     ```

2. **Query the Bot**:
   - Once the bot is running, you can send medical-related queries through the interface provided by Chainlit.

3. **Receive Answers**:
   - The bot will process your query and provide a response based on the medical information available in its database.
   - If sources are available, they will be cited alongside the answer.

4. **Customization**:
   - The bot's responses can be customized according to specific needs by adjusting the retrieval parameters and prompt templates.

## Support

For any issues or questions, please open an issue on this GitHub repository, and I'll get back to you as soon as possible.


---
