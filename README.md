# RAG Agent with LangGraph, Qdrant, and MongoDB Checkpointing

This project is a Retrieval-Augmented Generation (RAG) agent that leverages LangGraph for conversational flow, Qdrant for vector search, and MongoDB for checkpointing conversational state. It is designed to answer user queries based on the content of a PDF file, with iterative answer validation and improvement.

## Features

- **Retrieval-Augmented Generation**: Answers are generated using context retrieved from a PDF, indexed in Qdrant.
- **LangGraph Orchestration**: The conversational flow is managed using LangGraph, with nodes for search, LLM response, and answer validation.
- **Answer Validation Loop**: The agent validates the relevance of its answers and can retry up to 3 times to improve them.
- **MongoDB Checkpointing**: Conversation state is checkpointed using MongoDB for persistence and recovery.

## Requirements

- Python 3.10+
- Qdrant running locally (default: `http://localhost:6333`)
- MongoDB running locally (default: `mongodb://:27017`)
- OpenAI API key (set in your environment)

## Setup

1. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

   (Or install manually: `openai`, `langchain_qdrant`, `langchain_openai`, `langgraph`, `python-dotenv`)

2. **Prepare your environment**:
   - Ensure Qdrant and MongoDB are running locally.
   - Set your OpenAI API key in a `.env` file:

     ```env
     OPENAI_API_KEY=your-key-here
     ```

3. **Index your PDF**:
   - Use `embed.py` to index your PDF into Qdrant:

     ```bash
     python embed.py
     ```

4. **Run the agent**:

   ```bash
   python main.py
   ```

## Usage

- Enter your query at the prompt (`>>`).
- The agent will retrieve relevant context from the PDF, generate an answer, and validate its relevance.
- If the answer is not relevant, the agent will retry up to 3 times.
- If a highly relevant answer cannot be generated, you will be informed.

## File Structure

- `main.py` — Main conversational agent with LangGraph and MongoDB checkpointing.
- `embed.py` — Script to index the PDF into Qdrant.
- `nodejs.pdf` — Example PDF used for retrieval.
- `docker-compose.yml` — (Optional) For running Qdrant and MongoDB via Docker.
- `pyproject.toml`, `requirements.txt`, `uv.lock` — Project dependencies.

## Customization

- To use a different PDF, replace `nodejs.pdf` and re-run `embed.py`.
- Adjust the number of validation attempts or scoring logic in `main.py` as needed.
