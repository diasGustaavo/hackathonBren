# hackathonBren

## Overview

This project implements a conversational sales agent utilizing the LangChain framework to provide intelligent, context-aware interactions within a sales environment. Leveraging technologies like LangGraph, OpenAI, and FastAPI, the agent facilitates product discovery and recommendations through natural language conversations.

## Features

- Intelligent conversation handling using LangChain and OpenAI's GPT models.
- Product information retrieval based on user queries.
- Recommendations for products by brand, name, or category.
- In-memory storage for conversation histories.
- FastAPI server for easy interaction via HTTP requests.

## Getting Started

### Prerequisites

- Python 3.8 or later
- Pip for Python package installation
- An OpenAI API key

### Installation

1. Clone the repository to your local machine.
2. Navigate to the project directory and install the required dependencies.
3. Create a `.env` file in the root directory of the project and add your OpenAI API key:

    ```
    OPENAI_API_KEY='your_api_key_here'
    ```

## Example API Call

To start a conversation with the sales agent, make a POST request to `/chat/` with a JSON body containing the `user_input` and a `session_id`:

```json
{
  "user_input": "hi! can you help me find any products by Phillips?",
  "session_id": "unique-session-id-123"
}
```

The API will respond with the agent's reply, including product recommendations or information based on the input provided.

## Project Structure

- `.gitignore`: File containing a list of files and directories to be ignored by git.
- `ElectronicsStore.db`: SQLite database file containing the electronics store data.
- `LICENSE`: The license file that describes the terms under which the software is distributed.
- `main.py`: Entry point of the FastAPI application, defining routes and handling API requests.
- `README.md`: The file you're currently reading that provides an overview of the project, setup instructions, and other information.
