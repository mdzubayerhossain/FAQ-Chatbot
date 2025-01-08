# FAQ Chatbot

This repository contains the code for a **FAQ Chatbot** built using Streamlit and Groq AI APIs. The chatbot is designed to answer questions from a FAQ document in both Bengali and English.

## Features

- Supports both **Bengali** and **English** languages.
- Extracts relevant chunks from a FAQ document to answer user queries.
- Provides clear and concise responses.
- Uses **Streamlit** for an interactive and user-friendly web interface.
- Includes keyword-based matching for finding relevant content.
- Easy to customize and extend for other FAQs or datasets.

## Prerequisites

Before setting up the project, ensure you have the following installed:

- **Python 3.9 or later**
- **pip (Python Package Manager)**
- A valid Groq API key.

## Installation and Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/mdzubayerhossain/FAQ-Chatbot.git
   cd FAQ-Chatbot

## Installation and Setup

2. **Create a virtual environment and activate it**:
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Linux/Mac
   venv\Scripts\activate      # On Windows
* to the config.json file:*  Create a config.json file in the project directory and add your API key in the following format:

## Install the required dependencies
    ```bash
    pip install -r requirements.txt

## Run the Chatbot

    ```bash
    # To start the chatbot application, run the following command:
    streamlit run app.py

1. **Ask Questions**: Type your questions in the input field.
   - Example (Bengali): "পরিবার পরিকল্পনা কি?"
   - Example (English): "What is family planning?"

2. **Receive Responses**: 
   - The chatbot will display relevant answers from the FAQ content.

3. **Multilingual Support**: 
   - The chatbot supports both Bengali and English.
   - Responses will match the input language automatically.

4. **Debug Mode**:
   - Enable the "Show Debug Info" checkbox in the sidebar to see:
     - The number of FAQ chunks processed.
     - Relevant content length for your queries.

Ask Questions: Type your questions in either Bengali or English in the chat input field.
Receive Responses: The chatbot will display relevant answers from the FAQ content.
Multilingual Support: The chatbot seamlessly switches between Bengali and English based on the input language.
Debug Mode: Use the "Show Debug Info" checkbox in the sidebar to see additional details about the FAQ chunks and system prompts.


##  Debugging Tips
-If you encounter an error regarding the FAQ file, ensure the FAQ.txt file is located in the specified directory (D:\Coding\Llama chatbot\FAQ.txt by default).
-If you see an error with the Groq API key, double-check the config.json file.


