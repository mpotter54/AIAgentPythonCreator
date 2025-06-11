# AIAgentPythonCreator

Create a basic agent to create python code

# Purpose

To create python programs from prompts.

# Features

Gradio UI component to control the agent <br>
Langchain <br>
Uses gemini-2.0-flash LLM from Google <br>

# Installation

Create PYCharm AIAgentPythonCreator project locally in a chosen virtual environment <br>
Add dependencies to virtual environment as described in requirements.txt <br>
Add in src files main.py, PythonCreatorAgent.py <br>
Modify PythonCreatorAgent.py to update your GOOGLE API key. <br>


# Usage

Run main.py from PYCharm project <br>
![Run project in PyCharm](RunMainInPyCharm.png) <br>
System will create a local URL "* Running on local URL:  http://127.0.0.1:7860" <br>
Click on link to instance the Gradio UI in your default browser <br>
Paste contents of PythonPrompt.txt into Python Prompt text box <br>
Click Create Python button to run the agent to create the Python Program <br>
![Run agent in browser](ClickPythonCreateButtonInBrowser.png) <br>
Use the copy button to copy the Python Output text box into the paste buffer <br>

