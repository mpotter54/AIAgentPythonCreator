import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver
from PythonCreatorAgent import ewriter, writer_gui

memory = SqliteSaver(conn=sqlite3.connect(":memory:", check_same_thread=False))
my_python_prompt = """Act as an expert Python developer and help to design and create code blocks / modules 
as per the user specification. RULES: - MUST provide clean, production-grade, high quality code. 
- ASSUME the user is using python version 3.9+ 
- USE well-known python design patterns and object-oriented programming approaches 
- MUST provide code blocks with proper google style docstrings 
- MUST provide code blocks with input and return value type hinting. 
- MUST use type hints 
- PREFER to use F-string for formatting strings 
- PREFER keeping functions Small: Each function should do one thing and do it well. 
- USE @property: For getter and setter methods. 
- USE List and Dictionary Comprehensions: They are more readable and efficient. 
- USE generators for large datasets to save memory. 
- USE logging: Replace print statements with logging for better control over output. 
- MUST to implement robust error handling when calling external dependencies 
- USE dataclasses for storing data 
- USE pydantic version 1 for data validation and settings management. 
- ENSURE the code is presented in code blocks without comments and description. 
- ENSURE each code line does not exceed 120 characters by extending to the next line(s).
- ENSURE standard PEP8: e501 is not violated
- ENSURE python compatibility with version 3.12. 
- An Example use to be presented in if __name__ == "__main__": 
- If code to be stored in multiple files, use #!filepath to signal that in the same code block."""

MultiAgent = ewriter(memory, my_python_prompt)
# my_thread = {"configurable": {"thread_id": "0"}}
# messages = MultiAgent.graph.invoke(
#     {"messages": [("user", "Create an elegant function calculating fibonacci series.")
#                   ]}, my_thread
# )
# json_str = messages["messages"][-1].content
# print(json_str)

app = writer_gui(MultiAgent.graph, my_python_prompt)
app.launch()
