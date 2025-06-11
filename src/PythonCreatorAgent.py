from langchain_core.runnables import RunnableWithFallbacks, RunnableLambda
from langgraph.graph import StateGraph, START, END, add_messages
from langchain_core.messages import AnyMessage, AIMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field
from typing import TypedDict
from typing import Any, Annotated, Literal
import os
import warnings
import gradio as gr

warnings.filterwarnings("ignore", message=".*TqdmWarning.*")

os.environ["GOOGLE_API_KEY"] = "ADD GOOGLE API KEY HERE"

# Define the state for the agent
class State(TypedDict):
    task: str
    messages: Annotated[list[AnyMessage], add_messages]


class ewriter():
    def __init__(self, memory, python_prompt):
        self.python_prompt = python_prompt

        # Define a new graph
        self.workflow = StateGraph(State)

        # Add a node for a model to generate python based on the question
        self.python_prompt = ChatPromptTemplate.from_messages(
            [("system", self.python_prompt), ("placeholder", "{messages}")]
        )
        self.llm_python = ChatGoogleGenerativeAI(model="gemini-2.0-flash",
                                                 temperature=0,
                                                 max_tokens=8196,
                                                 timeout=None,
                                                 max_retries=2,
                                                 max_output_tokens=8196)
        self.llm_python_gen = (self.python_prompt | self.llm_python)
        self.workflow.add_node("python_gen", self.python_gen_node)

        # Specify the edges between the nodes
        self.workflow.add_edge(START, "python_gen")
        self.workflow.add_edge("python_gen", END)
        self.memory = memory
        self.graph = self.workflow.compile(checkpointer=self.memory)

    def python_gen_node(self, state: State):
        message = self.llm_python_gen.invoke(state)
        return {"messages": [message]}


class writer_gui():
    def __init__(self, graph, python_prompt, share=False):
        self.python_prompt = python_prompt
        self.graph = graph
        self.share = share
        self.partial_message = ""
        self.response = {}
        self.max_iterations = 10
        self.iterations = []
        self.threads = []
        self.thread_id = -1
        self.thread = {"configurable": {"thread_id": str(self.thread_id)}}
        self.demo = self.create_interface()

    def run_agent(self, start, topic, stop_after):
        if start:
            self.iterations.append(0)
            self.thread_id += 1  # new agent, new thread
            self.threads.append(self.thread_id)
            config = None
        self.thread = {"configurable": {"thread_id": str(self.thread_id)}}
        python_qry = {'task': topic,
                      "messages": [("user", topic)]}
        self.response = self.graph.invoke(python_qry, self.thread)
        self.iterations[self.thread_id] += 1
        self.partial_message += str(self.response)
        self.partial_message += f"\n------------------\n\n"
        # print("Hit the end")
        return

    def get_disp_state(self, ):
        current_state = self.graph.get_state(self.thread)
        # lnode = current_state.values["lnode"]
        # acount = current_state.values["count"]
        # rev = current_state.values["revision_number"]
        nnode = current_state.next
        #print  (lnode,nnode,self.thread_id,rev,acount)
        # return lnode, nnode, self.thread_id, rev, acount
        return nnode, self.thread_id

    def get_state(self, key):
        current_values = self.graph.get_state(self.thread)
        if key in current_values.values:
            nnode, self.thread_id, rev, astep = self.get_disp_state()
            new_label = f"thread_id: {self.thread_id}, step: {astep}"
            return gr.update(label=new_label, value=current_values.values[key])
        else:
            return ""

    def get_content(self, ):
        current_values = self.graph.get_state(self.thread)
        if "content" in current_values.values:
            content = current_values.values["content"]
            nnode, thread_id, astep = self.get_disp_state()
            new_label = f"thread_id: {self.thread_id}, step: {astep}"
            return gr.update(label=new_label, value="\n\n".join(item for item in content) + "\n\n")
        else:
            return ""

    def update_hist_pd(self, ):
        #print("update_hist_pd")
        hist = []
        # curiously, this generator returns the latest first
        for state in self.graph.get_state_history(self.thread):
            if state.metadata['step'] < 1:
                continue
            thread_ts = state.config['configurable']['thread_ts']
            tid = state.config['configurable']['thread_id']
            # count = state.values['count']
            # lnode = state.values['lnode']
            # rev = state.values['revision_number']
            nnode = state.next
            st = f"{tid}:{nnode}:{thread_ts}"
            hist.append(st)
        return gr.Dropdown(label="update_state from: thread:last_node:next_node:rev:thread_ts",
                           choices=hist, value=hist[0], interactive=True)

    def find_config(self, thread_ts):
        for state in self.graph.get_state_history(self.thread):
            config = state.config
            if config['configurable']['thread_ts'] == thread_ts:
                return config
        return (None)

    def copy_state(self, hist_str):
        ''' result of selecting an old state from the step pulldown. Note does not change thread.
             This copies an old state to a new current state.
        '''
        thread_ts = hist_str.split(":")[-1]
        # print(f"copy_state from {thread_ts}")
        config = self.find_config(thread_ts)
        # print(config)
        state = self.graph.get_state(config)
        self.graph.update_state(self.thread, state.values, as_node=state.values['lnode'])
        new_state = self.graph.get_state(self.thread)  # should now match
        new_thread_ts = new_state.config['configurable']['thread_ts']
        tid = new_state.config['configurable']['thread_id']
        # count = new_state.values['count']
        # lnode = new_state.values['lnode']
        # rev = new_state.values['revision_number']
        nnode = new_state.next
        return nnode, new_thread_ts

    def update_thread_pd(self, ):
        # print("update_thread_pd")
        return gr.Dropdown(label="choose thread", choices=self.threads, value=self.thread_id, interactive=True)

    def switch_thread(self, new_thread_id):
        # print(f"switch_thread{new_thread_id}")
        self.thread = {"configurable": {"thread_id": str(new_thread_id)}}
        self.thread_id = new_thread_id
        return

    def modify_state(self, key, asnode, new_state):
        ''' gets the current state, modifes a single value in the state identified by key, and updates state with it.
        note that this will create a new 'current state' node. If you do this multiple times with different keys, it will create
        one for each update. Note also that it doesn't resume after the update
        '''
        current_values = self.graph.get_state(self.thread)
        current_values.values[key] = new_state
        self.graph.update_state(self.thread, current_values.values, as_node=asnode)
        return

    def create_interface(self):
        def_python = '''Create an elegant function calculating fibonacci series'''

        with gr.Blocks(theme=gr.themes.Default(spacing_size='sm', text_size="sm")) as demo:

            def updt_disp():
                ''' general update display on state change '''
                json_str: str = '*****\n'
                current_state = self.graph.get_state(self.thread)
                hist = []
                # curiously, this generator returns the latest first
                for state in self.graph.get_state_history(self.thread):
                    if state.metadata['step'] < 1:  # ignore early states
                        continue
                    if "thread_ts" in state.config:
                        s_thread_ts = state.config['configurable']['thread_ts']
                    else:
                        s_thread_ts = ''
                    s_tid = state.config['configurable']['thread_id']
                    # s_count = state.values['count']
                    # s_lnode = state.values['lnode']
                    # s_rev = state.values['revision_number']
                    s_nnode = state.next
                    st = f"{s_tid}:{s_nnode}:{s_thread_ts}"
                    hist.append(st)
                if not current_state.metadata:  # handle init call
                    return {}
                else:
                    if len(self.response) < 1:
                        for msg in current_state[0]['messages']:
                            if len(msg.content) > 0 and isinstance(msg, AIMessage):
                                json_str += msg.content
                                json_str += '\n'
                        json_str += '*****'
                    else:
                        json_str = self.response["messages"][-1].content
                    return {
                        topic_bx: current_state.values["task"],
                        threadid_bx: self.thread_id,
                        live: json_str,
                        thread_pd: gr.Dropdown(label="choose thread", choices=self.threads, value=self.thread_id,
                                               interactive=True),
                        step_pd: gr.Dropdown(label="update_state from: thread:count:last_node:next_node:rev:thread_ts",
                                             choices=hist, value=hist[0], interactive=True),
                    }

            def get_snapshots():
                new_label = f"thread_id: {self.thread_id}, Summary of snapshots"
                sstate = ""
                for state in self.graph.get_state_history(self.thread):
                    for key in ['plan', 'draft', 'critique']:
                        if key in state.values:
                            state.values[key] = state.values[key][:80] + "..."
                    if 'content' in state.values:
                        for i in range(len(state.values['content'])):
                            state.values['content'][i] = state.values['content'][i][:20] + '...'
                    if 'writes' in state.metadata:
                        state.metadata['writes'] = "not shown"
                    sstate += str(state) + "\n\n"
                return gr.update(label=new_label, value=sstate)

            def vary_btn(stat):
                # print(f"vary_btn{stat}")
                return gr.update(variant=stat)

            with gr.Tab("Python Prompt"):
                with gr.Row():
                    topic_bx = gr.Textbox(label="Python Prompt", value=def_python, lines=10, max_lines=10)
                    gen_btn = gr.Button("Create Python", scale=0, min_width=80, variant='primary')
                    cont_btn = gr.Button("Continue Execution", scale=0, min_width=80, visible=False)
                with gr.Row():
                    threadid_bx = gr.Textbox(label="Thread", scale=0, min_width=10, visible=False)
                with gr.Accordion("Manage Agent", open=False):
                    checks = list(self.graph.nodes.keys())
                    checks.remove('__start__')
                    stop_after = gr.CheckboxGroup(checks, label="Interrupt After State", value=checks, scale=0,
                                                  min_width=400, visible=False)
                    with gr.Row():
                        thread_pd = gr.Dropdown(choices=self.threads, interactive=True, label="select thread",
                                                min_width=120, scale=0)
                        step_pd = gr.Dropdown(choices=['N/A'], interactive=True, label="select step", min_width=160,
                                              scale=1)

                live = gr.Textbox(label="Python Output", lines=25, max_lines=25, show_copy_button=True)

                # actions
                sdisps = [topic_bx, step_pd, threadid_bx, thread_pd, live]
                thread_pd.input(self.switch_thread, [thread_pd], None).then(
                    fn=updt_disp, inputs=None, outputs=sdisps)
                step_pd.input(self.copy_state, [step_pd], None).then(
                    fn=updt_disp, inputs=None, outputs=sdisps)
                gen_btn.click(vary_btn, gr.Number("secondary", visible=False), gen_btn).then(
                    fn=self.run_agent, inputs=[gr.Number(True, visible=False),
                                               topic_bx,
                                               stop_after], outputs=[live],
                    show_progress=True).then(
                    fn=updt_disp, inputs=None, outputs=sdisps).then(
                    vary_btn, gr.Number("primary", visible=False), gen_btn).then(
                    vary_btn, gr.Number("primary", visible=False), cont_btn)
                cont_btn.click(vary_btn, gr.Number("secondary", visible=False), cont_btn).then(
                    fn=self.run_agent, inputs=[gr.Number(False, visible=False),
                                               topic_bx,
                                               stop_after],
                    outputs=[live]).then(
                    fn=updt_disp, inputs=None, outputs=sdisps).then(
                    vary_btn, gr.Number("primary", visible=False), cont_btn)
            with gr.Tab("StateSnapShots"):
                with gr.Row():
                    refresh_btn = gr.Button("Refresh")
                snapshots = gr.Textbox(label="State Snapshots Summaries")
                refresh_btn.click(fn=get_snapshots, inputs=None, outputs=snapshots)
            with gr.Tab("LLM Python Rules"):
                python_rules = gr.Textbox(label="Python Rules", lines=25, max_lines=25, value=self.python_prompt)
        return demo

    def launch(self, share=None):
        if port := os.getenv("PORT1"):
            self.demo.launch(share=True, server_port=int(port), server_name="0.0.0.0")
        else:
            self.demo.launch(share=self.share)
