from dotenv import load_dotenv
import os
import json
from functools import partial
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uuid
from datetime import datetime, timezone

from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_community.vectorstores import SKLearnVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.pydantic_v1 import BaseModel, Field

from langgraph.prebuilt import ToolExecutor, ToolInvocation
from langgraph.graph.graph import START, END
from langgraph.graph import MessageGraph
from langgraph.checkpoint.sqlite import SqliteSaver

load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')

db = SQLDatabase.from_uri("sqlite:///ElectronicsStore.db")
db.get_table_names()

model = ChatOpenAI(temperature=0, streaming=True, model="gpt-4-turbo-preview")




@tool
def get_product_info(id: int):
    """Look up produtos info given their ID. ALWAYS make sure you have the sale ID before invoking this."""
    return db.run(f"SELECT * FROM Produtos WHERE ID = {id};")

sale_prompt = """Your job is to help a user to find products"""

def get_sale_messages(messages):
    return [SystemMessage(content=sale_prompt)] + messages

sale_chain = get_sale_messages | model.bind_tools([get_product_info])




products = db._execute("select * from Produtos")

products_retriever_by_name = SKLearnVectorStore.from_texts(
    [a['NomeProduto'] for a in products],
    OpenAIEmbeddings(), 
    metadatas=products
).as_retriever()

products_retriever_by_brand = SKLearnVectorStore.from_texts(
    [a['Marca'] for a in products],
    OpenAIEmbeddings(), 
    metadatas=products
).as_retriever()

products_retriever_by_category = SKLearnVectorStore.from_texts(
    [a['Categoria'] for a in products],
    OpenAIEmbeddings(), 
    metadatas=products
).as_retriever()



@tool
def get_products_by_brand(brand):
    """Get products by an brand (or similar brands)."""
    docs = products_retriever_by_brand.get_relevant_documents(brand)
    products_ids = ", ".join([str(d.metadata['ID']) for d in docs])
    return db.run(f"SELECT NomeProduto FROM Produtos WHERE ID in ({products_ids});", include_columns=True)


@tool
def get_products_by_name(name):
    """Get products by an brand (or similar brands)."""
    docs = products_retriever_by_name.get_relevant_documents(name)
    products_ids = ", ".join([str(d.metadata['ID']) for d in docs])
    return db.run(f"SELECT NomeProduto FROM Produtos WHERE ID in ({products_ids});", include_columns=True)

@tool
def get_products_by_category(category):
    """Get products by an category."""
    docs = products_retriever_by_category.get_relevant_documents(category)
    products_ids = ", ".join([str(d.metadata['ID']) for d in docs])
    return db.run(f"SELECT NomeProduto FROM Produtos WHERE ID in ({products_ids});", include_columns=True)


product_system_message = """Your job is to help a sale find any products they are looking for. 

You only have certain tools you can use. If a sale asks you to look something up that you don't know how, politely tell them what you can help with.

When looking up products, sometimes the product will not be found. In that case, the tools will return information \
on simliar products. This is intentional, it is not the tool messing up."""
def get_products_messages(messages):
    return [SystemMessage(content=product_system_message)] + messages

products_chain = get_products_messages | model.bind_tools([get_products_by_brand, get_products_by_name, get_products_by_category])


msgs = [HumanMessage(content="hi! can you help me find any products by phillips?")]
products_chain.invoke(msgs)





class Router(BaseModel):
    """Call this if you are able to route the user to the appropriate representative."""
    choice: str = Field(description="should be one of: product, recomendation")

system_message = """Your job is to help as a sale service representative for a hardware store.

You should interact politely with sales to try to figure out how you can help. You can help in a few ways:

- Finding product information: if a sale wants to know about a product in the database. Call the router with `sale`
- Finding product information: if a sale wants recomendation for other products. Call the router with `recomendation`

If the user is asking or wants to ask about a certain product, send them to that route.
If the user is asking or wants to have some product recomendation, send them to that route.
Otherwise, respond."""
def get_messages(messages):
    return [SystemMessage(content=system_message)] + messages

chain = get_messages | model.bind_tools([Router])

def add_name(message, name):
    _dict = message.dict()
    _dict["name"] = name
    return AIMessage(**_dict)

def _get_last_ai_message(messages):
    for m in messages[::-1]:
        if isinstance(m, AIMessage):
            return m
    return None


def _is_tool_call(msg):
    return hasattr(msg, "additional_kwargs") and 'tool_calls' in msg.additional_kwargs


def _route(messages):
    last_message = messages[-1]
    if isinstance(last_message, AIMessage):
        if not _is_tool_call(last_message):
            return END
        else:
            if last_message.name == "general":
                tool_calls = last_message.additional_kwargs['tool_calls']
                if len(tool_calls) > 1:
                    raise ValueError
                tool_call = tool_calls[0]
                return json.loads(tool_call['function']['arguments'])['choice']
            else:
                return "tools"
    last_m = _get_last_ai_message(messages)
    if last_m is None:
        return "general"
    if last_m.name == "recomendation":
        return "recomendation"
    elif last_m.name == "sale":
        return "sale"
    else:
        return "general"
    


tools = [get_products_by_brand, get_products_by_name, get_products_by_category]
tool_executor = ToolExecutor(tools)

def _filter_out_routes(messages):
    ms = []
    for m in messages:
        if _is_tool_call(m):
            if m.name == "general":
                continue
        ms.append(m)
    return ms

general_node = _filter_out_routes | chain | partial(add_name, name="general")
recomendation_node = _filter_out_routes | sale_chain | partial(add_name, name="recomendation")
sale_node = _filter_out_routes | products_chain | partial(add_name, name="sale")




async def call_tool(messages):
    actions = []
    # Based on the continue condition
    # we know the last message involves a function call
    last_message = messages[-1]
    for tool_call in last_message.additional_kwargs["tool_calls"]:
        function = tool_call["function"]
        function_name = function["name"]
        _tool_input = json.loads(function["arguments"] or "{}")
        # We construct an ToolInvocation from the function_call
        actions.append(
            ToolInvocation(
                tool=function_name,
                tool_input=_tool_input,
            )
        )
    # We call the tool_executor and get back a response
    responses = await tool_executor.abatch(actions)
    # We use the response to create a ToolMessage
    tool_messages = [
        ToolMessage(
            tool_call_id=tool_call["id"],
            content=str(response),
            additional_kwargs={"name": tool_call["function"]["name"]},
        )
        for tool_call, response in zip(
            last_message.additional_kwargs["tool_calls"], responses
        )
    ]
    return tool_messages



memory = SqliteSaver.from_conn_string(":memory:")
graph = MessageGraph()
nodes = {"general": "general", "recomendation": "recomendation", END: END, "tools": "tools", "sale": "sale"}
# Define a new graph
workflow = MessageGraph()
workflow.add_node("general", general_node)
workflow.add_node("recomendation", recomendation_node)
workflow.add_node("sale", sale_node)
workflow.add_node("tools", call_tool)
workflow.add_conditional_edges("general", _route, nodes)
workflow.add_conditional_edges("tools", _route, nodes)
workflow.add_conditional_edges("recomendation", _route, nodes)
workflow.add_conditional_edges("sale", _route, nodes)
workflow.set_conditional_entry_point(_route, nodes)
graph = workflow.compile()





















#####################################################
##################  API SERVER  #####################
#####################################################

from fastapi import FastAPI
from pydantic import BaseModel
import uuid
from datetime import datetime, timezone

app = FastAPI()

# In-memory storage for conversation histories
conversations = {}

class UserInput(BaseModel):
    user_input: str
    session_id: str  # Field to track the conversation

@app.post("/chat/")
async def chat(user_input: UserInput):
    if user_input.user_input in {'q', 'Q'}:
        return {"message": "Byebye"}
    
    # Retrieve or start a new conversation history
    session_id = user_input.session_id
    if session_id not in conversations:
        conversations[session_id] = []  # Start a new conversation history if not exists
    history = conversations[session_id]  # Retrieve the existing conversation history

    conversations[session_id] = []

    # Add the new message to the history
    history.append(HumanMessage(content=user_input.user_input))
    
    responses = []
    async for output in graph.astream(history):
        if END in output or START in output:
            continue
        for key, value in output.items():
            responses.append({"node": key, "response": value})
    
    # Update the conversation history in the in-memory storage
    conversations[session_id] = history
    final_content = responses[-1]["response"].content if responses else ""

    # Prepare and return the simplified response
    response_data = {
        "user_input": user_input.user_input,
        "session_id": user_input.session_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),  # Generate a current UTC timestamp
        "content": final_content  # The extracted content
    }
    
    return response_data

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
