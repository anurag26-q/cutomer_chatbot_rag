import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import logging
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from retriever.retrieval import Retriever
from utils.model_loader import ModelLoader
from prompt_library.prompt import PROMPT_TEMPLATES

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Allow CORS (optional for frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

load_dotenv()

retriever_obj = Retriever() 
model_loader = ModelLoader() 

def invoke_chain(query: str): 
    """
    Invoke the LangChain pipeline with error handling and fallback.
    """
    retriever = retriever_obj.load_retriever()
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATES["product_bot"])
    llm = model_loader.load_llm()

    logger.info(f"Processing query: {query}")
    
    # Try to retrieve documents
    context = ""
    try:
        retrieved_docs = retriever.invoke(query)
        logger.info(f"Retrieved {len(retrieved_docs)} documents")
        context = "\n".join([doc.page_content for doc in retrieved_docs])
    except Exception as e:
        logger.error(f"Error retrieving documents: {str(e)}")
        # Fallback: Use LLM without retrieved documents
        logger.info("Falling back to LLM without context")
        context = "No relevant documents found. Responding based on general knowledge."

    # Define the chain with or without context
    chain = (
        {"context": lambda _: context, "question": RunnablePassthrough()}
        | prompt 
        | llm 
        | StrOutputParser() 
    ) 

    try:
        output = chain.invoke(query)
        logger.info(f"Generated response: {output}")
        return output
    except Exception as e:
        logger.error(f"Error in chain invocation: {str(e)}")
        return "I'm sorry, an error occurred while processing your request. Please try again or rephrase your query."

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """
    Render the chat interface.
    """
    return templates.TemplateResponse("chat.html", {"request": request})

@app.post("/get", response_class=HTMLResponse)
async def chat(msg: str = Form(...)):
    result = invoke_chain(msg)
    return result

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)