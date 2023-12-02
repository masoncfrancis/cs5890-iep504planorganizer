from typing import Any
import gradio as gr
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
import fitz
from PIL import Image
import chromadb
import re
import uuid

import json


class my_app:
    def __init__(self, OPENAI_API_KEY: str = None) -> None:
        self.OPENAI_API_KEY: str = OPENAI_API_KEY
        self.chain = None
        self.N: int = 0
        self.count: int = 0

    def __call__(self, file: str) -> Any:
        self.chain = None
        self.chain = self.build_chain(file)
        return self.chain

    def chroma_client(self):
        # create a chroma client
        client = chromadb.Client()
        # create a collection
        collection = client.get_or_create_collection(name="my-collection")
        return client

    def process_file(self, file: str):
        loader = PyPDFLoader(file.name)
        documents = loader.load()
        pattern = r"/([^/]+)$"
        match = re.search(pattern, file.name)
        file_name = match.group(1)
        return documents, file_name

    def build_chain(self, file: str):
        documents, file_name = self.process_file(file)
        # Load embeddings model
        embeddings = OpenAIEmbeddings(openai_api_key=self.OPENAI_API_KEY)
        pdfsearch = Chroma.from_documents(documents, embeddings, collection_name=file_name, )
        chain = ConversationalRetrievalChain.from_llm(
            ChatOpenAI(temperature=0.0, openai_api_key=self.OPENAI_API_KEY),
            retriever=pdfsearch.as_retriever(search_kwargs={"k": 1}),
            return_source_documents=True)
        return chain


def get_accommodations(file):
    if not file:
        raise gr.Error(message='Upload a PDF')
    chain = app(file)
    file = None
    result = chain({"question": "Please give the person's name and a detailed list of all accomodations this person is to receive. Make sure to specify any special factors, assistive technologies, course subjects that accomodations apply to, and any other accomodation-related information. If this is not an IEP or 504 plan, say 'this is not an IEP or 504 plan'", "chat_history": []}, return_only_outputs=True)
    accommodations = result.get("answer", "There was an error")
    return accommodations


def render_file(file):
    doc = fitz.open(file.name)
    page = doc[app.N]
    # Render the page as a PNG image with a resolution of 300 DPI
    pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
    image = Image.frombytes('RGB', [pix.width, pix.height], pix.samples)
    return image


def render_first(file):
    doc = fitz.open(file.name)
    page = doc[0]
    # Render the page as a PNG image with a resolution of 300 DPI
    pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
    image = Image.frombytes('RGB', [pix.width, pix.height], pix.samples)
    return image, ""


app = my_app()

with gr.Blocks() as demo:
    with gr.Column():
        with gr.Row():
            accommodations_text = gr.Textbox(label="Accomodations", value="", placeholder="Accommodations will be displayed here")
            show_img = gr.Image(label='Upload PDF', tool='select').style(height=680)
    with gr.Row():
        with gr.Column():
            submit_btn = gr.Button('Get Accommodations')
        with gr.Column():
            btn = gr.UploadButton("📁 upload a PDF", file_types=[".pdf"]).style()

    btn.upload(
        fn=render_first,
        inputs=[btn],
        outputs=[show_img, accommodations_text],
    )

    submit_btn.click(
        fn=get_accommodations,
        inputs=[btn],
        outputs=accommodations_text,

    )


secretsFile = open('secrets.json')
secrets = json.load(secretsFile)
secretsFile.close()
app.OPENAI_API_KEY = secrets["openaiKey"]
    
demo.queue()
demo.launch()  
