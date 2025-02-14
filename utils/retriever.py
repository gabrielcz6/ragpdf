import os
import traceback
import uuid
import base64
from base64 import b64decode
from io import BytesIO
from IPython.display import display, Image
from langchain.schema import Document
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.storage import InMemoryByteStore
from langchain.retrievers import MultiVectorRetriever
from langchain.schema import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.messages import SystemMessage, HumanMessage
from chromadb import PersistentClient




def parse_docs(docs):
    """Split base64-encoded images and texts"""
    b64 = []
    text = []
    for doc in docs:
        try:
            b64decode(doc)
            b64.append(doc)
        except Exception:
            text.append(doc)
    return {"images": b64, "texts": text}

def build_prompt(kwargs):
    #input(kwargs)
    docs_by_type = kwargs["context"]
    user_question = kwargs["question"]

    context_text = ""

    if len(docs_by_type["texts"]) > 0:
        for text_element in docs_by_type["texts"]:
            context_text += text_element.text
    
    prompt_template = f"""
    Responda a la pregunta :
    - basándose únicamente en el siguiente Contexto, 
    - que puede incluir texto, tablas y la siguiente imagen.
    - no alucines, da respuesta solo del Contexto.
    
    Contexto: {context_text}
    Question: {user_question}
    """

    prompt_content = [{"type": "text", "text": prompt_template}]

    if len(docs_by_type["images"]) > 0:
        for image in docs_by_type["images"]:
            prompt_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image}"},
                }
            )

    return ChatPromptTemplate.from_messages([
        HumanMessage(content=prompt_content),
    ])

def display_base64_image(base64_code):
    """Decode and display base64 image"""
    image_data = base64.b64decode(base64_code)
    display(Image(data=image_data))

class DocumentRetriever:
    def __init__(self, filename: str, texts, text_summaries, tables_html, table_summaries, images, image_summaries):
        self.filename = filename
        self.embeddings = HuggingFaceEmbeddings(model_name='distiluse-base-multilingual-cased-v2')
        self.vectorstore = None
        self.store = None
        self.retriever = None
        self.id_key = "doc_id"
        self.texts = texts
        self.text_summaries = text_summaries
        self.tables_html = tables_html
        self.table_summaries = table_summaries
        self.images = images
        self.image_summaries = image_summaries
        self._setup()
        self._setup_chain()

    def _setup(self):
        #if os.path.exists(f"./dbs_chroma/{self.filename}"):

        db_path = f"./dbs_chroma/{self.filename}"
        if os.path.exists(db_path):
           self.vectorstore = Chroma(collection_name=self.filename,persist_directory=db_path, embedding_function=self.embeddings)

           try:
               #self.vectorstore.delete_collection()
      
               self.vectorstore.reset_collection()
               print("coleccion existia y se reseteo")
           except:
               print("coleccion no existe")
        else:
           self.vectorstore = Chroma(collection_name=self.filename, embedding_function=self.embeddings, persist_directory=db_path)
           
        self.store = InMemoryByteStore()
        self.retriever = MultiVectorRetriever(
            vectorstore=self.vectorstore,
            docstore=self.store,
            id_key=self.id_key,
        )

    def adding_data(self):
        try:
            doc_ids = [str(uuid.uuid4()) for _ in self.texts]
            summary_texts = [Document(page_content=summary, metadata={self.id_key: doc_ids[i]}) 
                             for i, summary in enumerate(self.text_summaries)]
            
           # for i in range(len(summary_texts)):
           #     input(summary_texts[i])

            self.retriever.vectorstore.add_documents(summary_texts)
            self.retriever.docstore.mset(list(zip(doc_ids, self.texts)))
            print(f"{len(summary_texts)} texto añadido")
            
        except Exception as e:
            print(f"Ocurrió un error: {e}")

        try:    
            table_ids = [str(uuid.uuid4()) for _ in self.tables_html]
            summary_tables = [Document(page_content=summary, metadata={self.id_key: table_ids[i]}) 
                              for i, summary in enumerate(self.table_summaries)]
            self.retriever.vectorstore.add_documents(summary_tables)
            self.retriever.docstore.mset(list(zip(table_ids, self.tables_html)))
            print("tablas añadido")
        except Exception as e:
            print(f"Ocurrió un error: {e}")    

        try:
            # Verifica que self.images y self.image_summaries no estén vacíos
            if not self.images or not self.image_summaries:
                raise ValueError("self.images o self.image_summaries están vacíos")
            
            # Genera IDs únicos para las imágenes
            img_ids = [str(uuid.uuid4()) for _ in self.images]
            #print(f"IDs generados: {img_ids}")
        
            # Crea documentos para el vectorstore
            summary_img = [
                Document(page_content=summary, metadata={self.id_key: img_ids[i]})
                for i, summary in enumerate(self.image_summaries)
            ]
           # print(f"Documentos creados: {summary_img}")


            print("Antes de agregar documentos...")
            input(type(summary_img))
            self.retriever.vectorstore.add_documents(summary_img[0:50])
            print("Documentos agregados al vectorstore ✅")
        
            # Almacena imágenes en el docstore
            print("Antes de almacenar imágenes en el docstore...")
            self.retriever.docstore.mset(list(zip(img_ids, self.images)))
            print("Imágenes almacenadas en el docstore ✅")
        
        except Exception as e:
            print(f"Ocurrió un error: {e}")
            print(traceback.format_exc())  # Muestra el traceback completo

        print("Datos añadidos a la BD Chroma")


    def _setup_chain(self):
        self.chain = (
            {
                "context": self.retriever | RunnableLambda(parse_docs),
                "question": RunnablePassthrough(),
            }
            | RunnableLambda(build_prompt)
            | ChatOpenAI(model="gpt-4o-mini")
            | StrOutputParser()
        )
        
        self.chain_with_sources = (
            {
                "context": self.retriever | RunnableLambda(parse_docs),
                "question": RunnablePassthrough(),
            }
            | RunnablePassthrough().assign(
                response=(
                    RunnableLambda(build_prompt)
                    | ChatOpenAI(model="gpt-4o-mini")
                    | StrOutputParser()
                )
            )
        )
    def base64_to_image(self,base64_code):
             """Decode base64 and return a PIL Image object"""
             image_data = base64.b64decode(base64_code)
             return Image.open(BytesIO(image_data))
    
    def invoke_chain(self, question):
             pages = []
             image_list = []
             answer=[]

             response = self.chain_with_sources.invoke(question)
            # for clave, valor in response["context"].items():
            #    input(f"Clave: {clave}, Tipo de valor: {type(valor).__name__}")
             # 📌 Obtener la respuesta
             answer = response['response']             
             #input(f"Respuesta : {answer}")
             #input(f"response : {response}")

             try:
              for text in response['context']['texts']:
                 print(text.text)
                 pages.append(text.metadata.page_number)  # Agregar el número de página a la lista
                 print("Page number:", text.metadata.page_number)
                 print("\n" + "-" * 50 + "\n")
             except:
                 print("no hay textos") 
             # 📌 Obtener la lista de imágenes
             
             try:
                image_list= response['context']['images']
             except:
                 print("imagenes no encontradas")
         
             # 📌 Retornar los valores
             return answer, pages, image_list
