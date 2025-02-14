import pickle
from utils.retriever import DocumentRetriever
from dotenv import load_dotenv
import os

class RetrieverLoader:
    def __init__(self, filename):
        load_dotenv()
        self.filename = filename
        self.retriever = None
    
    def load_data(self):
        """Carga los datos desde el archivo pickle y crea el objeto DocumentRetriever."""
        pkl_path = f'./pkl/{self.filename}.pkl'
        if not os.path.exists(pkl_path):
            raise FileNotFoundError(f"El archivo {pkl_path} no existe.")
        
        with open(pkl_path, 'rb') as file:
            loaded_data = pickle.load(file)

        self.retriever = DocumentRetriever(
            self.filename,
            loaded_data['texts'],
            loaded_data['text_summaries'],
            loaded_data['tables_html'],
            loaded_data['table_summaries'],
            loaded_data['images'],
            loaded_data['image_summaries']
        )
    
    def get_retriever(self) :
        """Devuelve el objeto DocumentRetriever después de cargar los datos y agrega los datos."""

        self.retriever.adding_data()
        return self.retriever
