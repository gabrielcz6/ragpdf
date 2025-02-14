import pickle
from utils.retriever import DocumentRetriever
from dotenv import load_dotenv
load_dotenv()


filename="reporte_nestle_2023.pdf"
# Cargar los datos desde el archivo .pkl
with open(f'./pkl/{filename}.pkl', 'rb') as file:
    loaded_data = pickle.load(file)


# Acceder a los datos cargados a través del diccionario
texts = loaded_data['texts']
text_summaries = loaded_data['text_summaries']
tables_html = loaded_data['tables_html']
table_summaries = loaded_data['table_summaries']
images = loaded_data['images']  # Aquí cargamos la imagen directamente como objeto
image_summaries = loaded_data['image_summaries']


retriever=DocumentRetriever(filename,texts,text_summaries,tables_html,table_summaries,images,image_summaries)
retriever.adding_data()
answer, pages, image_list = retriever.invoke_chain("que paso con la contaminacion en nestle")


input(answer)
input(f"paginas de referencia: {pages}")
input(image_list)


import base64
from io import BytesIO
from PIL import Image


for idx, img_base64 in enumerate(image_list):
    # Decodificar base64
    image_data = base64.b64decode(img_base64)
    image = Image.open(BytesIO(image_data))

    # Mostrar la imagen
    image.show(title=f"Imagen {idx+1}")
    input("cats")