from unstructured.partition.pdf import partition_pdf
from PIL import Image
import base64
import io
import pickle



def create_chunks(file_path):
    #creando los chunks

    chunks = partition_pdf(
    filename=file_path,
    #tablas
    infer_table_structure=True,         #Extraer tablas
    strategy='hi_res',                    #tipo de inferencia para tablas, estrategia alta resolucion
    #imagenes
    extract_image_block_types=["Image","Table"],  #Agregamos "Table" para extraer imagenes de las tablas ["Image","Table"]
    extract_image_block_to_payload=True,   # Si es True, extrae base64 para uso ed api
    #estrategia de fragmentacion:
    chunking_strategy="by_title",         #puede ser "basic" tambien
    max_characters=10000,                 #por default viene con 500
    combine_text_under_n_chars=2000,      # el default es 0
    new_after_n_chars=6000,
    )
    return chunks

def get_images_tables_text(chunks):

    tables=[]
    texts=[]
    
    for chunk in chunks:
        if "Table" in str(type(chunk)):
            tables.append(chunk)
    
        if "CompositeElement" in str(type(chunk)):
            texts.append(chunk)
    
    # Get the images from the CompositeElement objects
    def get_images_base64(chunks):
        images_b64 = []
        for chunk in chunks:
            if "CompositeElement" in str(type(chunk)):
                chunk_els = chunk.metadata.orig_elements
                for el in chunk_els:
                    if "Image" in str(type(el)):
                        images_b64.append(el.metadata.image_base64)
        return images_b64
    
    images = get_images_base64(chunks)

    return images,tables,texts

def get_resized_images_base64(chunks, width=500, height=500):
    images_b64_resized = []
    
    for chunk in chunks:
        if "CompositeElement" in str(type(chunk)):
            chunk_els = chunk.metadata.orig_elements
            for el in chunk_els:
                if "Image" in str(type(el)):
                    base64_str = el.metadata.image_base64
                    
                    # Decodificar la imagen base64
                    img_data = base64.b64decode(base64_str)
                    img = Image.open(io.BytesIO(img_data))
                    
                    # Redimensionar la imagen a las dimensiones deseadas
                    img = img.resize((width, height), Image.Resampling.LANCZOS)
                    
                    # Convertir la imagen a base64 en formato JPEG
                    buffered = io.BytesIO()
                    img.save(buffered, format="JPEG")
                    new_base64_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
                    
                    # Agregar la imagen redimensionada a la lista
                    images_b64_resized.append(new_base64_str)
    
    return images_b64_resized

# Crear un diccionario con todos los datos

def create_pkl(texts,text_summaries,tables_html,table_summaries,images,image_summaries,filename):
     data = {
         "text_summaries": text_summaries,
         "texts": texts,
         "tables_html": tables_html,
         "table_summaries": table_summaries,
         "images": images , # Aquí guardamos la imagen directamente como objeto
         "image_summaries" : image_summaries
     }
     
     # Guardar todos los datos en un archivo .pkl
     with open(f'./pkl/{filename}.pkl', 'wb') as file:
         pickle.dump(data, file)
