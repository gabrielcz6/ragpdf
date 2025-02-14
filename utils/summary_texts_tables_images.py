from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import GoogleGenerativeAI
import google.generativeai as genai
from PIL import Image
from io import BytesIO
import base64
from utils.images_blip import ImageCaptioner


def summary_texts_tables(texts,tables):

   table_summaries=[]
   #Prompt
   prompt_text1 = """
   Eres un asistente encargado de resumir textos.     
   Responde solo con el resumen, sin comentarios adicionales.  
   No comiences tu mensaje diciendo "Aquí tienes un resumen" ni nada similar.  
   Simplemente da el resumen tal cual es.  
   
   texto: {element}
   
   """
   prompt_text = """
   Eres un asistente encargado de resumir tablas y textos.  
   Da un resumen conciso de la tabla o el texto.  
   
   Responde solo con el resumen, sin comentarios adicionales.  
   No comiences tu mensaje diciendo "Aquí tienes un resumen" ni nada similar.  
   Simplemente da el resumen tal cual es.  
   
   Tabla o fragmento de texto: {element}
   
   """
       
   
   prompt = ChatPromptTemplate.from_template(prompt_text)
   
   # Summary chain
   model = ChatOpenAI(temperature=0.5, model="gpt-3.5-turbo")
   summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()
   
   text_summaries=summarize_chain.batch(texts,{"max_concurrency":3})
   
   try:
    tables_html=[table.metadata.text_as_html for table in tables]   
    table_summaries=summarize_chain.batch(tables_html,{"max_concurrency":3})
   except:
    pass
   
   return tables_html,text_summaries,table_summaries




def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        # Read and encode the image file to base64
        encoded_image = base64.b64encode(image_file.read())
        # Decode the base64 bytes to a string and return it
        return encoded_image.decode('utf-8')

def generar_resumenes_imagen(imagenes):
   # prompt_template = """Para contexto, la imagen es parte de un reporte de sostenibilidad de una empresa. 
   #                     Sé específico sobre los gráficos, como los diagramas de barras.
   #                     Hazlo en español.
   #                     La imagen a describir es: """  # Placeholder para la imagen
    prompt_template = """Para contexto, 
                       -tengo resumen de una imagen que es parte de un pdf y deberas describirla.
                       -Hazlo en español.
                       -debes basarte solo en el resumen
                       -resumen de la imagen a describir es: """  # Placeholder para la imagen
    # Configurar la clave de API
    genai.configure(api_key="AIzaSyAzLKUEUVqx7MYXfYilnN-msh5N9YILTUo")
    
    # Inicializar el modelo generativo
    model = genai.GenerativeModel(model_name="gemini-1.5-pro")
    
    resumenes = []
    
    # Procesar cada imagen en la lista de imágenes (cada una será un objeto de Pillow)
    blip_processor=ImageCaptioner()
    for imagen in imagenes:
        image_data = base64.b64decode(imagen)
        image_stream = BytesIO(image_data)
        image = Image.open(image_stream)
        summary_blip=blip_processor.generate_caption(image)

        # Aquí esperamos que 'imagen' sea un objeto PIL.Image
        if isinstance(image, Image.Image):
            # Si la imagen es un objeto PIL.Image, la procesamos directamente en el prompt
            response = model.generate_content([prompt_template] +[summary_blip])
        else:
            print("Error: La imagen debe ser un objeto PIL.Image")
            continue
        
        # Suponiendo que la respuesta contiene los resúmenes generados
        #input(response.text)
        print(response.text)
        resumenes.append(response.text)
        # Ajusta según el formato de la respuesta
    
    # Devolver la lista de resúmenes
    return resumenes


