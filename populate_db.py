from dotenv import load_dotenv
from utils.utils_populatedb import *
from utils.summary_texts_tables_images import *
load_dotenv()





filename="reporte_nestle_2023.pdf"
output_path="./content/"
file_path=output_path+filename

print("creando chunks")
chunks=create_chunks(file_path)
print("separando chunks")
images,tables,texts=get_images_tables_text(chunks)
print("creando imagenes")
images=get_resized_images_base64(chunks)
print("creando resumen textos,tablas")
tables_html,text_summaries,table_summaries=summary_texts_tables(texts,tables)
print("creando resumen imagenes")
#input(len(images))
image_summaries = generar_resumenes_imagen(images)
print("guardando pkl")
create_pkl(texts,text_summaries,tables_html,table_summaries,images,image_summaries,filename)