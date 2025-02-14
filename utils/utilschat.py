





def responderquery(query,retriever):

    querynuevo = query[-1]['content']

    answer, pages, image_list = retriever.invoke_chain(querynuevo)
    answer_completo = f"Respuesta:\n\n{answer}\n\nPáginas:\n{pages}"

    input(f"answer : \n\n{answer}")    
    input(f"pages : \n\n{pages}")    
    input(f"image_list : \n\n{image_list}")
    input("fin")    


    return [answer_completo,image_list]


