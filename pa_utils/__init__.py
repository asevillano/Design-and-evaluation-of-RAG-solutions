import tiktoken
import re
import json
encoding = tiktoken.get_encoding("cl100k_base")

# Calculate number of tokens of a text
def token_len(input: str) -> int:
    # Returns number of tokens for the input. Only for models > gpt-3.5 supported as we use 'cl100k_base' encoding
    return len(encoding.encode(input))

# Cut a text to a maximum number of tokens
def cut_max_tokens(text):
    tokens = encoding.encode(text)
    max_tokens = 8191 #-1536
    if len(tokens) > max_tokens:
        print(f'\t*** CUT TOKENS, tokens: {len(tokens)}') #, text: [{text}]')
        return encoding.decode(tokens[:max_tokens])
    else:
        return text

# Send a call to the model deployed on Azure OpenAI
def call_aoai(aoai_client, aoai_model_name, system_prompt, user_prompt, temperature, max_tokens):
    messages = [{'role' : 'system', 'content' : system_prompt},
                {'role' : 'user', 'content' : user_prompt}]
    try:
        response = aoai_client.chat.completions.create(
            model=aoai_model_name,
            messages=messages,
            temperature=temperature, #0.5,
            max_tokens=max_tokens #4096 #800,
        )
        json_response = json.loads(response.model_dump_json())
        response = json_response['choices'][0]['message']['content']
    except Exception as ex:
        print(ex)
        response = None
    
    return response

# Extract data between two delimiters
def extract_text(texto, start_delimiter, end_delimiter=''):
    if end_delimiter != '':
        # This regular expression searches for any text between the delimiters.
        patron = re.escape(start_delimiter) + '(.*?)' + re.escape(end_delimiter)
        resultado = re.search(patron, texto, re.DOTALL)
        if resultado:
            return resultado.group(1)
        else:
            return None
    else:
        # Find the position of the delimiter in the text
        delimiter_index = texto.find(start_delimiter)
        if delimiter_index != -1:
            # Extract the text from the delimiter to the end
            return texto[delimiter_index + len(start_delimiter):]
        else:
            return None

# Semantic Hybrid Search with filter (optional)
# Create first the index and upload documents with 'create_index_and_index_documents.ipynb'
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery, VectorFilterMode, QueryType, QueryCaptionType, QueryAnswerType
def semantic_hybrid_search_with_filter(search_client: SearchClient, query: str, aoai_embedding_client, embedding_model_name, embedding_fields, max_docs, select_fields, query_language, filter=''):
    # Semantic Hybrid Search
    #print(f'query semantic_hybrid_search_with_filter: {query}, filter: {filter}')

    embedding = aoai_embedding_client.embeddings.create(input=query, model=embedding_model_name).data[0].embedding
    vector_query = VectorizedQuery(vector=embedding, k_nearest_neighbors=max_docs, fields=embedding_fields) #, exhaustive=True)

    if filter == '':
        results = search_client.search(  
            search_text=query,  
            vector_queries=[vector_query],
            select=select_fields,
            query_type=QueryType.SEMANTIC,
            semantic_configuration_name='semantic-config',
            query_caption=QueryCaptionType.EXTRACTIVE,
            query_answer=QueryAnswerType.EXTRACTIVE,
            include_total_count=True,
            top=max_docs,
            query_language=query_language
        )
    else:
        results = search_client.search(  
            search_text=query,  
            vector_queries=[vector_query],
            select=select_fields,
            query_type=QueryType.SEMANTIC,
            semantic_configuration_name='semantic-config',
            query_caption=QueryCaptionType.EXTRACTIVE,
            query_answer=QueryAnswerType.EXTRACTIVE,
            include_total_count=True,
            top=max_docs,
            query_language=query_language,
            vector_filter_mode=VectorFilterMode.PRE_FILTER,
            filter=filter, # Sample: filter="customerSegment/any(s: s eq 'CP')",
            #search_fields=["docTitle", "sectionContent", "docSummary", "docKeywords"], #, QuestionsText"],
        )
    
    return results

# Calculate the confidence and generate the 'answer' from the content
def calculate_rank(aoai_rerank_client, rerank_model_name, text, question):
    # Include every relevant detail from the text to ensure all pertinent information is retained.
    system_prompt = """You are an assistant that returns content relevant to a search query from an telecommunications company agent serving customers.
    Return the content needed to understand the context of the answer and only what is relevant to the search query in a field called "answer". Include every relevant detail from the text to ensure all pertinent information is retained.
    In your response, include a percentage between 0 and 100 in a "confidence" field indicating how confident you are the answer provided includes content relevant to the search query.
    If the user asked a question, your confidence score should be based on how confident you are that it answered the question.
    Answer ONLY from the information listed in the text below.
    Respond in JSON format as follows, for instance:
    {
        "confidence": 100,
        "answer": "Our company offers a range of telecommunication products for home customers."
    }
    """

    user_prompt = """Search Query: """ + question + """
    Text:  """ + text + """
    """
    print(f'USER PROMPT CALCULATE RANK: {user_prompt}')
    response = call_aoai(aoai_rerank_client, rerank_model_name, system_prompt, user_prompt, 0.0, 800)

    if response != None:
        confidence = extract_text(response, 'confidence": ', ',')
        answer = extract_text(response, 'answer": ', '\n}')
        if answer == None or answer == '': answer = ''
        if confidence == None or confidence == '': confidence = 0
    else:
        confidence = 0
        answer = ''
    
    return confidence, answer

# Re-ranker: calculate in parallel the percentage of confidence and the answer comparing with the query
import concurrent.futures
def get_filtered_chunks(aoai_rerank_client, rerank_model_name, results, query, max_docs):
    i=1
    chunks = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_docs) as executor:
        futures=[]
        for result in results:
            chunk = f"{result['title']}. {result['content']}"
            #print(f'chunk: [{chunk}]')
            futures.append(executor.submit(calculate_rank, aoai_rerank_client, rerank_model_name, chunk, query))

        for future in concurrent.futures.as_completed(futures):
            confidence, answer = future.result()
            print(f'\tconfidence: {confidence}, \n\tanswer: {answer}')
            if int(confidence) >= 90:
                result['answer'] = answer
                chunks.append(result)

            i+=1
            if i == max_docs: break

    return chunks
        

# Generate the answer with the chunks filtered by the re-ranker
def generate_answer(aoai_answer_client, aoai_answer_model_name, texts, question, field='content'):
    
    system_prompt = """
        You are an assistant for Telefónica's agents (not for an end customer). You are replying to questions with information contained in a specific knowledge base provided. Both questions and knowledge base are in Spanish.
        To carry out this task, follow these steps:
        1. It's very important that you read carefully all the Document ID, Titles and Sections of the knowledge base provided.          
        2. Analyse the user Question provided.
        3. Reply to the question at step 2 using exclusively the information listed in step 1. In addition, when answering the question, follow these instructions:
            - The response should be as explanatory and orderly as possible, as it will contain the steps to carry out certain operations.
            - If more information or clarification is needed, it will ask the agent a question to disambiguate and give the correct information.
            - You must refrain from making up any information and ensure not to respond with data not explicitly mentioned in the provided knowledge base.
            - If your are not confident that the context is answering the query, please answer "No tengo suficiente información par responder a la pregunta, quizas si la reformulalas, podré encotnrar la respuesta" 
            - Do not refer to any telephone channel at the end of the answer, remember that you are talking with an agent now, not with an end customer.
            - Avoid any kind of profanity.
            - Avoid expressions of regret, or admission of errors in your responses.
            - Ensure that responses are concise and to the point, incorporating only the necessary information.
            - Use assertive statements.
            - It is better to give less but more useful information than a lot of information about the asked question.
            - Make a list of bullet points things whenever it is possible, so you dont give extra information.
            - Under no circumstances should references to websites or the provision of phone numbers be included in the responses.
            - It is essential to avoid any mention that could lead agents to seek information outside of the provided documents or suggest direct contact through external services to Telefónica.
            - Focus solely on the content of the supplied documents, without facilitating external points of contact.
            - Each response must rigorously adhere to the sequence of information exactly as it is laid out in the documents.
            - It is imperative to maintain this order with utmost precision, as any deviation or rearrangement of the information could lead to inaccuracies and misinterpretations. 
            - Under no circumstances is altering the order of information acceptable, as doing so compromises the accuracy and reliability of the provided guidance. This requirement applies to all responses, not only for procedures or lists but for every piece of information shared. Each answer must reflect the order and structure of the information in the documents without alterations, even if the question does not specify a procedure or list.
            - Do not be verbose and add only the necessary information in the response.
            - Whenever you give any price in the response, specify if that price is with IVA (VAT) or without IVA. This information should be searched for in the document. If the document does not specify whether the price includes IVA, it must be explicitly stated that the inclusion of IVA is not clear.
            - When providing responses, it is essential to use language and terminology that directly mirror those found within the source documents. The use of exact words and terms from the documents is crucial for preserving the fidelity of the information and facilitating clear, unambiguous communication with the agents.
            - Do not include any text between [] or <<>> in your search terms.
            - Do not exceed 1200 characters. 
        4. After generating the answer, identify the document ID or IDs used in the response by referring to the documents from step one. The answer may come from one or more document IDs.
        5. Integrate at the start and at the end of EVERY sentence or paragraph of the response the identified document ID or IDs, using this format: ((ID)). Important: Ensure that the selected document ID or IDs belong to the provided sections. Do not invent or use other document IDs that are not part of the given sections.
            Here two examples on how to integrate the document ID:
                - Response before integrating document ID: Las líneas móviles extras incluidas en Fusión se facturan de manera unificada en la factura Fusión. Sin embargo, los módulos y opciones de TV Satélite se facturan fuera de Movistar.
                - Response after integrating document ID: ((709094)) Las líneas móviles extras incluidas en Fusión se facturan de manera unificada en la factura Fusión. Sin embargo, los módulos y opciones de TV Satélite se facturan fuera de Movistar. ((709094))
                - Response before integrating document ID: Para acceder al Configurador de las Islas, hay dos opciones: a través del banner NBA o pinchando directamente en Configurador. Una vez dentro, se debe introducir la dirección del cliente para hacer la consulta de cobertura y zonificación. El configurador mostrará las ofertas adecuadas a la cobertura y zonificación de la dirección del cliente.
                - Response after integrating document ID: ((566754)) Para acceder al Configurador de las Islas, hay dos opciones: a través del banner NBA o pinchando directamente en Configurador. Una vez dentro, se debe introducir la dirección del cliente para hacer la consulta de cobertura y zonificación. ((566754)) ((896732)) El configurador mostrará las ofertas adecuadas a la cobertura y zonificación de la dirección del cliente. ((896732))
    """
    
    user_prompt = """Knowledge base:\n"""
    for text in texts:
        user_prompt = user_prompt + f'Document ID: {text['id']}. Title: {text['title']}. Section: {text[field]}\n'
    user_prompt = user_prompt + "\nQuestion: " + question + "\nFinal Response:"
    
    print(f'USER PROMPT: {user_prompt}')

    return call_aoai(aoai_answer_client, aoai_answer_model_name, system_prompt, user_prompt, 0.0, 800)

# Load in an array the content of every html file in a directory
import os
def load_files(input_dir, ext):
    print(f'Loading files in {input_dir}...')
    files_content = []
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(ext):
            file_path = os.path.join(input_dir, filename)
            with open(file_path, 'r', encoding="utf-8") as f:
                row = {"title": filename.replace('_', ' '), "content": f.read()}
                files_content.append(row)
    return files_content
