from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route
import asyncio
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.prompts import PromptTemplate
import os


async def retrive_relevent_doc(request):
    a = await request.form()
    name = a.get('name')
    query = a.get('query')
    response_q = asyncio.Queue()
    await request.app.model_queue.put((query, name, response_q))
    (context, reference) = await response_q.get()

    return JSONResponse({'context': context, 'reference': reference})


async def server_loop(q):
    model_name = "BAAI/bge-large-en-v1.5"

    # Create a dictionary with model configuration options, specifying to use the CPU for computations
    model_kwargs = {'device': 'cuda'}
    # set True to compute cosine similarity
    encode_kwargs = {'normalize_embeddings': True}
    bge_embeddings = HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
        cache_folder="embeddings/cache"
    )

    persist_directory = "thermal"
    vector_db = Chroma(persist_directory=persist_directory,
                       embedding_function=bge_embeddings)

    template = '''A chat between a curious user and an artificial intelligence assistant.

    USER:
    DOCUMENT: {context}

    Question: {question}

    Answer the users QUESTION using the DOCUMENT text above.
    Keep your answer ground in the facts of the DOCUMENT.
    If the DOCUMENT doesn’t contain the answer the Question, just say you cannot provide an Answer.
    If the answer is contained in the context, also report the Page Number.
    '''
    prompt = PromptTemplate(
        input_variables=["context", "question"], template=template)

    while True:
        (query, name, response_q) = await q.get()
        if name != None:
            retriever = vector_db.as_retriever(search_type="mmr", search_kwargs={
                                               "k": 10, 'filter': {'name': name}})
        else:
            retriever = vector_db.as_retriever(
                search_type="mmr", search_kwargs={"k": 10})
        docs_result = await retriever.aget_relevant_documents(query)
        docs_result.sort(key=lambda x: (
            x.metadata['name'], x.metadata['page']))
        docs_text = "\n".join([f"Document Name:{doc.metadata['name']}"
                               f"\nPage Number:{doc.metadata['page'] + 1}"
                               f"\nContent:{doc.page_content}" for doc in docs_result])
        reference = [{'docName': doc.metadata['name'], 'pageNumber': doc.metadata['page'] + 1, 'url': doc.metadata['source'].replace(os.sep, '/')}
                     for doc in docs_result]
        context = prompt.format(question=query, context=docs_text)
        await response_q.put((context, reference))


app = Starlette(
    routes=[
        Route("/retrive-doc/", retrive_relevent_doc, methods=["POST"]),
    ],
)


@app.on_event("startup")
async def startup_event():
    q = asyncio.Queue()
    app.model_queue = q
    asyncio.create_task(server_loop(q))
