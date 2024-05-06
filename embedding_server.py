from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route
import asyncio
from langchain_community.vectorstores.chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
import aspose.words as aw
from spire.pdf.common import *
from spire.pdf import *
import os
import pandas as pd
import numpy as np
from langchain_community.docstore.document import Document
from starlette.background import BackgroundTask
from pathlib import Path
import requests
from chromadb import HttpClient


async def embedd_doc(request):
    a = await request.form()
    name = a['name']
    path = f'./doc/{a["file"].filename}'
    source = a['source']
    content = await a['file'].read()
    dir = a.get('directory')
    if (dir == None):
        dir = 'other'
    # await a['file'].write(content)
    fs = open(path, 'wb')
    fs.write(content)
    fs.close()
    response_q = asyncio.Queue()
    await request.app.model_queue.put((path, source, name, dir, 'embedd', response_q))
    task = await response_q.get()
    requests.post(
        'http://127.0.0.1:8000/api/chatbot/embedding-status-change/', data=({'name': name, 'status': 'PROCESSING'}))
    return JSONResponse({'message': 'added to queue'}, background=task)


async def delete_doc(request):
    a = await request.form()
    name = a['name']
    response_q = asyncio.Queue()
    await request.app.model_queue.put(('', '', name, '', 'delete', response_q))

    task = await response_q.get()
    return JSONResponse({'message': 'added to queue'}, background=task)


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
    client = HttpClient(host='localhost', port=1237)
    vector_db = Chroma(client=client,
                       embedding_function=bge_embeddings)

    while True:
        (path, source, name, dir, function, response_q) = await q.get()
        if function == 'embedd':

            task = BackgroundTask(add_doc_to_vectorDB, path=path, source=source,
                                  name=name, dir=dir, vector_db=vector_db)
            await response_q.put(task)
        elif function == 'delete':
            task = BackgroundTask(remove_doc_from_vectorDB,
                                  name=name, vector_db=vector_db)
            await response_q.put(task)


async def add_doc_to_vectorDB(path, source, name, dir, vector_db):
    await vector_db.aadd_documents(await embedd_documents(path, source, name, dir))
    requests.post(
        'http://127.0.0.1:8000/api/chatbot/embedding-status-change/', data=({'name': name, 'status': 'COMPLETED'}))
    Path(path).unlink(missing_ok=True)


async def remove_doc_from_vectorDB(name, vector_db):
    ids = vector_db.get(where={"name": name})['ids']
    await vector_db.adelete(ids=ids)

app = Starlette(
    routes=[
        Route("/embedd-doc/", embedd_doc, methods=["POST"]),
        Route("/delete-doc/", delete_doc, methods=["DELETE"]),
    ],
)


@app.on_event("startup")
async def startup_event():
    Path('./doc').mkdir(parents=True, exist_ok=True)
    q = asyncio.Queue()
    app.model_queue = q
    asyncio.create_task(server_loop(q))


async def embedd_documents(file_path, source, name, dir):

    print('Loading document...')
    file_loader = PyPDFLoader(file_path)
    documents = await file_loader.aload()

    for doc in documents:
        doc.page_content = doc.page_content.replace("\n", "")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    doc_list = []

    print('Extraction tables...')
    doc_list.extend(await give_prompt_from_document(file_path=file_path))

    texts.extend(doc_list[0])

    for text in texts:
        text.metadata = {
            'page': text.metadata['page'], 'source': source, 'name': name, 'directory': dir}

    print('Embedding Documents...')
    return texts

    # api call to django server


async def extract_dataframe_list(file_path, temp_dir):
    page = []
    try:
        file_extention = file_path[file_path.rindex('.'):]
        df_list = []
        if (file_extention in [".xlsx", ".csv", ".html", ".pdf"]):
            if (file_extention == ".html"):
                df_list = pd.read_html(file_path)
            elif (file_extention == ".csv"):
                df_list.append(pd.read_csv(file_path))
            elif (file_extention == ".xlsx"):
                df_list.append(pd.read_excel(file_path))
            elif (file_extention == ".pdf"):
                pdf = PdfDocument()
                pdf.LoadFromFile(file_path)
                total_pages = pdf.Pages.Count
                pdf.Split('/temp/temp_file_{0}.pdf')
                for i in range(total_pages):
                    temp_html_name = f'/temp/temp_file_{i}.html'
                    temp_pdf_name = f'/temp/temp_file_{i}.pdf'

                    doc = aw.Document(temp_pdf_name)
                    os.remove(temp_pdf_name)
                    doc.save(temp_html_name)

                    try:
                        ls = pd.read_html(temp_html_name)
                        os.remove(temp_html_name)
                        df_list.extend(ls)
                        for j in range(len(ls)):
                            page.append(i+1)
                    except:
                        pass
                for filename in os.listdir(temp_dir):
                    if os.path.isfile(os.path.join(temp_dir, filename)):
                        os.remove(os.path.join(temp_dir, filename))

        return df_list, page
    except:
        return [], page


async def is_incremental(s1, s2):
    s1_ascii, s2_ascii = sum([ord(ele) for ele in s1]), sum(
        [ord(ele) for ele in s2])
    if (s1_ascii+1 == s2_ascii or s1_ascii == s2_ascii):
        return True
    else:
        return False


async def check_incremental_row(tab_1, tab_2):
    # If first column incremental
    tab_1_last = tab_1.iloc[-1, 0]
    tab_2_first = tab_2.iloc[0, 0]
    tab_2_1st_col_name = tab_2.columns[0]

    # IF TAB1(1,LAST) == TAB2(1,1)
    if (await is_incremental(str(tab_1_last), str(tab_2_first))):
        tab_1.loc[len(tab_1)] = tab_2.values[0]
        tab_2.columns = tab_1.columns
        return True
    # IF TAB1(1,LAST) == COL_NAME(1)
    elif (await is_incremental(str(tab_1_last), str(tab_2_1st_col_name))):
        tab_2.columns
        tab_1.loc[len(tab_1)] = tab_2.columns
        tab_2.columns = tab_1.columns

        return True
    else:
        return False


async def merge_same_column(df):
    i = 1
    while i < len(df.columns):
        if (df.columns[i-1] == df.columns[i]):
            for j in df.iterrows():
                if (j[1][0]) != j[1][1]:
                    val = f"{j[1][0]}"+f"{j[1][1]}"
                    j[1][0] = j[1][1] = val

            df = df.loc[:, ~df.columns.duplicated()].copy()

        else:

            same_in_both_column = 0
            for j in df.iterrows():
                try:
                    if (j[1, 0]) == j[1, 1]:
                        same_in_both_column += 1
                except:
                    continue
            if (same_in_both_column == df.shape[0]):
                val = f"{df.columns[i-1]}"+f"{df.columns[i]}"
                temp_dict = {df.columns[i-1]: val, df.columns[i]: val}
                df.rename(columns=temp_dict, inplace=True)
            else:
                i += 1
            df = df.loc[:, ~df.columns.duplicated()].copy()

    return df


async def filter_table(df_list, page):
    # filter individual table
    filtered_pages = []
    clean_tables = []
    for i, df in enumerate(df_list):
        await merge_same_column(df)

        total_row = df.shape[0]
        total_column = df.shape[1]

        # to specify the column name if named [0,1,2,3...]
        if (df.columns == range(total_column)).all():
            updated_column_names = dict(zip(df.columns, df.iloc[0]))

            # Drop the first row since it's used for column renaming
            df = df.drop(0)
            df.reset_index(drop=True, inplace=True)
            total_row -= 1

            # Rename columns using the updated_column_names dictionary
            df.rename(columns=updated_column_names, inplace=True)

    # Delete rows if all elements are the same
        all_same_rows = df.apply(lambda row: row.nunique() != 1, axis=1)
        df = df[all_same_rows]

        # Delete columns if column name is null or if all elements are null
        df = df.dropna(axis=1, how='all')

        # delete columns if column name is nan
        df = df.drop(columns=[col for col in df.columns if col is np.nan])

        df = df.fillna("")
        df.index = range(len(df.index))
        try:
            if ((df.columns == [df.columns[0]]*len(df.columns)).all()):
                df.columns = df.iloc[0]
                df.drop(df.index[0], inplace=True)
                df.reset_index(drop=True, inplace=True)
            else:
                df = await merge_same_column(df)
        except:
            pass
        filtered_pages.append(page[i])
        clean_tables.append(df)

    # merge tables
    i = 1
    while i < len(clean_tables):
        try:

            if clean_tables[i].shape[1] == clean_tables[i-1].shape[1]:
                # IF NO OF COLUMN SAME
                if ((clean_tables[i].columns == clean_tables[i-1].columns).all()):

                    # JOIN IF COLUMN NAME SAME
                    clean_tables[i -
                                 1] = pd.concat([clean_tables[i-1], clean_tables[i]])
                    clean_tables.pop(i)
                    filtered_pages.remove(filtered_pages[i])

                elif (await check_incremental_row(clean_tables[i-1], clean_tables[i])):
                    clean_tables[i -
                                 1] = pd.concat([clean_tables[i-1], clean_tables[i]])
                    clean_tables.pop(i)
                    filtered_pages.remove(filtered_pages[i])
                else:
                    i += 1

            else:
                i += 1
        except:
            i += 1
    fcl = []
    for df in clean_tables:
        # delete single row column table
        #         if (df.shape[0] <= 1 or df.shape[1] <=1):
        #             continue
        # #         elif()
        #         else:
        fcl.append(df)

    return fcl, filtered_pages


async def give_prompt_from_document(file_path):
    extracted_dataframes, pages = await extract_dataframe_list(
        file_path, './temp')

    filtered_tables, filtered_pages = await filter_table(
        extracted_dataframes, pages)

    doc_list = []
    for x, df in enumerate(filtered_tables):
        df.reset_index(drop=True, inplace=True)

        des_list = []
        for i in range(df.shape[0]):
            description = f"For {i} "
            row_values = [
                f"{df.columns[j]} is {df.iloc[i, j]}" for j in range(df.shape[1])]
            null_element_in_row = sum(
                1 for value in row_values if value == "")
            description += ", ".join(row_values) + "."
            des_list.append(description)
        doc_text = ''
        for i, des in enumerate(des_list):
            if len(doc_text+des) < 1200:
                doc_text += des
                if i == len(des_list)-1:
                    doc = Document(page_content=doc_text, metadata={
                        "source": file_path, "page": filtered_pages[x]})
                    doc_list.append(doc)
            else:
                doc = Document(page_content=doc_text, metadata={
                    "source": file_path, "page": filtered_pages[x]})
                doc_list.append(doc)
                doc_text = des

    return doc_list, filtered_tables
