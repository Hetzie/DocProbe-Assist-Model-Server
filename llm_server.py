from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route
import asyncio
import time


async def resolve_query(request):
    data = await request.json()
    query = data.get("query", None)
    if query == None:
        return JSONResponse({'message': 'enter query'}, status_code=400)
    response_q = asyncio.Queue()
    await app.model_queue.put(())
    answer = await response_q.get()

    return JSONResponse({'message': 'ok'})


async def server_loop(q: asyncio.Queue):
    # Intialize LLM model
    local_llm = str
    while True:
        (query, response_q) = await q.get()
        prompt = query
        time.sleep(10)
        await response_q.put(local_llm(prompt))


app = Starlette(
    routes=[
        Route('/resolve-query/', resolve_query, methods=["POST"])
    ]
)


@app.on_event("startup")
async def startup_event():
    q = asyncio.Queue()
    app.model_queue = q
    asyncio.create_task(server_loop(q))
