# from fastapi import Body, Request
# from sse_starlette.sse import EventSourceResponse
# from fastapi.concurrency import run_in_threadpool
# from configs import (LLM_MODELS,
#                      VECTOR_SEARCH_TOP_K,
#                      SCORE_THRESHOLD,
#                      TEMPERATURE,
#                      USE_RERANKER,
#                      RERANKER_MODEL,
#                      RERANKER_MAX_LENGTH)
# from server.utils import wrap_done, get_ChatOpenAI, get_model_path
# from server.utils import BaseResponse, get_prompt_template
# from langchain.chains import LLMChain
# from langchain.callbacks import AsyncIteratorCallbackHandler
# from typing import AsyncIterable, List, Optional
# import asyncio, json
# from langchain.prompts.chat import ChatPromptTemplate
# from server.chat.utils import History
# from server.knowledge_base.kb_service.base import KBServiceFactory
# from urllib.parse import urlencode
# from server.knowledge_base.kb_doc_api import search_docs
# from server.reranker.reranker import LangchainReranker
# from server.utils import embedding_device
# async def knowledge_base_chat(query: str = Body(..., description="用户输入", examples=["你好"]),
#                               knowledge_base_name: str = Body(..., description="知识库名称", examples=["samples"]),
#                               top_k: int = Body(VECTOR_SEARCH_TOP_K, description="匹配向量数"),
#                               score_threshold: float = Body(
#                                   SCORE_THRESHOLD,
#                                   description="知识库匹配相关度阈值，取值范围在0-1之间，SCORE越小，相关度越高，取到1相当于不筛选，建议设置在0.5左右",
#                                   ge=0,
#                                   le=2
#                               ),
#                               history: List[History] = Body(
#                                   [],
#                                   description="历史对话",
#                                   examples=[[
#                                       {"role": "user",
#                                        "content": "我们来玩成语接龙，我先来，生龙活虎"},
#                                       {"role": "assistant",
#                                        "content": "虎头虎脑"}]]
#                               ),
#                               stream: bool = Body(False, description="流式输出"),
#                               model_name: str = Body(LLM_MODELS[0], description="LLM 模型名称。"),
#                               temperature: float = Body(TEMPERATURE, description="LLM 采样温度", ge=0.0, le=1.0),
#                               max_tokens: Optional[int] = Body(
#                                   None,
#                                   description="限制LLM生成Token数量，默认None代表模型最大值"
#                               ),
#                               prompt_name: str = Body(
#                                   "default",
#                                   description="使用的prompt模板名称(在configs/prompt_config.py中配置)"
#                               ),
#                               request: Request = None,
#                               ):
#     kb = KBServiceFactory.get_service_by_name(knowledge_base_name)
#     if kb is None:
#         return BaseResponse(code=404, msg=f"未找到知识库 {knowledge_base_name}")
#
#     history = [History.from_data(h) for h in history]
#
#     async def knowledge_base_chat_iterator(
#             query: str,
#             top_k: int,
#             history: Optional[List[History]],
#             model_name: str = model_name,
#             prompt_name: str = prompt_name,
#     ) -> AsyncIterable[str]:
#         nonlocal max_tokens
#         callback = AsyncIteratorCallbackHandler()
#         if isinstance(max_tokens, int) and max_tokens <= 0:
#             max_tokens = None
#
#         callbacks = [callback]
#         # Enable langchain-chatchat to support langfuse
#         import os
#         langfuse_secret_key = os.environ.get('LANGFUSE_SECRET_KEY')
#         langfuse_public_key = os.environ.get('LANGFUSE_PUBLIC_KEY')
#         langfuse_host = os.environ.get('LANGFUSE_HOST')
#         if langfuse_secret_key and langfuse_public_key and langfuse_host :
#             from langfuse import Langfuse
#             from langfuse.callback import CallbackHandler
#             langfuse_handler = CallbackHandler()
#             callbacks.append(langfuse_handler)
#
#         model = get_ChatOpenAI(
#             model_name=model_name,
#             temperature=temperature,
#             max_tokens=max_tokens,
#             callbacks=callbacks,
#         )
#         docs = await run_in_threadpool(search_docs,
#                                        query=query,
#                                        knowledge_base_name=knowledge_base_name,
#                                        top_k=top_k,
#                                        score_threshold=score_threshold)
#
#         # 加入reranker
#         if USE_RERANKER:
#             reranker_model_path = get_model_path(RERANKER_MODEL)
#             reranker_model = LangchainReranker(top_n=top_k,
#                                             device=embedding_device(),
#                                             max_length=RERANKER_MAX_LENGTH,
#                                             model_name_or_path=reranker_model_path
#                                             )
#             print("-------------before rerank-----------------")
#             print(docs)
#             docs = reranker_model.compress_documents(documents=docs,
#                                                      query=query)
#             print("------------after rerank------------------")
#             print(docs)
#         context = "\n".join([doc.page_content for doc in docs])
#
#         if len(docs) == 0:  # 如果没有找到相关文档，使用empty模板
#             prompt_template = get_prompt_template("knowledge_base_chat", "empty")
#         else:
#             prompt_template = get_prompt_template("knowledge_base_chat", prompt_name)
#         input_msg = History(role="user", content=prompt_template).to_msg_template(False)
#         chat_prompt = ChatPromptTemplate.from_messages(
#             [i.to_msg_template() for i in history] + [input_msg])
#
#         chain = LLMChain(prompt=chat_prompt, llm=model)
#
#         # Begin a task that runs in the background.
#         task = asyncio.create_task(wrap_done(
#             chain.acall({"context": context, "question": query}),
#             callback.done),
#         )
#
#         source_documents = []
#         for inum, doc in enumerate(docs):
#             filename = doc.metadata.get("source")
#             parameters = urlencode({"knowledge_base_name": knowledge_base_name, "file_name": filename})
#             base_url = request.base_url
#             url = f"{base_url}knowledge_base/download_doc?" + parameters
#             text = f"""出处 [{inum + 1}] [{filename}]({url}) \n\n{doc.page_content}\n\n"""
#             source_documents.append(text)
#
#         if len(source_documents) == 0:  # 没有找到相关文档
#             source_documents.append(f"<span style='color:red'>未找到相关文档,该回答为大模型自身能力解答！</span>")
#
#         if stream:
#             async for token in callback.aiter():
#                 # Use server-sent-events to stream the response
#                 yield json.dumps({"answer": token}, ensure_ascii=False)
#             yield json.dumps({"docs": source_documents}, ensure_ascii=False)
#         else:
#             answer = ""
#             async for token in callback.aiter():
#                 answer += token
#             yield json.dumps({"answer": answer,
#                               "docs": source_documents},
#                              ensure_ascii=False)
#         await task
#
#     return EventSourceResponse(knowledge_base_chat_iterator(query, top_k, history,model_name,prompt_name))


from fastapi import Body, Request
from sse_starlette.sse import EventSourceResponse
from fastapi.concurrency import run_in_threadpool
from configs import (LLM_MODELS,
                     VECTOR_SEARCH_TOP_K,
                     SCORE_THRESHOLD,
                     TEMPERATURE,
EMBEDDING_KEYWORD_FILE,
MODEL_PATH,
EMBEDDING_MODEL,
                     # REWRITE_QUERY_PROMPT,
                     USE_RERANKER,
                     RERANKER_MODEL,
                     RERANKER_MAX_LENGTH)
from server.utils import wrap_done, get_ChatOpenAI, get_model_path
from server.utils import BaseResponse, get_prompt_template
from langchain.chains import LLMChain
from langchain.callbacks import AsyncIteratorCallbackHandler
from typing import AsyncIterable, List, Optional
import asyncio, json
from langchain.prompts.chat import ChatPromptTemplate
from langchain.prompts import PromptTemplate
from server.chat.utils import History
from server.knowledge_base.kb_service.base import KBServiceFactory
from urllib.parse import urlencode
from server.knowledge_base.kb_doc_api import search_docs
from server.reranker.reranker import LangchainReranker
from server.utils import embedding_device
from server.memory.conversation_db_buffer_memory import ConversationBufferDBMemory
from server.db.repository import add_message_to_db
from server.callback_handler.conversation_callback_handler import ConversationCallbackHandler
from typing import List, Optional, Union

# def calculate_bm25_score(curr_ctx, response):
#     tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH["llm_model"][LLM_MODELS[0]], use_fast=False, trust_remote_code=True)
#     tokenized_context = tokenizer.tokenize(curr_ctx)
#     bm25 = BM25Okapi(tokenized_context)
#     tokenized_response = tokenizer.tokenize(response)
#     scores = bm25.get_scores(tokenized_response)
#     return scores.mean()
#
# def calculate_embedding_score(curr_ctx, response):
#     # model = SentenceTransformer(r"E:\coalchat\web\chatchat\bge-large-zh")
#     model = SentenceTransformer(MODEL_PATH["embed_model"][EMBEDDING_MODEL])
#
#     q_embeddings = model.encode(response, normalize_embeddings=True)
#     p_embeddings = model.encode(curr_ctx, normalize_embeddings=True)
#     scores = q_embeddings @ p_embeddings.T
#     scores_list = scores.tolist()
#     return scores_list
#
# class RewriteQuestion:
#     def __init__(self,history,query,max_retries=3):
#         self.instruction = '''
#         我的目标是优化查询，以便检索到高质量的答案文档。
#         给定一个问题及其相关上下文，需要严格遵循以下准则：
#         1. 通过消除歧义、解决核心参照问题和补充缺失信息，对问题进行去语境化。改写后的问题既要保持原意，又要提供尽可能多的相关信息，而且不应重复上下文中已经提出过的问题。
#         2. 评估新问题是当前讨论的延续还是一个全新话题的引入。如果是对新话题的介绍，则不应重新措辞，应保持问题的原意，且不应重复上下文中已提出的问题；如果新问题是当前讨论的延续，则应根据第一条准则重新措辞。
#         3. 如果原始问题包含专有名词或特殊术语，改写时一定要保留这些词，因为它们对于准确检索相关文档至关重要。
#         4. 确保输出结果是一个重新措辞的问题，没有任何冗余内容，并且必须以问号结束。
#                 '''
#         self.condense_q_system_prompt = """
#         你是一个优秀的查询改写器。
#         """
#         self.prompt = f"{self.instruction}\n\n历史对话: {history}\n原始问题: {query}\n重写问题: "
#         self.messages = [
#             {"role": "system", "content": self.condense_q_system_prompt},
#             {"role": "user", "content": self.prompt}
#         ]
#         self.client = OpenAI(
#             api_key="sk-85f0c206022840d286a279e438b22f8c",
#             base_url="https://api.deepseek.com",
#         )
#         self.history = history
#         self.max_retries = max_retries
#     def forward(self):
#         rewritten_questions = []  # 存储所有生成的问题
#         iter_nums = 4
#         for i in range(iter_nums):
#             try:
#                 response = self.client.chat.completions.create(
#                     model="deepseek-chat",
#                     messages=self.messages,
#                     temperature=0.9,
#                     stream=False,
#                     max_tokens=2048
#                 )
#                 rewritten_question = response.choices[0].message.content
#             except:
#                 continue
#             rewritten_questions.append(rewritten_question)
#             if len(rewritten_questions) == self.max_retries:
#                 break
#         bm25_scores = [calculate_bm25_score(self.history, rq) for rq in rewritten_questions]
#         dense_scores = [calculate_embedding_score(self.history, rq) for rq in rewritten_questions]
#
#         weight_bm25 = 0.5
#         weight_dense = 1
#         combined_scores = [weight_bm25 * bm25 + weight_dense * dense for bm25, dense in zip(bm25_scores, dense_scores)]
#         # 找出最大分数的索引
#         max_index = combined_scores.index(max(combined_scores))
#         best_rewrite = rewritten_questions[max_index]
#         return best_rewrite

import asyncio
from concurrent.futures import ThreadPoolExecutor
from transformers import AutoTokenizer
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from openai import OpenAI

def calculate_bm25_model(curr_ctx):
    """ 预计算 BM25 模型，避免重复初始化 """
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH["llm_model"][LLM_MODELS[0]], use_fast=False,
                                              trust_remote_code=True)

    tokenized_context = tokenizer.tokenize(curr_ctx)
    return BM25Okapi(tokenized_context)


def calculate_bm25_score(bm25, response):
    """ 直接使用预计算 BM25 计算分数 """
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH["llm_model"][LLM_MODELS[0]], use_fast=False,
                                              trust_remote_code=True)
    tokenized_response = tokenizer.tokenize(response)
    scores = bm25.get_scores(tokenized_response)
    return scores.mean()


def calculate_embedding_scores(model, curr_ctx, responses):
    """ 批量计算 embedding 余弦相似度 """
    embeddings = model.encode([curr_ctx] + responses, normalize_embeddings=True)
    p_embeddings = embeddings[0]  # 当前上下文
    q_embeddings = embeddings[1:]  # 生成的改写问题
    scores = (q_embeddings @ p_embeddings.T).tolist()
    return scores


class RewriteQuestion:
    def __init__(self, history, query, max_retries=2):
        self.instruction = '''
        我的目标是优化查询，以便检索到高质量的答案文档。
        给定一个问题及其相关上下文，需要严格遵循以下准则：
        1. 通过消除歧义、解决核心参照问题和补充缺失信息，对问题进行去语境化。改写后的问题既要保持原意，又要提供尽可能多的相关信息，而且不应重复上下文中已经提出过的问题。
        2. 评估新问题是当前讨论的延续还是一个全新话题的引入。如果是对新话题的介绍，则不应重新措辞，应保持问题的原意，且不应重复上下文中已提出的问题；如果新问题是当前讨论的延续，则应根据第一条准则重新措辞。
        3. 如果原始问题包含专有名词或特殊术语，改写时一定要保留这些词，因为它们对于准确检索相关文档至关重要。
        4. 确保输出结果是一个重新措辞的问题，没有任何冗余内容，并且必须以问号结束。
        '''
        self.condense_q_system_prompt = "你是一个优秀的查询改写器。"
        self.prompt = f"{self.instruction}\n\n历史对话: {history}\n原始问题: {query}\n重写问题: "
        self.messages = [
            {"role": "system", "content": self.condense_q_system_prompt},
            {"role": "user", "content": self.prompt}
        ]
        self.client = OpenAI(
            api_key="sk-85f0c206022840d286a279e438b22f8c",
            base_url="https://api.deepseek.com",
        )
        self.history = history
        self.max_retries = max_retries
        self.bm25 = calculate_bm25_model(history)  # 预计算 BM25
        self.model = SentenceTransformer(MODEL_PATH["embed_model"][EMBEDDING_MODEL])  # 加载 embedding 模型

    async def async_generate(self):
        """ 异步并行调用 API """
        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor() as pool:
            tasks = [loop.run_in_executor(pool, self.generate_once) for _ in range(self.max_retries)]
            return await asyncio.gather(*tasks)

    def generate_once(self):
        """ 生成单个改写问题 """
        try:
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=self.messages,
                temperature=0.7,
                stream=False,
                max_tokens=2048
            )
            return response.choices[0].message.content
        except:
            return None

    async def forward(self):
        """ 计算最佳改写问题 """
        rewritten_questions = await  self.async_generate()
        rewritten_questions = [rq for rq in rewritten_questions if rq]  # 过滤掉 None

        if not rewritten_questions:
            return self.history  # 若未生成，返回原问题

        # 计算 BM25 分数
        bm25_scores = [calculate_bm25_score(self.bm25, rq) for rq in rewritten_questions]

        # 计算 embedding 余弦相似度
        dense_scores = calculate_embedding_scores(self.model, self.history, rewritten_questions)

        # 计算加权分数
        weight_bm25, weight_dense = 0.5, 1
        combined_scores = [weight_bm25 * bm25 + weight_dense * dense for bm25, dense in zip(bm25_scores, dense_scores)]

        # 选择最佳改写问题
        best_rewrite = rewritten_questions[combined_scores.index(max(combined_scores))]
        return best_rewrite


async def knowledge_base_chat(query: str = Body(..., description="用户输入", examples=["你好"]),
                    knowledge_base_name: str = Body(..., description="知识库名称", examples=["samples"]),
                    top_k: int = Body(VECTOR_SEARCH_TOP_K, description="匹配向量数"),
                    score_threshold: float = Body(
                        SCORE_THRESHOLD,
                        description="知识库匹配相关度阈值，取值范围在0-1之间，SCORE越小，相关度越高，取到1相当于不筛选，建议设置在0.5左右",
                        ge=0,
                        le=2
                    ),
                    history: List[History] = Body(
                        [],
                        description="历史对话",
                        examples=[[
                            {"role": "user",
                             "content": "我们来玩成语接龙，我先来，生龙活虎"},
                            {"role": "assistant",
                             "content": "虎头虎脑"}]]
                    ),
                    stream: bool = Body(False, description="流式输出"),
                    model_name: str = Body(LLM_MODELS[0], description="LLM 模型名称。"),
                    temperature: float = Body(TEMPERATURE, description="LLM 采样温度", ge=0.0, le=1.0),
                    max_tokens: Optional[int] = Body(
                        None,
                        description="限制LLM生成Token数量，默认None代表模型最大值"
                    ),
                    prompt_name: str = Body(
                        "default",
                        description="使用的prompt模板名称(在configs/prompt_config.py中配置)"
                    ),
                    conversation_id: str = Body("", description="对话框ID"),
                    history_len: int = Body(-1, description="从数据库中取历史消息的数量"),
                    request: Request = None,
                    ):
    kb = KBServiceFactory.get_service_by_name(knowledge_base_name)
    if kb is None:
        return BaseResponse(code=404, msg=f"未找到知识库 {knowledge_base_name}")

    history = [History.from_data(h) for h in history]


    async def knowledge_base_chat_iterator(
            query: str,
            top_k: int,
            history: Optional[List[History]],
            model_name: str = model_name,
            prompt_name: str = prompt_name,
    ) -> AsyncIterable[str]:
        nonlocal max_tokens
        if not history:
            callback_knowledge_base = AsyncIteratorCallbackHandler()
            callbacks_knowledge_base = [callback_knowledge_base]

            if isinstance(max_tokens, int) and max_tokens <= 0:
                max_tokens = None

            model_knowledge_base = get_ChatOpenAI(
                model_name=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                callbacks=callbacks_knowledge_base,
            )
            # 直接召回
            docs = await run_in_threadpool(search_docs,
                                           query=query,
                                           knowledge_base_name=knowledge_base_name,
                                           top_k=top_k,
                                           score_threshold=score_threshold)

            context = "\n".join([doc.page_content for doc in docs])

            if len(docs) == 0:  # 如果没有找到相关文档，使用empty模板
                prompt_template = get_prompt_template("knowledge_base_chat", "empty")
            else:
                prompt_template = get_prompt_template("knowledge_base_chat", prompt_name)

            input_msg = History(role="user", content=prompt_template).to_msg_template(False)
            knowledge_base_chat_prompt = ChatPromptTemplate.from_messages([input_msg])


            chain_knowledge_base_chat = LLMChain(prompt=knowledge_base_chat_prompt, llm=model_knowledge_base)
            task = asyncio.create_task(wrap_done(
                chain_knowledge_base_chat.acall({"context": context, "question": query}),
                callback_knowledge_base.done),
            )

            source_documents = []

            for inum, doc in enumerate(docs):
                filename = doc.metadata.get("source")
                parameters = urlencode({"knowledge_base_name": knowledge_base_name, "file_name": filename})
                base_url = request.base_url
                url = f"{base_url}knowledge_base/download_doc?" + parameters
                text = f"""出处[{inum + 1}] [{filename}]({url}) \n\n{doc.page_content.strip()}\n\n"""
                source_documents.append(text)

            if len(source_documents) == 0:  # 没有找到相关文档
                source_documents.append(f"<span style='color:red'>未找到相关文档,该回答为大模型自身能力解答！</span>")

            if stream:
                async for token in callback_knowledge_base.aiter():
                    yield json.dumps({"answer": token}, ensure_ascii=False)
                yield json.dumps({"docs": source_documents}, ensure_ascii=False)
            else:
                answer = ''
                async for token in callback_knowledge_base.aiter():
                    answer += token
                yield json.dumps({"answer": answer,
                                  "docs": source_documents,
                                  },
                                 ensure_ascii=False)
            await task
        else:
            # 使用历史对话中的最近三轮内容进行查询改写
            recent_history = history[-6:]  # 每轮包含一个user和assistant消息，最近三轮是6条消息
            recent_history_text = "\n".join([f"{h.role}: {h.content}" for h in recent_history])
            # # 查询改写
            # callback_rewrite = AsyncIteratorCallbackHandler()
            # callbacks_rewrite = [callback_rewrite]
            # message_id_rewrite = add_message_to_db(chat_type="llm_chat", query=query, conversation_id=conversation_id)
            # conversation_callback_rewrite = ConversationCallbackHandler(conversation_id=conversation_id,
            #                                                             message_id=message_id_rewrite,
            #                                                             chat_type="llm_chat",
            #                                                             query=query)
            # callbacks_rewrite.append(conversation_callback_rewrite)
            #
            # if isinstance(max_tokens, int) and max_tokens <= 0:
            #     max_tokens = None
            #
            # model_rewrite = get_ChatOpenAI(
            #     model_name=model_name,
            #     temperature=temperature,
            #     max_tokens=max_tokens,
            #     callbacks=callbacks_rewrite,
            # )
            #
            # rewrite_prompt_template = get_prompt_template("rewrite", 'default')
            # rewrite_llm_chat_prompt = PromptTemplate.from_template(rewrite_prompt_template)
            # print(rewrite_llm_chat_prompt)
            # rewrite_memory = ConversationBufferDBMemory(conversation_id=conversation_id,
            #                                             llm=model_rewrite,
            #                                             message_limit=history_len)
            # chain_llm_query = LLMChain(prompt=rewrite_llm_chat_prompt, llm=model_rewrite, memory=rewrite_memory)
            # task = asyncio.create_task(wrap_done(
            #     chain_llm_query.acall({"recent_history_text": recent_history_text,"query": query}),
            #     callback_rewrite.done),
            # )
            # query_rewrite_text = ""
            # async for token in callback_rewrite.aiter():
            #     query_rewrite_text += token
            # yield json.dumps({"query_rewrite_text": query_rewrite_text}, ensure_ascii=False)
            #
            # await task
            condense_question_best = query
            RQ = RewriteQuestion(history=recent_history_text, query=condense_question_best)
            condense_question_best = await RQ.forward()

            # 先输出 condense_question_best
            yield json.dumps(
                {"query_rewrite_text": condense_question_best, },
                ensure_ascii=False
            )

            # 召回
            docs = await run_in_threadpool(search_docs,
                                           query=condense_question_best ,
                                           knowledge_base_name=knowledge_base_name,
                                           top_k=top_k,
                                           score_threshold=score_threshold)

            context = "\n".join([doc.page_content for doc in docs])

            if len(docs) == 0:  # 如果没有找到相关文档，使用empty模板
                prompt_template = get_prompt_template("knowledge_base_chat", "empty")
            else:
                prompt_template = get_prompt_template("knowledge_base_chat", prompt_name)

            input_msg = History(role="user", content=prompt_template).to_msg_template(False)
            knowledge_base_chat_prompt = ChatPromptTemplate.from_messages(
                [i.to_msg_template() for i in history] + [input_msg])

            callback_knowledge_base = AsyncIteratorCallbackHandler()
            callbacks_knowledge_base = [callback_knowledge_base]
            model_knowledge_base = get_ChatOpenAI(
                model_name=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                callbacks=callbacks_knowledge_base,
            )
            chain_knowledge_base_chat = LLMChain(prompt=knowledge_base_chat_prompt, llm=model_knowledge_base)
            task = asyncio.create_task(wrap_done(
                chain_knowledge_base_chat.acall({"context": context, "question": query}),
                callback_knowledge_base.done),
            )

            source_documents = []

            for inum, doc in enumerate(docs):
                filename = doc.metadata.get("source")
                parameters = urlencode({"knowledge_base_name": knowledge_base_name, "file_name": filename})
                base_url = request.base_url
                url = f"{base_url}knowledge_base/download_doc?" + parameters
                text = f"""出处[{inum + 1}] [{filename}]({url}) \n\n{doc.page_content.strip()}\n\n"""
                source_documents.append(text)

            if len(source_documents) == 0:  # 没有找到相关文档
                source_documents.append(f"<span style='color:red'>未找到相关文档,该回答为大模型自身能力解答！</span>")

            if stream:
                async for token in callback_knowledge_base.aiter():
                    yield json.dumps({"answer": token}, ensure_ascii=False)
                yield json.dumps({"docs": source_documents}, ensure_ascii=False)
            else:
                answer = ""
                async for token in callback_knowledge_base.aiter():
                    answer += token
                yield json.dumps({"answer": answer,
                                  "docs": source_documents,
                                  },
                                 ensure_ascii=False)
            await task

    return EventSourceResponse(knowledge_base_chat_iterator(query, top_k, history,model_name,prompt_name))


# from fastapi import Body, Request
# from sse_starlette.sse import EventSourceResponse
# from fastapi.concurrency import run_in_threadpool
# from configs import (LLM_MODELS,
#                      VECTOR_SEARCH_TOP_K,
#                      SCORE_THRESHOLD,
#                      TEMPERATURE,
#                      EMBEDDING_KEYWORD_FILE,
#                      MODEL_PATH,
#                      EMBEDDING_MODEL,
#                      USE_RERANKER,
#                      RERANKER_MODEL,
#                      RERANKER_MAX_LENGTH)
# from server.utils import wrap_done, get_ChatOpenAI, get_model_path
# from server.utils import BaseResponse, get_prompt_template
# from langchain.chains import LLMChain
# from langchain.callbacks import AsyncIteratorCallbackHandler
# from typing import AsyncIterable, List, Optional
# import asyncio, json
# from langchain.prompts.chat import ChatPromptTemplate
# from langchain.prompts import PromptTemplate
# from server.chat.utils import History
# from server.knowledge_base.kb_service.base import KBServiceFactory
# from urllib.parse import urlencode
# from server.knowledge_base.kb_doc_api import search_docs
# from server.reranker.reranker import LangchainReranker
# from server.utils import embedding_device
# from server.memory.conversation_db_buffer_memory import ConversationBufferDBMemory
# from server.db.repository import add_message_to_db
# from server.callback_handler.conversation_callback_handler import ConversationCallbackHandler
# from typing import List, Optional, Union
#
# from sentence_transformers import SentenceTransformer
# from rank_bm25 import BM25Okapi
# from transformers import AutoTokenizer
#
#
# def calculate_bm25_score(curr_ctx, response):
#     tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH["llm_model"][LLM_MODELS[1]], use_fast=False,
#                                               trust_remote_code=True)
#     tokenized_context = tokenizer.tokenize(curr_ctx)
#     bm25 = BM25Okapi(tokenized_context)
#     tokenized_response = tokenizer.tokenize(response)
#     scores = bm25.get_scores(tokenized_response)
#     return scores.mean()
#
#
# def calculate_embedding_score(curr_ctx, response):
#     model = SentenceTransformer(MODEL_PATH["embed_model"][EMBEDDING_MODEL])
#     q_embeddings = model.encode(response, normalize_embeddings=True)
#     p_embeddings = model.encode(curr_ctx, normalize_embeddings=True)
#     scores = q_embeddings @ p_embeddings.T
#     scores_list = scores.tolist()
#     return scores_list
#
#
# def get_best_rewrite_query(history, rewrite_query_lsit):
#     bm25_scores = [calculate_bm25_score(history, rq) for rq in rewrite_query_lsit]
#     dense_scores = [calculate_embedding_score(history, rq) for rq in rewrite_query_lsit]
#     weight_bm25 = 0.5
#     weight_dense = 1
#     combined_scores = [weight_bm25 * bm25 + weight_dense * dense for bm25, dense in zip(bm25_scores, dense_scores)]
#     # 找出最大分数的索引
#     max_index = combined_scores.index(max(combined_scores))
#     best_rewrite = rewrite_query_lsit[max_index]
#     return best_rewrite
#
#
# async def knowledge_base_chat(query: str = Body(..., description="用户输入", examples=["你好"]),
#                               knowledge_base_name: str = Body(..., description="知识库名称", examples=["samples"]),
#                               top_k: int = Body(VECTOR_SEARCH_TOP_K, description="匹配向量数"),
#                               score_threshold: float = Body(
#                                   SCORE_THRESHOLD,
#                                   description="知识库匹配相关度阈值，取值范围在0-1之间，SCORE越小，相关度越高，取到1相当于不筛选，建议设置在0.5左右",
#                                   ge=0,
#                                   le=2
#                               ),
#                               history: List[History] = Body(
#                                   [],
#                                   description="历史对话",
#                                   examples=[[
#                                       {"role": "user",
#                                        "content": "我们来玩成语接龙，我先来，生龙活虎"},
#                                       {"role": "assistant",
#                                        "content": "虎头虎脑"}]]
#                               ),
#                               stream: bool = Body(False, description="流式输出"),
#                               model_name: str = Body(LLM_MODELS[0], description="LLM 模型名称。"),
#                               temperature: float = Body(TEMPERATURE, description="LLM 采样温度", ge=0.0, le=1.0),
#                               max_tokens: Optional[int] = Body(
#                                   None,
#                                   description="限制LLM生成Token数量，默认None代表模型最大值"
#                               ),
#                               prompt_name: str = Body(
#                                   "default",
#                                   description="使用的prompt模板名称(在configs/prompt_config.py中配置)"
#                               ),
#                               conversation_id: str = Body("", description="对话框ID"),
#                               history_len: int = Body(-1, description="从数据库中取历史消息的数量"),
#                               request: Request = None,
#                               ):
#     kb = KBServiceFactory.get_service_by_name(knowledge_base_name)
#     if kb is None:
#         return BaseResponse(code=404, msg=f"未找到知识库 {knowledge_base_name}")
#
#     history = [History.from_data(h) for h in history]
#
#     async def knowledge_base_chat_iterator(
#             query: str,
#             top_k: int,
#             history: Optional[List[History]],
#             model_name: str = model_name,
#             prompt_name: str = prompt_name,
#     ) -> AsyncIterable[str]:
#         nonlocal max_tokens
#         if not history:
#             callback_knowledge_base = AsyncIteratorCallbackHandler()
#             callbacks_knowledge_base = [callback_knowledge_base]
#
#             if isinstance(max_tokens, int) and max_tokens <= 0:
#                 max_tokens = None
#
#             model_knowledge_base = get_ChatOpenAI(
#                 model_name=model_name,
#                 temperature=temperature,
#                 max_tokens=max_tokens,
#                 callbacks=callbacks_knowledge_base,
#             )
#             # 直接召回
#             docs = await run_in_threadpool(search_docs,
#                                            query=query,
#                                            knowledge_base_name=knowledge_base_name,
#                                            top_k=top_k,
#                                            score_threshold=score_threshold)
#             context = "\n".join([doc.page_content for doc in docs])
#             if len(docs) == 0:  # 如果没有找到相关文档，使用empty模板
#                 prompt_template = get_prompt_template("knowledge_base_chat", "empty")
#             else:
#                 prompt_template = get_prompt_template("knowledge_base_chat", prompt_name)
#
#             input_msg = History(role="user", content=prompt_template).to_msg_template(False)
#             knowledge_base_chat_prompt = ChatPromptTemplate.from_messages([input_msg])
#
#             chain_knowledge_base_chat = LLMChain(prompt=knowledge_base_chat_prompt, llm=model_knowledge_base)
#             task = asyncio.create_task(wrap_done(
#                 chain_knowledge_base_chat.acall({"context": context, "question": query}),
#                 callback_knowledge_base.done),
#             )
#
#             source_documents = []
#
#             for inum, doc in enumerate(docs):
#                 filename = doc.metadata.get("source")
#                 parameters = urlencode({"knowledge_base_name": knowledge_base_name, "file_name": filename})
#                 base_url = request.base_url
#                 url = f"{base_url}knowledge_base/download_doc?" + parameters
#                 text = f"""出处[{inum + 1}] [{filename}]({url}) \n\n{doc.page_content.strip()}\n\n"""
#                 source_documents.append(text)
#
#             if len(source_documents) == 0:  # 没有找到相关文档
#                 source_documents.append(f"<span style='color:red'>未找到相关文档,该回答为大模型自身能力解答！</span>")
#
#             if stream:
#                 async for token in callback_knowledge_base.aiter():
#                     yield json.dumps({"answer": token}, ensure_ascii=False)
#                 yield json.dumps({"docs": source_documents}, ensure_ascii=False)
#             else:
#                 answer = ''
#                 async for token in callback_knowledge_base.aiter():
#                     answer += token
#                 yield json.dumps({"answer": answer,
#                                   "docs": source_documents,
#                                   },
#                                  ensure_ascii=False)
#             await task
#         else:
#             # 使用历史对话中的最近两轮内容进行查询改写
#             recent_history = history[-4:]  # 每轮包含一个user和assistant消息，最近两轮是4条消息
#             recent_history_text = "\n".join([h.content for h in recent_history])
#
#             rewrite_querys = []
#             for i in range(3):
#                 # 查询改写
#                 callback_rewrite = AsyncIteratorCallbackHandler()
#                 callbacks_rewrite = [callback_rewrite]
#                 message_id_rewrite = add_message_to_db(chat_type="llm_chat", query=query,
#                                                        conversation_id=conversation_id)
#                 conversation_callback_rewrite = ConversationCallbackHandler(conversation_id=conversation_id,
#                                                                             message_id=message_id_rewrite,
#                                                                             chat_type="llm_chat",
#                                                                             query=query)
#                 callbacks_rewrite.append(conversation_callback_rewrite)
#
#                 if isinstance(max_tokens, int) and max_tokens <= 0:
#                     max_tokens = None
#
#                 model_rewrite = get_ChatOpenAI(
#                     model_name=model_name,
#                     temperature=temperature,
#                     max_tokens=max_tokens,
#                     callbacks=callbacks_rewrite,
#                 )
#
#                 rewrite_prompt_template = get_prompt_template("rewrite", 'default')
#                 rewrite_llm_chat_prompt = PromptTemplate.from_template(rewrite_prompt_template)
#                 rewrite_memory = ConversationBufferDBMemory(conversation_id=conversation_id,
#                                                             llm=model_rewrite,
#                                                             message_limit=history_len)
#
#                 chain_llm_query = LLMChain(prompt=rewrite_llm_chat_prompt, llm=model_rewrite, memory=rewrite_memory)
#                 task = asyncio.create_task(wrap_done(
#                     chain_llm_query.acall({"recent_history_text": recent_history_text, "query": query}),
#                     callback_rewrite.done),
#                 )
#
#                 query_rewrite_text = ""
#                 async for token in callback_rewrite.aiter():
#                     query_rewrite_text += token
#                 await task
#                 rewrite_querys.append(query_rewrite_text)
#
#             best_query = get_best_rewrite_query(recent_history_text, rewrite_querys)
#             yield json.dumps({"query_rewrite_text": best_query}, ensure_ascii=False)
#
#             # 召回
#             docs = await run_in_threadpool(search_docs,
#                                            query=best_query,
#                                            knowledge_base_name=knowledge_base_name,
#                                            top_k=top_k,
#                                            score_threshold=score_threshold)
#
#             context = "\n".join([doc.page_content for doc in docs])
#
#             if len(docs) == 0:  # 如果没有找到相关文档，使用empty模板
#                 prompt_template = get_prompt_template("knowledge_base_chat", "empty")
#             else:
#                 prompt_template = get_prompt_template("knowledge_base_chat", prompt_name)
#
#             input_msg = History(role="user", content=prompt_template).to_msg_template(False)
#             knowledge_base_chat_prompt = ChatPromptTemplate.from_messages(
#                 [i.to_msg_template() for i in history] + [input_msg])
#
#             callback_knowledge_base = AsyncIteratorCallbackHandler()
#             callbacks_knowledge_base = [callback_knowledge_base]
#             model_knowledge_base = get_ChatOpenAI(
#                 model_name=model_name,
#                 temperature=temperature,
#                 max_tokens=max_tokens,
#                 callbacks=callbacks_knowledge_base,
#             )
#             chain_knowledge_base_chat = LLMChain(prompt=knowledge_base_chat_prompt, llm=model_knowledge_base)
#             task = asyncio.create_task(wrap_done(
#                 chain_knowledge_base_chat.acall({"context": context, "question": query}),
#                 callback_knowledge_base.done),
#             )
#
#             source_documents = []
#
#             for inum, doc in enumerate(docs):
#                 filename = doc.metadata.get("source")
#                 parameters = urlencode({"knowledge_base_name": knowledge_base_name, "file_name": filename})
#                 base_url = request.base_url
#                 url = f"{base_url}knowledge_base/download_doc?" + parameters
#                 text = f"""出处[{inum + 1}] [{filename}]({url}) \n\n{doc.page_content.strip()}\n\n"""
#                 source_documents.append(text)
#
#             if len(source_documents) == 0:  # 没有找到相关文档
#                 source_documents.append(f"<span style='color:red'>未找到相关文档,该回答为大模型自身能力解答！</span>")
#
#             if stream:
#                 async for token in callback_knowledge_base.aiter():
#                     yield json.dumps({"answer": token}, ensure_ascii=False)
#                 yield json.dumps({"docs": source_documents}, ensure_ascii=False)
#             else:
#                 answer = ""
#                 async for token in callback_knowledge_base.aiter():
#                     answer += token
#                 yield json.dumps({"answer": answer,
#                                   "docs": source_documents,
#                                   },
#                                  ensure_ascii=False)
#             await task
#
#     return EventSourceResponse(knowledge_base_chat_iterator(query, top_k, history, model_name, prompt_name))