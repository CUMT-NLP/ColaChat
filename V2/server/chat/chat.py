from fastapi import Body
from sse_starlette.sse import EventSourceResponse
from configs import LLM_MODELS, TEMPERATURE
from server.utils import wrap_done, get_ChatOpenAI
from langchain.chains import LLMChain
from langchain.callbacks import AsyncIteratorCallbackHandler
from typing import AsyncIterable
import asyncio
import json
from langchain.prompts.chat import ChatPromptTemplate, MessagesPlaceholder
from typing import List, Optional, Union
from server.chat.utils import History
from langchain.prompts import PromptTemplate
from server.utils import get_prompt_template
from server.memory.conversation_db_buffer_memory import ConversationBufferDBMemory
from server.db.repository import add_message_to_db
from server.callback_handler.conversation_callback_handler import ConversationCallbackHandler
from langchain_core.messages import convert_to_messages

from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer
from openai import OpenAI
import concurrent.futures
def calculate_bm25_score(curr_ctx, response):
    tokenizer = AutoTokenizer.from_pretrained(r"E:\LLaMA-Factory\models\qwen2.5\qwen2.5-gptq", use_fast=False, trust_remote_code=True)
    tokenized_context = tokenizer.tokenize(curr_ctx)
    bm25 = BM25Okapi(tokenized_context)
    tokenized_response = tokenizer.tokenize(response)
    scores = bm25.get_scores(tokenized_response)
    return scores.mean()

def calculate_embedding_score(curr_ctx, response):
    model = SentenceTransformer(r"E:\coalchat\web\chatchat\bge-large-zh")
    q_embeddings = model.encode(response, normalize_embeddings=True)
    p_embeddings = model.encode(curr_ctx, normalize_embeddings=True)
    scores = q_embeddings @ p_embeddings.T
    scores_list = scores.tolist()
    return scores_list

class RewriteQuestion:
    def __init__(self,history,query,max_retries=3):
        self.instruction = '''
        我的目标是优化查询，以便检索到高质量的答案文档。
        给定一个问题及其相关上下文，需要严格遵循以下准则：
        1. 通过消除歧义、解决核心参照问题和补充缺失信息，对问题进行去语境化。改写后的问题既要保持原意，又要提供尽可能多的相关信息，而且不应重复上下文中已经提出过的问题。
        2. 评估新问题是当前讨论的延续还是一个全新话题的引入。如果是对新话题的介绍，则不应重新措辞，应保持问题的原意，且不应重复上下文中已提出的问题；如果新问题是当前讨论的延续，则应根据第一条准则重新措辞。
        3. 如果原始问题包含专有名词或特殊术语，改写时一定要保留这些词，因为它们对于准确检索相关文档至关重要。
        4. 确保输出结果是一个重新措辞的问题，没有任何冗余内容，并且必须以问号结束。
                '''
        self.condense_q_system_prompt = """
        你是一个优秀的查询改写器。
        """
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
    def forward(self):
        rewritten_questions = []  # 存储所有生成的问题
        delay = 0.5
        iter_nums = 4
        import time
        for i in range(iter_nums):
            try:
                response = self.client.chat.completions.create(
                    model="deepseek-chat",
                    messages=self.messages,
                    temperature=0.9,
                    stream=False,
                    max_tokens=2048
                )
                rewritten_question = response.choices[0].message.content
                # print(rewritten_question)
            except:
                continue
            time.sleep(delay)
            rewritten_questions.append(rewritten_question)
            if len(rewritten_questions) == self.max_retries:
                break
        bm25_scores = [calculate_bm25_score(self.history, rq) for rq in rewritten_questions]
        dense_scores = [calculate_embedding_score(self.history, rq) for rq in rewritten_questions]

        weight_bm25 = 0.5
        weight_dense = 1
        combined_scores = [weight_bm25 * bm25 + weight_dense * dense for bm25, dense in zip(bm25_scores, dense_scores)]
        # 找出最大分数的索引
        max_index = combined_scores.index(max(combined_scores))
        best_rewrite = rewritten_questions[max_index]
        return best_rewrite

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
#
#     def _call_api(self):
#         try:
#             response = self.client.chat.completions.create(
#                 model="deepseek-chat",
#                 messages=self.messages,
#                 temperature=0.7,
#                 stream=False,
#                 max_tokens=2048
#             )
#             return response.choices[0].message.content.strip()
#         except Exception:
#             return None
#
#     def forward(self):
#         rewritten_questions = []
#         with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
#             futures = [executor.submit(self._call_api) for _ in range(self.max_retries)]
#             for future in concurrent.futures.as_completed(futures):
#                 result = future.result()
#                 if result:
#                     rewritten_questions.append(result)
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

async def chat(query: str = Body(..., description="用户输入", examples=["恼羞成怒"]),
               conversation_id: str = Body("", description="对话框ID"),
               history_len: int = Body(-1, description="从数据库中取历史消息的数量"),
               history: Union[int, List[History]] = Body([],
                                                         description="历史对话，设为一个整数可以从数据库中读取历史消息",
                                                         examples=[[
                                                             {"role": "user",
                                                              "content": "我们来玩成语接龙，我先来，生龙活虎"},
                                                             {"role": "assistant", "content": "虎头虎脑"}]]
                                                         ),
               stream: bool = Body(False, description="流式输出"),
               model_name: str = Body(LLM_MODELS[0], description="LLM 模型名称。"),
               temperature: float = Body(TEMPERATURE, description="LLM 采样温度", ge=0.0, le=2.0),
               max_tokens: Optional[int] = Body(None, description="限制LLM生成Token数量，默认None代表模型最大值"),
               # top_p: float = Body(TOP_P, description="LLM 核采样。勿与temperature同时设置", gt=0.0, lt=1.0),
               prompt_name: str = Body("default", description="使用的prompt模板名称(在configs/prompt_config.py中配置)"),
               ):
    async def chat_iterator() -> AsyncIterable[str]:
        nonlocal history, max_tokens
        callback = AsyncIteratorCallbackHandler()
        callbacks = [callback]
        memory = None

        # 负责保存llm response到message db
        message_id = add_message_to_db(chat_type="llm_chat", query=query, conversation_id=conversation_id)
        conversation_callback = ConversationCallbackHandler(conversation_id=conversation_id, message_id=message_id,
                                                            chat_type="llm_chat",
                                                            query=query)
        callbacks.append(conversation_callback)

        # Enable langchain-chatchat to support langfuse
        import os
        langfuse_secret_key = os.environ.get('LANGFUSE_SECRET_KEY')
        langfuse_public_key = os.environ.get('LANGFUSE_PUBLIC_KEY')
        langfuse_host = os.environ.get('LANGFUSE_HOST')
        if langfuse_secret_key and langfuse_public_key and langfuse_host :
            from langfuse import Langfuse
            from langfuse.callback import CallbackHandler
            langfuse_handler = CallbackHandler()
            callbacks.append(langfuse_handler)


        if isinstance(max_tokens, int) and max_tokens <= 0:
            max_tokens = None

        model = get_ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            callbacks=callbacks,
        )

        if history: # 优先使用前端传入的历史消息
            history = [History.from_data(h) for h in history]
            prompt_template = get_prompt_template("llm_chat", prompt_name)
            input_msg = History(role="user", content=prompt_template).to_msg_template(False)
            chat_prompt = ChatPromptTemplate.from_messages(
                [i.to_msg_template() for i in history] + [input_msg])
        elif conversation_id and history_len > 0: # 前端要求从数据库取历史消息
            # 使用memory 时必须 prompt 必须含有memory.memory_key 对应的变量
            prompt_template = get_prompt_template("llm_chat", "with_history")
            chat_prompt = PromptTemplate.from_template(prompt_template)
            # 根据conversation_id 获取message 列表进而拼凑 memory
            memory = ConversationBufferDBMemory(conversation_id=conversation_id,
                                                llm=model,
                                                message_limit=history_len)
        else:
            prompt_template = get_prompt_template("llm_chat", prompt_name)
            input_msg = History(role="user", content=prompt_template).to_msg_template(False)
            chat_prompt = ChatPromptTemplate.from_messages([input_msg])
        chain = LLMChain(prompt=chat_prompt, llm=model, memory=memory)

        condense_question_best = query
        # if isinstance(history, list) and len(history) > 0:
        #     history_message_text = "\n".join([f"{h.role}: {h.content}" for h in history])

            # RQ = RewriteQuestion(history=history_message_text,query=condense_question_best)
            # condense_question_best = RQ.forward()

        # Begin a task that runs in the background.
        task = asyncio.create_task(wrap_done(
            chain.acall({"input": condense_question_best}),
            callback.done),
        )
        if stream:
            async for token in callback.aiter():
                # Use server-sent-events to stream the response
                yield json.dumps(
                    {"text": token, "message_id": message_id},
                    ensure_ascii=False)
        else:
            answer = ""
            async for token in callback.aiter():
                answer += token
            yield json.dumps(
                {"text": answer, "message_id": message_id},
                ensure_ascii=False)
        await task

    return EventSourceResponse(chat_iterator())
