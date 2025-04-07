# from fastapi import Body
# from sse_starlette.sse import EventSourceResponse
# from configs import LLM_MODELS, TEMPERATURE
# from server.utils import wrap_done, get_ChatOpenAI
# from langchain.chains import LLMChain
# from langchain.callbacks import AsyncIteratorCallbackHandler
# from typing import AsyncIterable
# import asyncio
# import json
# from langchain.prompts.chat import ChatPromptTemplate, MessagesPlaceholder
# from typing import List, Optional, Union
# from server.chat.utils import History
# from langchain.prompts import PromptTemplate
# from server.utils import get_prompt_template
# from server.memory.conversation_db_buffer_memory import ConversationBufferDBMemory
# from server.db.repository import add_message_to_db
# from server.callback_handler.conversation_callback_handler import ConversationCallbackHandler
# from langchain_core.messages import convert_to_messages
#
# from sentence_transformers import SentenceTransformer
# from rank_bm25 import BM25Okapi
# from transformers import AutoTokenizer
# from langchain.schema import StrOutputParser
# def calculate_bm25_score(curr_ctx, response):
#     tokenizer = AutoTokenizer.from_pretrained(r"E:\LLaMA-Factory\models\qwen2.5\qwen2.5-gptq", use_fast=False, trust_remote_code=True)
#     tokenized_context = tokenizer.tokenize(curr_ctx)
#     bm25 = BM25Okapi(tokenized_context)
#     tokenized_response = tokenizer.tokenize(response)
#     scores = bm25.get_scores(tokenized_response)
#     return scores.mean()
#
# def calculate_embedding_score(curr_ctx, response):
#     model = SentenceTransformer(r"E:\coalchat\web\chatchat\bge-large-zh")
#     q_embeddings = model.encode(response, normalize_embeddings=True)
#     p_embeddings = model.encode(curr_ctx, normalize_embeddings=True)
#     scores = q_embeddings @ p_embeddings.T
#     scores_list = scores.tolist()
#     return scores_list
#
# def history_message_text(history: List[History]) -> str:
#     """将历史消息转换为字符串格式，用于评分计算。"""
#     parts = []
#     for idx, sent in enumerate(history):
#         if idx % 2 == 0:
#             text = sent.content.strip()
#             parts.append(f"human: {text}")
#         else:
#             text = sent.content.split('\n\n\n\n\n\n\n\n')[0].strip()
#             parts.append(f"ai: {text}")
#     return " ".join(parts)
#
# class RewriteQuestionChain:
#     def __init__(self, model):
#
#         self.chat_model = model
#
#         self.condense_q_system_prompt = """
# 假设你是极其专业的英语和汉语语言专家。你的任务是：给定一个聊天历史记录和一个可能涉及此聊天历史的用户最新的问题(新问题)，请构造一个不需要聊天历史就能理解的独立且语义完整的问题。
#
# 你可以假设这个问题是在用户与聊天机器人对话的背景下。
#
# instructions:
# - 请始终记住，你的任务是生成独立问题，而不是直接回答新问题！
# - 根据用户的新问题和聊天历史记录，判断新问题是否已经是独立且语义完整的。如果新问题已经独立且完整，直接输出新问题，无需任何改动；否则，你需要对新问题进行改写，使其成为独立问题。
# - 确保问题在重新构造前后语种保持一致。
# - 确保问题在重新构造前后意思保持一致。
# - 在构建独立问题时，尽可能将代词（如"她"、"他们"、"它"等）替换为聊天历史记录中对应的具体的名词或实体引用，以提高问题的明确性和易理解性。
#
# ```
# Example input:
# HumanMessage: `北京明天出门需要带伞吗？`
# AIMessage: `今天北京的天气是全天阴，气温19摄氏度到27摄氏度，因此不需要带伞噢。`
# 新问题: `那后天呢？`  # 问题与上文有关，不独立且语义不完整，需要改写
# Example output: `北京后天出门需要带伞吗？`  # 根据聊天历史改写新问题，使其独立
#
# Example input:
# HumanMessage: `明天北京的天气是多云转晴，适合出门野炊吗？`
# AIMessage: `当然可以，这样的天气非常适合出门野炊呢！不过在出门前最好还是要做好防晒措施噢~`
# 新问题: `那北京哪里适合野炊呢？`  # 问题已经是独立且语义完整的，不需要改写
# Example output: `那北京哪里适合野炊呢？` # 直接返回新问题，不需要改写
# ```
#
# """
#         self.condense_q_prompt = ChatPromptTemplate.from_messages(
#             [
#                 ("system", self.condense_q_system_prompt),
#                 MessagesPlaceholder(variable_name="chat_history"),
#                 ("human", "新问题:{question}\n请构造不需要聊天历史就能理解的独立且语义完整的问题。\n独立问题:"),
#             ]
#         )
#
#         self.condense_q_chain = self.condense_q_prompt | self.chat_model | StrOutputParser()
#
# async def chat(query: str = Body(..., description="用户输入", examples=["恼羞成怒"]),
#                conversation_id: str = Body("", description="对话框ID"),
#                history_len: int = Body(-1, description="从数据库中取历史消息的数量"),
#                history: Union[int, List[History]] = Body([],
#                                                          description="历史对话，设为一个整数可以从数据库中读取历史消息",
#                                                          examples=[[
#                                                              {"role": "user",
#                                                               "content": "我们来玩成语接龙，我先来，生龙活虎"},
#                                                              {"role": "assistant", "content": "虎头虎脑"}]]
#                                                          ),
#                stream: bool = Body(False, description="流式输出"),
#                model_name: str = Body(LLM_MODELS[0], description="LLM 模型名称。"),
#                temperature: float = Body(TEMPERATURE, description="LLM 采样温度", ge=0.0, le=2.0),
#                max_tokens: Optional[int] = Body(None, description="限制LLM生成Token数量，默认None代表模型最大值"),
#                # top_p: float = Body(TOP_P, description="LLM 核采样。勿与temperature同时设置", gt=0.0, lt=1.0),
#                prompt_name: str = Body("default", description="使用的prompt模板名称(在configs/prompt_config.py中配置)"),
#                ):
#     async def chat_iterator() -> AsyncIterable[str]:
#         nonlocal history, max_tokens
#
#         callback = AsyncIteratorCallbackHandler()
#         callbacks = [callback]
#         memory = None
#
#         # 负责保存llm response到message db
#         message_id = add_message_to_db(chat_type="llm_chat", query=query, conversation_id=conversation_id)
#         conversation_callback = ConversationCallbackHandler(conversation_id=conversation_id, message_id=message_id,
#                                                             chat_type="llm_chat",
#                                                             query=query)
#         callbacks.append(conversation_callback)
#
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
#
#         if isinstance(max_tokens, int) and max_tokens <= 0:
#             max_tokens = None
#
#         model = get_ChatOpenAI(
#             model_name=model_name,
#             temperature=temperature,
#             max_tokens=max_tokens,
#             callbacks=callbacks,
#         )
#
#         if history: # 优先使用前端传入的历史消息
#             history = [History.from_data(h) for h in history]
#             prompt_template = get_prompt_template("llm_chat", prompt_name)
#             input_msg = History(role="user", content=prompt_template).to_msg_template(False)
#             chat_prompt = ChatPromptTemplate.from_messages(
#                 [i.to_msg_template() for i in history] + [input_msg])
#         elif conversation_id and history_len > 0: # 前端要求从数据库取历史消息
#             # 使用memory 时必须 prompt 必须含有memory.memory_key 对应的变量
#             prompt = get_prompt_template("llm_chat", "with_history")
#             chat_prompt = PromptTemplate.from_template(prompt)
#             # 根据conversation_id 获取message 列表进而拼凑 memory
#             memory = ConversationBufferDBMemory(conversation_id=conversation_id,
#                                                 llm=model,
#                                                 message_limit=history_len)
#         else:
#             prompt_template = get_prompt_template("llm_chat", prompt_name)
#             input_msg = History(role="user", content=prompt_template).to_msg_template(False)
#             chat_prompt = ChatPromptTemplate.from_messages([input_msg])
#         print(chat_prompt)
#         chain = LLMChain(prompt=chat_prompt, llm=model, memory=memory)
#
#         condense_question_best = query
#         if history:
#             print(history)
#             _history = [History.from_data(h) for h in history]
#             chat_history = [h.to_msg_tuple() for h in _history]
#             print(chat_history)
#             history_message = convert_to_messages(chat_history)
#             history_message_text = []
#             for idx, sent in enumerate(chat_history):
#                 if idx % 2 == 0:
#                     text = sent[1].strip()
#                     history_message_text.append(f"human: {text}")
#                 else:
#                     text = sent[1].split('\n\n\n\n\n\n\n\n')[0].strip()
#                     history_message_text.append(f"ai: {text}")
#             history_message_text = " ".join(history_message_text)
#             history_message_text = f"[{history_message_text}]"
#             print(history_message_text)
#             rewrite_q_chain = RewriteQuestionChain(model)
#             condense_question_list = []
#             for i in range(3):
#                 condense_question = await rewrite_q_chain.condense_q_chain.ainvoke(
#                     {
#                         "chat_history": history_message,
#                         "question": query,
#                     },
#                 )
#                 condense_question_list.append(condense_question)
#                 bm25_scores = [calculate_bm25_score(history_message_text, rq) for rq in condense_question_list]
#                 dense_scores = [calculate_embedding_score(history_message_text, rq) for rq in condense_question_list]
#
#                 weight_bm25 = 0.5
#                 weight_dense = 1
#                 combined_scores = [weight_bm25 * bm25 + weight_dense * dense for bm25, dense in
#                                    zip(bm25_scores, dense_scores)]
#                 # 找出最大分数的索引
#                 max_index = combined_scores.index(max(combined_scores))
#                 condense_question_best = condense_question_list[max_index]
#             # Begin a task that runs in the background.
#             task = asyncio.create_task(wrap_done(
#                 chain.acall({"input": condense_question_best}),
#                 callback.done),
#             )
#             if stream:
#                 async for token in callback.aiter():
#                     # Use server-sent-events to stream the response
#                     yield json.dumps(
#                         {"text": token, "message_id": message_id},
#                         ensure_ascii=False)
#             else:
#                 answer = ""
#                 async for token in callback.aiter():
#                     answer += token
#                 yield json.dumps(
#                     {"text": answer, "message_id": message_id},
#                     ensure_ascii=False)
#
#             await task
#         print(condense_question_best)
#
#         # Begin a task that runs in the background.
#         task = asyncio.create_task(wrap_done(
#             chain.acall({"input": condense_question_best}),
#             callback.done),
#         )
#
#         if stream:
#             async for token in callback.aiter():
#                 # Use server-sent-events to stream the response
#                 yield json.dumps(
#                     {"text": token, "message_id": message_id},
#                     ensure_ascii=False)
#         else:
#             answer = ""
#             async for token in callback.aiter():
#                 answer += token
#             yield json.dumps(
#                 {"text": answer, "message_id": message_id},
#                 ensure_ascii=False)
#
#         await task
#
#     return EventSourceResponse(chat_iterator())

from transformers import AutoTokenizer
from configs import (LLM_MODELS,
                     VECTOR_SEARCH_TOP_K,
                     SCORE_THRESHOLD,
                     TEMPERATURE,
EMBEDDING_KEYWORD_FILE,
MODEL_PATH,
                     # REWRITE_QUERY_PROMPT,
                     USE_RERANKER,
                     RERANKER_MODEL,
                     RERANKER_MAX_LENGTH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH["llm_model"][LLM_MODELS[0]], use_fast=False, trust_remote_code=True)
print(tokenizer)
