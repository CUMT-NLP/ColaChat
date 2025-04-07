from fastapi import Body, Request
from sse_starlette.sse import EventSourceResponse
from fastapi.concurrency import run_in_threadpool
from configs import (LLM_MODELS,
                     VECTOR_SEARCH_TOP_K,
                     SCORE_THRESHOLD,
                     TEMPERATURE,
                     REWRITE_QUERY_PROMPT,
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

            answer = ""
            if stream:
                async for token in callback_knowledge_base.aiter():
                    answer += token
                    yield json.dumps({"answer": token}, ensure_ascii=False)
                yield json.dumps({"docs": source_documents}, ensure_ascii=False)
            else:
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
            recent_history_text = "\n".join([h.content for h in recent_history])

            # 查询改写
            callback_rewrite = AsyncIteratorCallbackHandler()
            callbacks_rewrite = [callback_rewrite]
            message_id_rewrite = add_message_to_db(chat_type="llm_chat", query=query, conversation_id=conversation_id)
            conversation_callback_rewrite = ConversationCallbackHandler(conversation_id=conversation_id,
                                                                        message_id=message_id_rewrite,
                                                                        chat_type="llm_chat",
                                                                        query=query)
            callbacks_rewrite.append(conversation_callback_rewrite)

            if isinstance(max_tokens, int) and max_tokens <= 0:
                max_tokens = None

            model_rewrite = get_ChatOpenAI(
                model_name=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                callbacks=callbacks_rewrite,
            )

            rewrite_prompt_template = get_prompt_template("rewrite", 'default')
            rewrite_llm_chat_prompt = PromptTemplate.from_template(rewrite_prompt_template)
            rewrite_memory = ConversationBufferDBMemory(conversation_id=conversation_id,
                                                        llm=model_rewrite,
                                                        message_limit=history_len)
            chain_llm_query = LLMChain(prompt=rewrite_llm_chat_prompt, llm=model_rewrite, memory=rewrite_memory)
            task = asyncio.create_task(wrap_done(
                chain_llm_query.acall({"recent_history_text": recent_history_text,"query": query}),
                callback_rewrite.done),
            )
            query_rewrite_text = ""
            async for token in callback_rewrite.aiter():
                query_rewrite_text += token
            yield json.dumps({"query_rewrite_text": query_rewrite_text}, ensure_ascii=False)

            await task

            # 召回
            docs = await run_in_threadpool(search_docs,
                                           query=query_rewrite_text ,
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

            answer = ""
            if stream:
                async for token in callback_knowledge_base.aiter():
                    answer += token
                    yield json.dumps({"answer": token}, ensure_ascii=False)
                yield json.dumps({"docs": source_documents}, ensure_ascii=False)
            else:
                async for token in callback_knowledge_base.aiter():
                    answer += token
                yield json.dumps({"final_answer": answer,
                                  "docs": source_documents,
                                  },
                                 ensure_ascii=False)
            await task

    return EventSourceResponse(knowledge_base_chat_iterator(query, top_k, history,model_name,prompt_name))