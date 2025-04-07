# import streamlit as st
# from webui_pages.utils import *
# from streamlit_chatbox import *
# from streamlit_modal import Modal
# from datetime import datetime
# import os
# import re
# import time
# from configs import (TEMPERATURE, HISTORY_LEN, PROMPT_TEMPLATES, LLM_MODELS,
#                      DEFAULT_KNOWLEDGE_BASE, DEFAULT_SEARCH_ENGINE, SUPPORT_AGENT_MODEL)
# from server.knowledge_base.utils import LOADER_DICT
# import uuid
# from typing import List, Dict

# chat_box = ChatBox(
#     assistant_avatar=os.path.join(
#         "img",
#         "chatchat_icon_blue_square_v2.png"
#     )
# )


# def get_messages_history(history_len: int, content_in_expander: bool = False) -> List[Dict]:
#     '''
#     返回消息历史。
#     content_in_expander控制是否返回expander元素中的内容，一般导出的时候可以选上，传入LLM的history不需要
#     '''

#     def filter(msg):
#         content = [x for x in msg["elements"] if x._output_method in ["markdown", "text"]]
#         if not content_in_expander:
#             content = [x for x in content if not x._in_expander]
#         content = [x.content for x in content]

#         return {
#             "role": msg["role"],
#             "content": "\n\n".join(content),
#         }

#     return chat_box.filter_history(history_len=history_len, filter=filter)


# @st.cache_data
# def upload_temp_docs(files, _api: ApiRequest) -> str:
#     '''
#     将文件上传到临时目录，用于文件对话
#     返回临时向量库ID
#     '''
#     return _api.upload_temp_docs(files).get("data", {}).get("id")


# def parse_command(text: str, modal: Modal) -> bool:
#     '''
#     检查用户是否输入了自定义命令，当前支持：
#     /new {session_name}。如果未提供名称，默认为“会话X”
#     /del {session_name}。如果未提供名称，在会话数量>1的情况下，删除当前会话。
#     /clear {session_name}。如果未提供名称，默认清除当前会话
#     /help。查看命令帮助
#     返回值：输入的是命令返回True，否则返回False
#     '''
#     if m := re.match(r"/([^\s]+)\s*(.*)", text):
#         cmd, name = m.groups()
#         name = name.strip()
#         conv_names = chat_box.get_chat_names()
#         if cmd == "help":
#             modal.open()
#         elif cmd == "new":
#             if not name:
#                 i = 1
#                 while True:
#                     name = f"会话{i}"
#                     if name not in conv_names:
#                         break
#                     i += 1
#             if name in st.session_state["conversation_ids"]:
#                 st.error(f"该会话名称 “{name}” 已存在")
#                 time.sleep(1)
#             else:
#                 st.session_state["conversation_ids"][name] = uuid.uuid4().hex
#                 st.session_state["cur_conv_name"] = name
#         elif cmd == "del":
#             name = name or st.session_state.get("cur_conv_name")
#             if len(conv_names) == 1:
#                 st.error("这是最后一个会话，无法删除")
#                 time.sleep(1)
#             elif not name or name not in st.session_state["conversation_ids"]:
#                 st.error(f"无效的会话名称：“{name}”")
#                 time.sleep(1)
#             else:
#                 st.session_state["conversation_ids"].pop(name, None)
#                 chat_box.del_chat_name(name)
#                 st.session_state["cur_conv_name"] = ""
#         elif cmd == "clear":
#             chat_box.reset_history(name=name or None)
#         return True
#     return False


# def dialogue_page(api: ApiRequest, is_lite: bool = False):
#     st.session_state.setdefault("conversation_ids", {})
#     st.session_state["conversation_ids"].setdefault(chat_box.cur_chat_name, uuid.uuid4().hex)
#     st.session_state.setdefault("file_chat_id", None)
#     default_model = api.get_default_llm_model()[0]

#     if not chat_box.chat_inited:
#         st.toast(
#             f"欢迎使用 [`Langchain-Chatchat`](https://github.com/chatchat-space/Langchain-Chatchat) ! \n\n"
#             f"当前运行的模型`{default_model}`, 您可以开始提问了."
#         )
#         chat_box.init_session()

#     # 弹出自定义命令帮助信息
#     modal = Modal("自定义命令", key="cmd_help", max_width="500")
#     if modal.is_open():
#         with modal.container():
#             cmds = [x for x in parse_command.__doc__.split("\n") if x.strip().startswith("/")]
#             st.write("\n\n".join(cmds))

#     with st.sidebar:
#         # 多会话
#         conv_names = list(st.session_state["conversation_ids"].keys())
#         index = 0
#         if st.session_state.get("cur_conv_name") in conv_names:
#             index = conv_names.index(st.session_state.get("cur_conv_name"))
#         conversation_name = st.selectbox("当前会话：", conv_names, index=index)
#         chat_box.use_chat_name(conversation_name)
#         conversation_id = st.session_state["conversation_ids"][conversation_name]

#         def on_mode_change():
#             mode = st.session_state.dialogue_mode
#             text = f"已切换到 {mode} 模式。"
#             if mode == "知识库问答":
#                 cur_kb = st.session_state.get("selected_kb")
#                 if cur_kb:
#                     text = f"{text} 当前知识库： `{cur_kb}`。"
#             st.toast(text)

#         dialogue_modes = ["LLM 对话",
#                           "知识库问答",
#                           "文件对话",
#                           "搜索引擎问答",
#                           "自定义Agent问答",
#                           ]
#         dialogue_mode = st.selectbox("请选择对话模式：",
#                                      dialogue_modes,
#                                      index=0,
#                                      on_change=on_mode_change,
#                                      key="dialogue_mode",
#                                      )

#         def on_llm_change():
#             if llm_model:
#                 config = api.get_model_config(llm_model)
#                 if not config.get("online_api"):  # 只有本地model_worker可以切换模型
#                     st.session_state["prev_llm_model"] = llm_model
#                 st.session_state["cur_llm_model"] = st.session_state.llm_model

#         def llm_model_format_func(x):
#             if x in running_models:
#                 return f"{x} (Running)"
#             return x

#         running_models = list(api.list_running_models())
#         available_models = []
#         config_models = api.list_config_models()
#         if not is_lite:
#             for k, v in config_models.get("local", {}).items():
#                 if (v.get("model_path_exists")
#                         and k not in running_models):
#                     available_models.append(k)
#         for k, v in config_models.get("online", {}).items():
#             if not v.get("provider") and k not in running_models and k in LLM_MODELS:
#                 available_models.append(k)
#         llm_models = running_models + available_models
#         cur_llm_model = st.session_state.get("cur_llm_model", default_model)
#         if cur_llm_model in llm_models:
#             index = llm_models.index(cur_llm_model)
#         else:
#             index = 0
#         llm_model = st.selectbox("选择LLM模型：",
#                                  llm_models,
#                                  index,
#                                  format_func=llm_model_format_func,
#                                  on_change=on_llm_change,
#                                  key="llm_model",
#                                  )
#         if (st.session_state.get("prev_llm_model") != llm_model
#                 and not is_lite
#                 and not llm_model in config_models.get("online", {})
#                 and not llm_model in config_models.get("langchain", {})
#                 and llm_model not in running_models):
#             with st.spinner(f"正在加载模型： {llm_model}，请勿进行操作或刷新页面"):
#                 prev_model = st.session_state.get("prev_llm_model")
#                 r = api.change_llm_model(prev_model, llm_model)
#                 if msg := check_error_msg(r):
#                     st.error(msg)
#                 elif msg := check_success_msg(r):
#                     st.success(msg)
#                     st.session_state["prev_llm_model"] = llm_model

#         index_prompt = {
#             "LLM 对话": "llm_chat",
#             "自定义Agent问答": "agent_chat",
#             "搜索引擎问答": "search_engine_chat",
#             "知识库问答": "knowledge_base_chat",
#             "文件对话": "knowledge_base_chat",
#         }
#         prompt_templates_kb_list = list(PROMPT_TEMPLATES[index_prompt[dialogue_mode]].keys())
#         prompt_template_name = prompt_templates_kb_list[0]
#         if "prompt_template_select" not in st.session_state:
#             st.session_state.prompt_template_select = prompt_templates_kb_list[0]

#         def prompt_change():
#             text = f"已切换为 {prompt_template_name} 模板。"
#             st.toast(text)

#         prompt_template_select = st.selectbox(
#             "请选择Prompt模板：",
#             prompt_templates_kb_list,
#             index=0,
#             on_change=prompt_change,
#             key="prompt_template_select",
#         )
#         prompt_template_name = st.session_state.prompt_template_select
#         temperature = st.slider("Temperature：", 0.0, 2.0, TEMPERATURE, 0.05)
#         history_len = st.number_input("历史对话轮数：", 0, 20, HISTORY_LEN)

#         def on_kb_change():
#             st.toast(f"已加载知识库： {st.session_state.selected_kb}")

#         if dialogue_mode == "知识库问答":
#             with st.expander("知识库配置", True):
#                 kb_list = api.list_knowledge_bases()
#                 index = 0
#                 if DEFAULT_KNOWLEDGE_BASE in kb_list:
#                     index = kb_list.index(DEFAULT_KNOWLEDGE_BASE)
#                 selected_kb = st.selectbox(
#                     "请选择知识库：",
#                     kb_list,
#                     index=index,
#                     on_change=on_kb_change,
#                     key="selected_kb",
#                 )
#                 kb_top_k = st.number_input("匹配知识条数：", 1, 20, VECTOR_SEARCH_TOP_K)

#                 ## Bge 模型会超过1
#                 score_threshold = st.slider("知识匹配分数阈值：", 0.0, 2.0, float(SCORE_THRESHOLD), 0.01)
#         elif dialogue_mode == "文件对话":
#             with st.expander("文件对话配置", True):
#                 files = st.file_uploader("上传知识文件：",
#                                          [i for ls in LOADER_DICT.values() for i in ls],
#                                          accept_multiple_files=True,
#                                          )
#                 kb_top_k = st.number_input("匹配知识条数：", 1, 20, VECTOR_SEARCH_TOP_K)

#                 ## Bge 模型会超过1
#                 score_threshold = st.slider("知识匹配分数阈值：", 0.0, 2.0, float(SCORE_THRESHOLD), 0.01)
#                 if st.button("开始上传", disabled=len(files) == 0):
#                     st.session_state["file_chat_id"] = upload_temp_docs(files, api)
#         elif dialogue_mode == "搜索引擎问答":
#             search_engine_list = api.list_search_engines()
#             if DEFAULT_SEARCH_ENGINE in search_engine_list:
#                 index = search_engine_list.index(DEFAULT_SEARCH_ENGINE)
#             else:
#                 index = search_engine_list.index("duckduckgo") if "duckduckgo" in search_engine_list else 0
#             with st.expander("搜索引擎配置", True):
#                 search_engine = st.selectbox(
#                     label="请选择搜索引擎",
#                     options=search_engine_list,
#                     index=index,
#                 )
#                 se_top_k = st.number_input("匹配搜索结果条数：", 1, 20, SEARCH_ENGINE_TOP_K)

#     # Display chat messages from history on app rerun
#     chat_box.output_messages()

#     chat_input_placeholder = "请输入对话内容，换行请使用Shift+Enter。输入/help查看自定义命令 "

#     def on_feedback(
#             feedback,
#             message_id: str = "",
#             history_index: int = -1,
#     ):
#         reason = feedback["text"]
#         score_int = chat_box.set_feedback(feedback=feedback, history_index=history_index)
#         api.chat_feedback(message_id=message_id,
#                           score=score_int,
#                           reason=reason)
#         st.session_state["need_rerun"] = True

#     feedback_kwargs = {
#         "feedback_type": "thumbs",
#         "optional_text_label": "欢迎反馈您打分的理由",
#     }

#     if prompt := st.chat_input(chat_input_placeholder, key="prompt"):
#         if parse_command(text=prompt, modal=modal):  # 用户输入自定义命令
#             st.rerun()
#         else:
#             history = get_messages_history(history_len)
#             chat_box.user_say(prompt)
#             if dialogue_mode == "LLM 对话":
#                 chat_box.ai_say("正在思考...")
#                 text = ""
#                 message_id = ""
#                 r = api.chat_chat(prompt,
#                                   history=history,
#                                   conversation_id=conversation_id,
#                                   model=llm_model,
#                                   prompt_name=prompt_template_name,
#                                   temperature=temperature)
#                 for t in r:
#                     if error_msg := check_error_msg(t):  # check whether error occured
#                         st.error(error_msg)
#                         break
#                     text += t.get("text", "")
#                     chat_box.update_msg(text)
#                     message_id = t.get("message_id", "")

#                 metadata = {
#                     "message_id": message_id,
#                 }
#                 chat_box.update_msg(text, streaming=False, metadata=metadata)  # 更新最终的字符串，去除光标
#                 chat_box.show_feedback(**feedback_kwargs,
#                                        key=message_id,
#                                        on_submit=on_feedback,
#                                        kwargs={"message_id": message_id, "history_index": len(chat_box.history) - 1})

#             elif dialogue_mode == "自定义Agent问答":
#                 if not any(agent in llm_model for agent in SUPPORT_AGENT_MODEL):
#                     chat_box.ai_say([
#                         f"正在思考... \n\n <span style='color:red'>该模型并没有进行Agent对齐，请更换支持Agent的模型获得更好的体验！</span>\n\n\n",
#                         Markdown("...", in_expander=True, title="思考过程", state="complete"),

#                     ])
#                 else:
#                     chat_box.ai_say([
#                         f"正在思考...",
#                         Markdown("...", in_expander=True, title="思考过程", state="complete"),

#                     ])
#                 text = ""
#                 ans = ""
#                 for d in api.agent_chat(prompt,
#                                         history=history,
#                                         model=llm_model,
#                                         prompt_name=prompt_template_name,
#                                         temperature=temperature,
#                                         ):
#                     try:
#                         d = json.loads(d)
#                     except:
#                         pass
#                     if error_msg := check_error_msg(d):  # check whether error occured
#                         st.error(error_msg)
#                     if chunk := d.get("answer"):
#                         text += chunk
#                         chat_box.update_msg(text, element_index=1)
#                     if chunk := d.get("final_answer"):
#                         ans += chunk
#                         chat_box.update_msg(ans, element_index=0)
#                     if chunk := d.get("tools"):
#                         text += "\n\n".join(d.get("tools", []))
#                         chat_box.update_msg(text, element_index=1)
#                 chat_box.update_msg(ans, element_index=0, streaming=False)
#                 chat_box.update_msg(text, element_index=1, streaming=False)
#             elif dialogue_mode == "知识库问答":
#                 chat_box.ai_say([
#                     f"正在查询知识库 `{selected_kb}` ...",
#                     Markdown("...", in_expander=True, title="知识库匹配结果", state="complete"),
#                 ])
#                 text = ""
#                 for d in api.knowledge_base_chat(prompt,
#                                                  knowledge_base_name=selected_kb,
#                                                  top_k=kb_top_k,
#                                                  score_threshold=score_threshold,
#                                                  history=history,
#                                                  model=llm_model,
#                                                  prompt_name=prompt_template_name,
#                                                  temperature=temperature):
#                     if error_msg := check_error_msg(d):  # check whether error occured
#                         st.error(error_msg)
#                     elif chunk := d.get("answer"):
#                         text += chunk
#                         chat_box.update_msg(text, element_index=0)
#                 chat_box.update_msg(text, element_index=0, streaming=False)
#                 chat_box.update_msg("\n\n".join(d.get("docs", [])), element_index=1, streaming=False)
#             elif dialogue_mode == "文件对话":
#                 if st.session_state["file_chat_id"] is None:
#                     st.error("请先上传文件再进行对话")
#                     st.stop()
#                 chat_box.ai_say([
#                     f"正在查询文件 `{st.session_state['file_chat_id']}` ...",
#                     Markdown("...", in_expander=True, title="文件匹配结果", state="complete"),
#                 ])
#                 text = ""
#                 for d in api.file_chat(prompt,
#                                        knowledge_id=st.session_state["file_chat_id"],
#                                        top_k=kb_top_k,
#                                        score_threshold=score_threshold,
#                                        history=history,
#                                        model=llm_model,
#                                        prompt_name=prompt_template_name,
#                                        temperature=temperature):
#                     if error_msg := check_error_msg(d):  # check whether error occured
#                         st.error(error_msg)
#                     elif chunk := d.get("answer"):
#                         text += chunk
#                         chat_box.update_msg(text, element_index=0)
#                 chat_box.update_msg(text, element_index=0, streaming=False)
#                 chat_box.update_msg("\n\n".join(d.get("docs", [])), element_index=1, streaming=False)
#             elif dialogue_mode == "搜索引擎问答":
#                 chat_box.ai_say([
#                     f"正在执行 `{search_engine}` 搜索...",
#                     Markdown("...", in_expander=True, title="网络搜索结果", state="complete"),
#                 ])
#                 text = ""
#                 for d in api.search_engine_chat(prompt,
#                                                 search_engine_name=search_engine,
#                                                 top_k=se_top_k,
#                                                 history=history,
#                                                 model=llm_model,
#                                                 prompt_name=prompt_template_name,
#                                                 temperature=temperature,
#                                                 split_result=se_top_k > 1):
#                     if error_msg := check_error_msg(d):  # check whether error occured
#                         st.error(error_msg)
#                     elif chunk := d.get("answer"):
#                         text += chunk
#                         chat_box.update_msg(text, element_index=0)
#                 chat_box.update_msg(text, element_index=0, streaming=False)
#                 chat_box.update_msg("\n\n".join(d.get("docs", [])), element_index=1, streaming=False)

#     if st.session_state.get("need_rerun"):
#         st.session_state["need_rerun"] = False
#         st.rerun()

#     now = datetime.now()
#     with st.sidebar:

#         cols = st.columns(2)
#         export_btn = cols[0]
#         if cols[1].button(
#                 "清空对话",
#                 use_container_width=True,
#         ):
#             chat_box.reset_history()
#             st.rerun()

#     export_btn.download_button(
#         "导出记录",
#         "".join(chat_box.export2md()),
#         file_name=f"{now:%Y-%m-%d %H.%M}_对话记录.md",
#         mime="text/markdown",
#         use_container_width=True,
#     )


import streamlit as st
from webui_pages.utils import *
from streamlit_chatbox import *
from streamlit_modal import Modal
from datetime import datetime
import os
import re
import time
from configs import (TEMPERATURE, HISTORY_LEN, PROMPT_TEMPLATES, LLM_MODELS,
                     DEFAULT_KNOWLEDGE_BASE, DEFAULT_SEARCH_ENGINE, SUPPORT_AGENT_MODEL)
from server.knowledge_base.utils import LOADER_DICT
import uuid
from typing import List, Dict
from openai import OpenAI

chat_box = ChatBox(
    assistant_avatar=os.path.join(
        "img",
        "AI.jpg"
    )
)
##  chatchat_icon_blue_square_v2.png

def get_messages_history(history_len: int, content_in_expander: bool = False) -> List[Dict]:
    '''
    返回消息历史。
    content_in_expander控制是否返回expander元素中的内容，一般导出的时候可以选上，传入LLM的history不需要
    '''

    def filter(msg):
        content = [x for x in msg["elements"] if x._output_method in ["markdown", "text"]]
        if not content_in_expander:
            content = [x for x in content if not x._in_expander]
        content = [x.content for x in content]

        return {
            "role": msg["role"],
            "content": "\n\n".join(content),
        }

    return chat_box.filter_history(history_len=history_len, filter=filter)


@st.cache_data
def upload_temp_docs(files, _api: ApiRequest) -> str:
    '''
    将文件上传到临时目录，用于文件对话
    返回临时向量库ID
    '''
    return _api.upload_temp_docs(files).get("data", {}).get("id")


def parse_command(text: str, modal: Modal) -> bool:
    '''
    检查用户是否输入了自定义命令，当前支持：
    /new {session_name}。如果未提供名称，默认为“会话X”
    /del {session_name}。如果未提供名称，在会话数量>1的情况下，删除当前会话。
    /clear {session_name}。如果未提供名称，默认清除当前会话
    /help。查看命令帮助
    返回值：输入的是命令返回True，否则返回False
    '''
    if m := re.match(r"/([^\s]+)\s*(.*)", text):
        cmd, name = m.groups()
        name = name.strip()
        conv_names = chat_box.get_chat_names()
        if cmd == "help":
            modal.open()
        elif cmd == "new":
            if not name:
                i = 1
                while True:
                    name = f"会话{i}"
                    if name not in conv_names:
                        break
                    i += 1
            if name in st.session_state["conversation_ids"]:
                st.error(f"该会话名称 “{name}” 已存在")
                time.sleep(1)
            else:
                st.session_state["conversation_ids"][name] = uuid.uuid4().hex
                st.session_state["cur_conv_name"] = name
        elif cmd == "del":
            name = name or st.session_state.get("cur_conv_name")
            if len(conv_names) == 1:
                st.error("这是最后一个会话，无法删除")
                time.sleep(1)
            elif not name or name not in st.session_state["conversation_ids"]:
                st.error(f"无效的会话名称：“{name}”")
                time.sleep(1)
            else:
                st.session_state["conversation_ids"].pop(name, None)
                chat_box.del_chat_name(name)
                st.session_state["cur_conv_name"] = ""
        elif cmd == "clear":
            chat_box.reset_history(name=name or None)
        return True
    return False


def dialogue_page(api: ApiRequest, is_lite: bool = False):
    st.session_state.setdefault("conversation_ids", {})
    st.session_state["conversation_ids"].setdefault(chat_box.cur_chat_name, uuid.uuid4().hex)
    st.session_state.setdefault("file_chat_id", None)
    default_model = api.get_default_llm_model()[0]

    if not chat_box.chat_inited:
        st.toast(
            f"欢迎使用coalchat v1 版本  \n\n"
            f"当前运行的模型为`{default_model}`, 祝您使用愉快."
        )
        chat_box.init_session()

    # 弹出自定义命令帮助信息
    modal = Modal("自定义命令", key="cmd_help", max_width="500")
    if modal.is_open():
        with modal.container():
            cmds = [x for x in parse_command.__doc__.split("\n") if x.strip().startswith("/")]
            st.write("\n\n".join(cmds))

    with st.sidebar:
        # 多会话
        conv_names = list(st.session_state["conversation_ids"].keys())       # 从Streamlit应用的会话状态中获取所有已保存的会话标识符，并转化为一个列表
        index = 0            # 初始化索引为0，用于默认选择列表中的第一个会话
        if st.session_state.get("cur_conv_name") in conv_names:              # 检查当前激活的会话名称是否在所有会话名称列表中存在
            index = conv_names.index(st.session_state.get("cur_conv_name"))     # 如果存在，则获取该会话在列表中的位置，以便在下拉菜单中预先选中它
        conversation_name = st.selectbox("当前会话：", conv_names, index=index , help=("输入/new新建对话，/del删除对话"))  # 使用Streamlit的selectbox组件创建一个多选菜单让用户选择会话，
        chat_box.use_chat_name(conversation_name)                # 调用chat_box对象的use_chat_name方法，传入用户选择的会话名称，
        conversation_id = st.session_state["conversation_ids"][conversation_name]

        def on_mode_change():
            mode = st.session_state.dialogue_mode           # 从Streamlit的应用会话状态中获取当前选择的对话模式
            text = f"已切换到 {mode} 模式。"
            if mode == "知识库问答":                      # 如果切换到了“知识库问答”模式
                cur_kb = st.session_state.get("selected_kb")             # 尝试获取当前选中的知识库名称
                if cur_kb:      #   如果知识库名称存在（即用户已选择）
                    text = f"{text} 当前知识库： `{cur_kb}`。"
            st.toast(text)

        dialogue_modes = ["LLM 对话",
                          "知识库问答",
                          ]
        ###################################################################################################
        dialogue_mode = st.selectbox("对话模式：",          ################ 在Streamlit界面中创建一个下拉选择框让用户选择对话模式  ################
                                     dialogue_modes,
                                     index=0,
                                     on_change=on_mode_change,          # 使用on_change参数绑定了上面定义的on_mode_change函数，当用户更改选项时自动调用
                                     help="选择纯大模型对话或是带RAG检索对话",
                                     key="dialogue_mode",               # 设置初始选中索引为0（即列表中的第一个选项），并为该组件分配一个唯一key
                                     )




        ###################################################################################################

        def on_llm_change():
            if llm_model:
                config = api.get_model_config(llm_model)
                if not config.get("online_api"):  # 只有本地model_worker可以切换模型
                    st.session_state["prev_llm_model"] = llm_model
                st.session_state["cur_llm_model"] = st.session_state.llm_model

        # 定义一个函数，用于格式化显示语言模型的名称
        # 如果传入的模型名称x在running_models列表中，即该模型正在运行：
        def llm_model_format_func(x):
            if x in running_models:
                return f"{x} (Running)"     # 返回模型名称后附加 "(Running)" 标记，表明该模型当前是活动状态
            return x
#######################################################################################

        # running_models = list(api.list_running_models())
        # 使用条件表达式来设置 running_models，如果返回值是 None，则使用空列表
        running_models = list(api.list_running_models()) if api.list_running_models() is not None else []
        #available_models = ["coalchat"]
#######################################################################################
        available_models = []
        config_models = api.list_config_models() if api.list_config_models() is not None else {}           # 获取配置中的所有模型信息
        if not is_lite:         # 非精简模式下执行以下逻辑
            for k, v in config_models.get("local", {}).items():     # 遍历配置中的本地模型，检查模型路径是否存在且模型未在运行中
                if (v.get("model_path_exists")
                        and k not in running_models):            # 符合条件的本地模型添加到可选模型列表
                    available_models.append(k)
        for k, v in config_models.get("online", {}).items():        # 遍历配置中的在线模型，检查模型没有指定提供商、不在运行中且属于预定义的LLM_MODELS列表
            if not v.get("provider") and k not in running_models and k in LLM_MODELS:
                available_models.append(k)       # 符合条件的在线模型也添加到可选模型列表
        llm_models = running_models + available_models
        cur_llm_model = st.session_state.get("cur_llm_model", default_model)        # 获取当前会话状态中设置的模型，如果没有则使用默认模型
        if cur_llm_model in llm_models:
            index = llm_models.index(cur_llm_model)         # 如果当前模型在模型列表中，则找到其索引；否则索引设为0，以便默认选中列表中的第一个模型
        else:
            index = 0

        # 使用Streamlit创建一个下拉选择框让用户选择语言模型
        # 选项为合并后的模型列表，初始选中项为当前模型或列表的第一个模型
        # 使用自定义的格式化函数显示模型状态，并绑定模型变更时的回调函数
        # 组件被赋予唯一的key以区分状态
        llm_model = st.selectbox("LLM模型：",
                                 llm_models,
                                 index,
                                 format_func=llm_model_format_func,
                                 on_change=on_llm_change,
                                 key="llm_model",
                                 help=("所要提问的大模型")
                                 )
        if (st.session_state.get("prev_llm_model") != llm_model             # 检查是否需要切换模型：
                and not is_lite
                and not llm_model in config_models.get("online", {})
                and not llm_model in config_models.get("langchain", {})
                and llm_model not in running_models):
            with st.spinner(f"正在加载模型： {llm_model}，请勿进行操作或刷新页面"):            # 使用st.spinner显示加载提示
                prev_model = st.session_state.get("prev_llm_model")          # 获取上一个模型名称
                r = api.change_llm_model(prev_model, llm_model)                 # 调用API接口来切换语言模型
                if msg := check_error_msg(r):
                    st.error(msg)
                elif msg := check_success_msg(r):
                    st.success(msg)
                    st.session_state["prev_llm_model"] = llm_model

        index_prompt = {
            "LLM 对话": "llm_chat",
            "知识库问答": "knowledge_base_chat",

        }
        # index_prompt = {
        #     "LLM 对话": "llm_chat",
        #     "自定义Agent问答": "agent_chat",
        #     "搜索引擎问答": "search_engine_chat",
        #     "知识库问答": "knowledge_base_chat",
        #     "文件对话": "knowledge_base_chat",
        # }

        # 根据当前选择的对话模式（dialogue_mode），从PROMPT_TEMPLATES字典中获取对应的Prompt模板类别，
        # 然后提取这些模板的键（即模板名称）并转换为列表
        prompt_templates_kb_list = list(PROMPT_TEMPLATES[index_prompt[dialogue_mode]].keys())

        prompt_template_name = prompt_templates_kb_list[0]      # 设置默认的Prompt模板名称为列表中的第一个模板
        if "prompt_template_select" not in st.session_state:
            st.session_state.prompt_template_select = prompt_templates_kb_list[0]

        def prompt_change():        # 定义一个函数，用于在Prompt模板切换时通知用户
            text = f"已切换为 {prompt_template_name} 模板。"
            st.toast(text)

        prompt_template_select = st.selectbox(
            "请选择Prompt模板：",
            prompt_templates_kb_list,
            index=0,
            on_change=prompt_change,
            key="prompt_template_select",
            help=("提问时采用的prompt模版")

        )
        prompt_template_name = st.session_state.prompt_template_select
        temperature = st.slider("Temperature：", 0.0, 2.0, TEMPERATURE, 0.05)            # 添加滑块让用户调整Temperature（用于控制生成文本的随机性和创造性）
        ##################################################################
        # history_len = st.number_input("历史对话轮数：", 0, 20, HISTORY_LEN)
        history_len = 5
        #################################################################

        def on_kb_change():         # 定义一个函数，在知识库更改时通知用户
            st.toast(f"已加载知识库： {st.session_state.selected_kb}")

        if dialogue_mode == "知识库问答":
            with st.expander("知识库配置", True):         # 当用户选择了"知识库问答"的对话模式时，显示以下配置选项   ： 创建一个可折叠的区块，标题为"知识库配置"，默认展开状态(True)
                kb_list = api.list_knowledge_bases()        # 调用api获取当前可用的知识库列表
                index = 0
                if DEFAULT_KNOWLEDGE_BASE in kb_list:       # 如果有预设的DEFAULT_KNOWLEDGE_BASE且存在于kb_list中，找到其索引位置
                    index = kb_list.index(DEFAULT_KNOWLEDGE_BASE)
                selected_kb = st.selectbox(
                    "请选择知识库：",
                    kb_list,             # 可供选择的知识库列表
                    index=index,
                    on_change=on_kb_change,         # 当选择的知识库改变时，调用此函数处理变化
                    key="selected_kb",
                )
                # 创建下拉选择框让用户选择知识库
                kb_top_k = st.number_input("匹配知识条数：", 1, 10, VECTOR_SEARCH_TOP_K)    # 添加数字输入框，让用户设定想要查询到的知识条目数，范围1-10，初始值为VECTOR_SEARCH_TOP_K

                ## Bge 模型会超过1
                score_threshold = st.slider("知识匹配分数阈值：", 0.0, 2.0, float(SCORE_THRESHOLD), 0.01)        # 创建滑块让用户设置知识匹配的最低分数阈值，范围0.0到2.0

    chat_box.output_messages()          # 调用chat_box的output_messages方法，用于在界面上展示之前的对话消息记录。

    chat_input_placeholder = "请输入对话内容，例如：如何观测新凿立井的涌水量？"        # 定义一个字符串，作为用户在输入框中输入对话内容时的占位提示文本。

    def on_feedback(
            feedback,
            message_id: str = "",
            history_index: int = -1,

    ):
        reason = feedback["text"]
        score_int = chat_box.set_feedback(feedback=feedback, history_index=history_index)
        api.chat_feedback(message_id=message_id,
                          score=score_int,
                          reason=reason)
        st.session_state["need_rerun"] = True
        feedback_data = {
            "message_id": message_id,
            "意见（0为赞同1为否认）": score_int,
            "反馈原因": reason,

        }
        additional_data = {
            "question": prompt,
            "answer": text
        }
        with open(r'E:\coalchat\web\反馈备忘录\Feedback.txt', 'a', encoding='utf-8') as file:
            #file.write(f"message_id: {feedback_data['message_id']}\n")
            file.write("------------------------------------------------------------------------------------------------------------------------------------------------------------")
            file.write(f"\n意见（0为赞同1为否认）: {feedback_data['意见（0为赞同1为否认）']}\n")
            file.write(f"\n反馈原因: {feedback_data['反馈原因']}\n")
            file.write(f"question: {additional_data['question']}\n\n\n")
            file.write(f"answer: {additional_data['answer']}\n")
            file.write("\n\n\n------------------------------------------------------------------------------------------------------------------------------------------------------------\n")


    #######################################↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓##############################################
    def on_feedback_doc(
            feedback,
            message_id: str = "",
            history_index: int = -1,

    ):
        reason = feedback["text"]
        score_int = chat_box.set_feedback(feedback=feedback, history_index=history_index)
        api.chat_feedback(message_id=message_id,
                          score=score_int,
                          reason=reason)
        st.session_state["need_rerun"] = True
        feedback_data = {
            "message_id": message_id,
            "意见（0为赞同1为否认）": score_int,
            "反馈原因": reason,

        }
        additional_data = {
            "question": prompt,
            "answer": text,
            "referencr":docs_content
        }
        with open('E:/coalchat-v1/Feedback.txt', 'a', encoding='utf-8') as file:
            #file.write(f"message_id: {feedback_data['message_id']}\n")
            file.write("------------------------------------------------------------------------------------------------------------------------------------------------------------")
            file.write(f"\n意见（0为赞同1为否认）: {feedback_data['意见（0为赞同1为否认）']}\n")
            file.write(f"\n反馈原因: {feedback_data['反馈原因']}\n")
            file.write(f"question: {additional_data['question']}\n\n\n")
            file.write(f"answer: {additional_data['answer']}\n")
            file.write(f"referencr: {additional_data['referencr']}\n")
            file.write("\n\n\n------------------------------------------------------------------------------------------------------------------------------------------------------------\n")

    # # 预设的问题列表
    # preset_questions = ["如何观测新凿立井的涌水量？", "如何提高生产效率？", "如何确保安全生产？"]
    #
    # def get_recommendations(input_text, questions):
    #     recommendations = [q for q in questions if input_text in q]
    #     return recommendations
    #
    # # 显示输入框
    # user_input = st.text_input("请输入对话内容，例如：如何观测新凿立井的涌水量？", key="pro")
    #
    # # 获取推荐的内容
    # recommendations = get_recommendations(user_input, preset_questions)
    #
    # # 显示推荐的内容
    # if recommendations:
    #     st.write("推荐的问题:")
    #     for rec in recommendations:
    #         if st.button(rec):
    #             user_input = rec
    #             st.session_state["pro"] = user_input
    #
    # # 最终处理逻辑
    # if user_input:
    #     st.write(f"User has sent the following prompt: {user_input}")

    #########################################↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑#############################################################
    feedback_kwargs = {             # 定义一个字典feedback_kwargs，用来定制反馈组件的外观或行为
        "feedback_type": "thumbs",
        "optional_text_label": "欢迎反馈您打分的理由",
    }
    timestamp = float(time.time() * 1000)  # 获取当前时间戳，精确到毫秒
    if prompt := st.chat_input(chat_input_placeholder, key="prompt"):        # 用户通过聊天界面输入内容
        #st.write(f"User has sent the following prompt: {prompt}")
        if parse_command(text=prompt, modal=modal):  # 用户输入自定义命令
            st.rerun()
        else:
            history = get_messages_history(history_len)         # 获取历史对话记录
            chat_box.user_say(prompt)                    # 显示用户提问
            if dialogue_mode == "LLM 对话":
                chat_box.ai_say(["...ai思考中...",Markdown("...",  title="推荐问题")])
                text = ""
                message_id = ""                         # 发起API请求获取AI回复
                r = api.chat_chat(prompt,
                                  history=history,
                                  conversation_id=conversation_id,
                                  model=llm_model,
                                  prompt_name=prompt_template_name,
                                  temperature=temperature)
                for t in r:
                    if error_msg := check_error_msg(t):  # check whether error occured
                        st.error(error_msg)
                        break
                    text += t.get("text", "")
                    chat_box.update_msg(text, element_index=0)
                    message_id = t.get("message_id", "")
                # 更新最终消息，移除光标效果，附带反馈功能
                metadata = {
                    "message_id": message_id,
                }

                # client = OpenAI(api_key="sk-6beaca2b483a46b89f640d12d2c29abc", base_url="https://api.deepseek.com")
                # response = client.chat.completions.create(
                #     model="deepseek-chat",
                #     messages=[
                #         {"role": "system", "content": "你是一个阅读理解专家"},
                #         {"role": "user", "content": f"根据我给出的内容提炼出3个相关的问题，3个问题要大概能涵盖到我所给出的所有内容，下面是我给出的内容：{text}"},
                #     ],
                #     stream=False
                # )
                #
                # # 获取API返回的建议
                # advice_llm = response.choices[0].message.content.strip()
                # additional_prompt = "\n\n\n\n\n\n**您还可以问:**\n\n\n\n\n"
                # advice_llm = f"{additional_prompt}{advice_llm}"


                chat_box.update_msg(text, streaming=False, metadata=metadata,element_index=0)  # 更新最终的字符串，去除光标

                # chat_box.show_feedback(**feedback_kwargs,
                #                        key=timestamp,
                #                        on_submit=on_feedback,
                #                        kwargs={"message_id": message_id, "history_index": len(chat_box.history) - 1})
                chat_box.show_feedback(**feedback_kwargs,
                                       key=message_id,
                                       on_submit=on_feedback,
                                       kwargs={"message_id": message_id, "history_index": len(chat_box.history) - 1})
                # chat_box.update_msg(advice_llm, element_index=1, streaming=False)
            elif dialogue_mode == "知识库问答":
                chat_box.ai_say([                # 显示正在查询知识库的信息
                    f"正在查询知识库 `{selected_kb}` ...",
                    Markdown("...", in_expander=True, title="知识库匹配结果", state="complete"),
                    Markdown("查询重写...", in_expander=True, title="查询重写结果", state="complete"),
                    # Markdown("...",  title="推荐问题")
                ])
                text = ""
                rewrite_query_text = ''
                all_rewrites = ''
                message_id = ""
                # 发起API请求查询知识库
                for d in api.knowledge_base_chat(prompt,
                                                 knowledge_base_name=selected_kb,
                                                 top_k=kb_top_k,
                                                 score_threshold=score_threshold,
                                                 history=history,
                                                 model=llm_model,
                                                 prompt_name=prompt_template_name,
                                                 temperature=temperature):
                    if error_msg := check_error_msg(d):  # check whether error occured
                        st.error(error_msg)
                    elif chunk := d.get("answer"):      # 更新答案信息
                        text += chunk
                        chat_box.update_msg(text, element_index=0)
                    elif chunk := d.get("query_rewrite_text"):
                        rewrite_query_text += chunk
                        chat_box.update_msg(rewrite_query_text, element_index=2)

                        ################################
                        #message_id = d.get("message_id", "")
                ####################################################################################################
                # # 最终更新完整回答，包括文档匹配结果
                # client = OpenAI(api_key="sk-6beaca2b483a46b89f640d12d2c29abc", base_url="https://api.deepseek.com")
                # response = client.chat.completions.create(
                #     model="deepseek-chat",
                #     messages=[
                #         {"role": "system", "content": "你是一个煤矿领域专家"},
                #         {"role": "user", "content": f"根据我给出的内容提炼出3个与煤矿领域相关的问题，3个问题要大概能涵盖到我所给出的所有内容，下面是我给出的内容：{text}"},
                #     ],
                #     stream=False
                # )
                #
                # # 获取API返回的建议
                # advice = response.choices[0].message.content.strip()
                # additional_prompt = "\n\n\n\n\n\n**您还可以问:**\n\n\n\n\n"
                # advice = f"{additional_prompt}{advice}"
                # #将 message_id 和当前时间戳结合起来生成唯一的 key

                #unique_key = f"{message_id}_{timestamp}" if message_id else f"{timestamp}"
                chat_box.update_msg(text, element_index=0, streaming=False)

                # chat_box.update_msg(text, streaming=False, metadata=metadata)  # 更新最终的字符串，去除光标
                # chat_box.show_feedback(**feedback_kwargs,
                #                        key=timestamp,
                #                        on_submit=on_feedback_doc,
                #                        kwargs={"message_id": timestamp, "history_index": len(chat_box.history) - 1})
                chat_box.update_msg("\n\n".join(d.get("docs", [])), element_index=1, streaming=False)
                chat_box.update_msg(rewrite_query_text, element_index=2, streaming=False)

                # chat_box.update_msg(advice, element_index=2, streaming=False)
                ####################################↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑##########################################


    # 检查会话状态中是否设置了"need_rerun"标志为True，如果是，则复位该标志并重新运行整个应用程序界面，以响应之前设置的更新需求。
    if st.session_state.get("need_rerun"):
        st.session_state["need_rerun"] = False
        st.rerun()
        # 设置 Session State 的初始值
    if 'show_modal' not in st.session_state:
        st.session_state.show_modal = False

    now = datetime.now()        # 获取当前时间，用于命名导出的文件。


        # 创建 Modal 实例
    GUANY = Modal(title="", key="guany",max_width=1100)              #,max_width=1000
    GONGNENG = Modal(title="", key="gongneng",max_width=1100)


    with st.sidebar:        # 在侧边栏开始一个新的布局。


        cols = st.columns(1)        # 创建两列布局，为接下来的按钮分配空间。
        export_btn = cols[0]            # 第一列分配给“导出记录”按钮变量 第二列放置一个“清空对话”按钮
        if cols[0].button(
                "清空对话",
                use_container_width=True,
        ):
            chat_box.reset_history()      # 将调用chat_box的reset_history方法清除对话历史，并重新运行应用以反映更新。
            st.rerun()
        export_btn.download_button(
                "导出记录",
                "".join(chat_box.export2md()),
                file_name=f"{now:%Y-%m-%d %H.%M}_对话记录.md",
                mime="text/markdown",
                use_container_width=True,
            )
        # 插入几个空白行或Markdown文本来调整布局
        st.text("")
        # st.text("")
        # st.text("")
        # st.text("")
        # st.text("")
        col1,col2 =st.columns([1,1])      ##比例
        if col1.button(" 关于 ",help= "关于该系统的大概说明。"):
            # st.session_state.show_modal = not st.session_state.show_modal  # 切换弹窗显示状态
            if GUANY.is_open():
                GUANY.close()
            else:
                if GONGNENG.is_open():
                    GONGNENG.close()
                    GUANY.open()
                else:
                    GUANY.open()
        if col2.button("功能示例",help="该系统主要能实现的功能。"):
            if GONGNENG.is_open():
                GONGNENG.close()
            else:
                if GUANY.is_open():
                    GUANY.close()
                    GONGNENG.open()
                else:
                    GONGNENG.open()
        #if col3.button("使用说明"):
            #pass

############功能弹窗设置###############功能弹窗设置#################功能弹窗设置###################






############功能弹窗设置###############功能弹窗设置#################功能弹窗设置###################


    if GONGNENG.is_open():
        with GONGNENG.container():
            tabs = st.tabs(["煤矿相关法律咨询", "煤矿安全教育培训", "煤矿安全生产知识综合问答",
                            "煤矿工种岗位知识问答", "煤矿设备维护知识问答", "煤矿事故报告和分析",
                            "煤矿政策研究", "煤矿其它特色服务"])
            with tabs[0]:
                st.markdown("<p style='text-align: left;'><b>1.煤矿相关法律咨询：</b>煤矿大模型可以用于生成煤矿相关法律咨询材料。</p>",
                            unsafe_allow_html=True)
                st.image("img/gongnengshili/1-1.png")     #,height=600
            with tabs[1]:
                st.markdown("<p style='text-align: left;'><b>2.煤矿安全教育培训：</b>煤矿大模型可以用于生成煤矿安全教育材料，如安全教育培训内容的解释等。</p>",
                            unsafe_allow_html=True)
                st.image("img/gongnengshili/2-.png")
            with tabs[2]:
                st.markdown("<p style='text-align: left;'><b>3.煤矿安全生产知识综合问答：</b>煤矿大模型可以针对提问，给出煤矿专业知识回复。</p>",
                            unsafe_allow_html=True)
                st.image("img/gongnengshili/3-.png")
            with tabs[3]:
                st.markdown("<p style='text-align: left;'><b>4.煤矿工种岗位知识问答：</b>煤矿大模型可以针对煤矿工种上岗所需技能，给出专业回复。</p>",
                            unsafe_allow_html=True)
                st.image("img/gongnengshili/4.png")
            with tabs[4]:
                st.markdown("<p style='text-align: left;'><b>5.煤矿设备维护知识问答：</b>煤矿大模型可以针对煤矿设备维护知识，给出专业回复。</p>",
                            unsafe_allow_html=True)
                st.image("img/gongnengshili/5.png")
            with tabs[5]:
                st.markdown("<p style='text-align: left;'><b>6.煤矿事故报告和分析：</b>煤矿大模型可以帮助快速理解和分类事故报告，并提供事故原因的初步分析等。</p>",
                            unsafe_allow_html=True)
                st.image("img/gongnengshili/6.png")
            with tabs[6]:
                st.markdown("<p style='text-align: left;'><b>7.煤矿政策研究：</b>煤矿大模型可以用于分析公众对于相关煤矿政策的反馈，这可以帮助政策制定者更好地了解政策的实际效果。</p>",
                            unsafe_allow_html=True)
                st.image("img/gongnengshili/7.png")
            with tabs[7]:
                st.markdown("<p style='text-align: left;'><b>8.煤矿其它特色服务（研发中）：</b>譬如，根据用户输入的静态和实时数据，生成煤矿灾害预警逻辑和结果。其它煤矿特色服务，持续扩展中。</p>",
                            unsafe_allow_html=True)

############关于弹窗设置###############关于弹窗设置#################关于弹窗设置###################
    # 定义弹窗中显示的内容，使用 HTML 标签来设置样式

    # 定义弹窗内容
    modal_content = """
    <style>
        .modal {
            max-height: 70vh; /* 设置弹窗内容的最大高度为视窗高度的90% */
            overflow-y: auto; /* 允许垂直滚动 */
            padding: 20px; /* 弹窗内容的内边距 */
            margin: 0 auto; /* 水平居中 */
            position: relative; /* 相对定位 */
            
        }
    </style>
    <div class="modal" class="show" id="myModal">
            <h2 style="text-align: center;">煤矿安全生产大模型</h2>
            <p style="text-align: center; font-size: small;">中国矿业大学-自然语言处理课题组</p>
            <p style="text-align: center; font-weight: bold; font-size: small;">联系方式：18361210108（微信同号）</p>
            <p style="margin-top: 20px; text-align: left;">
                CoalChat，中文煤矿安全生产大模型，由中国矿业大学自然语言处理课题组（CUMT-NLPLab）研发。直面大模型带来的学术和产业界变化，搜集清洗筛选海量煤矿安全生产语料，基于中文开源基座大模型Baichuan2-7b-chat，研究无监督预训练、有监督多场景多任务微调、煤矿业者职业习惯对齐等技术方法，构建煤矿大模型CoalChat，旨在提供广泛而准确的煤矿安全生产知识问答和智能分析、决策服务，有效助力煤矿智能化建设。
            </p>
            <p style="text-align: left; font-weight: bold; font-size: larger;">
                主要功能和特点
            </p>
            <p style="text-align: left;">
                <strong>智能问答与解析：</strong>CoalChat能够接收用户的问题或需求，并通过大模型进行智能解析，提供准确、全面的回答和解决方案。
            </p>
            <p style="text-align: left;">
                <strong>行业知识库集成：</strong>产品内置了庞大的矿山行业知识库，涵盖了安全、矿企工种、设备维修、灾害防治等多个领域的专业知识和经验，并不断地迭代更新。
                </p>
            <p style="text-align: left;">
                <strong>多场景应用支持：</strong>产品适用于矿山勘探、开采、管理等多个场景，能够满足不同用户的需求。
            </p>
            <p style="text-align: left; font-weight: bold; font-size: larger;">
                适用于哪些行业或领域
            </p>
            <p style="text-align: left;">
                CoalChat主要适用于矿山行业及其相关领域，包括但不限于金属矿山、非金属矿山、煤炭矿山等。它能够为这些行业的企业和从业者提供智能化的决策支持、生产优化和安全管理等方面的服务。
            </p>
            <p style="text-align: left; font-weight: bold; font-size: larger;">
                解决什么问题
            </p>
            <p style="text-align: left;">
                <strong>提高决策效率：</strong>通过快速准确地回答用户的问题和提供解决方案，CoalChat能够帮助企业缩短决策周期，提高决策效率。
                </p>
            <p style="text-align: left;">
                <strong>增强安全管理：</strong>CoalChat集成了丰富的安全知识和经验，能够帮助企业建立完善的安全管理体系，预防事故的发生，保障员工的安全和健康。
                </p>
            <p style="text-align: left;">
                <strong>提高工作效率：</strong>作为一款智能助手，CoalChat可以辅助行业人群撰写报告、快速获取专业信息等，不断提升工作效率。
            </p>
            <p style="text-align: left; font-weight: bold; font-size: larger;">
                优势与特点
            </p>
            <p style="text-align: left;">
                <strong>基于大模型：</strong>CoalChat建立在大模型的基础之上，这意味着它能够充分利用先进的自然语言处理技术和深度学习能力。
                大模型具备强大的文本生成、理解和对话能力，使得CoalChat在处理复杂的矿山行业问题时更加得心应手。
                </p>
            <p style="text-align: left;">
                <strong>持续学习与优化：</strong>CoalChat利用大型模型的持续学习和优化功能，可以持续地吸收新的行业知识和经验，从而提升自身性能和准确度。这表示，用户在使用CoalChat时将会不断得到更加精准和全面的答复。</p>
            <p style="text-align: left;">
                <strong>专为矿山行业打造：</strong>CoalChat是专为矿山行业定制的大型模型应用，内置了丰富而庞大的矿山行业知识库。这一独特的设计使得它能够更深入地理解矿山安全生产行业的专业术语、业务流程和实际需求，
                从而为用户提供更加个性化、贴合行业特点的服务。
            </p>
    </div>
    """
    # 检查 Modal 是否打开，并显示内容
    if GUANY.is_open():
        with GUANY.container():
            st.markdown(modal_content, unsafe_allow_html=True)
     ###################关于弹窗设置#####################关于弹窗设置##########################关于弹窗设置###############################



    ##############################################test#################################################
