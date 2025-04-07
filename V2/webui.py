# import streamlit as st
# from webui_pages.utils import *
# from streamlit_option_menu import option_menu
# from webui_pages.dialogue.dialogue import dialogue_page, chat_box
# from webui_pages.knowledge_base.knowledge_base import knowledge_base_page
# import os
# import sys
# from configs import VERSION
# from server.utils import api_address


# api = ApiRequest(base_url=api_address())

# if __name__ == "__main__":
#     is_lite = "lite" in sys.argv

#     st.set_page_config(
#         "Langchain-Chatchat WebUI",
#         os.path.join("img", "chatchat_icon_blue_square_v2.png"),
#         initial_sidebar_state="expanded",
#         menu_items={
#             'Get Help': 'https://github.com/chatchat-space/Langchain-Chatchat',
#             'Report a bug': "https://github.com/chatchat-space/Langchain-Chatchat/issues",
#             'About': f"""欢迎使用 Langchain-Chatchat WebUI {VERSION}！"""
#         }
#     )

#     pages = {
#         "对话": {
#             "icon": "chat",
#             "func": dialogue_page,
#         },
#         "知识库管理": {
#             "icon": "hdd-stack",
#             "func": knowledge_base_page,
#         },
#     }

#     with st.sidebar:
#         st.image(
#             os.path.join(
#                 "img",
#                 "logo-long-chatchat-trans-v2.png"
#             ),
#             use_column_width=True
#         )
#         st.caption(
#             f"""<p align="right">当前版本：{VERSION}</p>""",
#             unsafe_allow_html=True,
#         )
#         options = list(pages)
#         icons = [x["icon"] for x in pages.values()]

#         default_index = 0
#         selected_page = option_menu(
#             "",
#             options=options,
#             icons=icons,
#             # menu_icon="chat-quote",
#             default_index=default_index,
#         )

#     if selected_page in pages:
#         pages[selected_page]["func"](api=api, is_lite=is_lite)


# import streamlit as st
# from webui_pages.utils import *
# from streamlit_option_menu import option_menu
# from webui_pages.dialogue.dialogue import dialogue_page, chat_box
# from webui_pages.knowledge_base.knowledge_base import knowledge_base_page
# import os
# import sys
# from configs import VERSION
# from server.utils import api_address
#
#
# api = ApiRequest(base_url=api_address())
#
# if __name__ == "__main__":
#     is_lite = "lite" in sys.argv
#
#     st.set_page_config(
#         "CoalChat V1",
#         os.path.join("img", "煤矿安全模型图标.png"),
#         initial_sidebar_state="expanded",
#         menu_items={
#             'About': f"""欢迎使用 CoalChat V1 ！"""
#         }
#     )
#
#     pages = {
#         "LLM对话": {
#             "icon": "chat",
#             "func": dialogue_page,
#         },
#         "知识库管理": {
#             "icon": "hdd-stack",
#             "func": knowledge_base_page,
#         },
#     }
#
#     # with st.sidebar:        # 在侧边栏开始一个新的布局。
#
#     #     cols = st.columns(2)        # 创建两列布局，为接下来的按钮分配空间。
#     #     #export_btn = cols[0]            # 第一列分配给“导出记录”按钮变量 第二列放置一个“清空对话”按钮
#     #     if cols[0].button(
#     #             "LLM对话",
#     #             use_container_width=True,
#     #     ):
#     #         dialogue_page()      # 将调用chat_box的reset_history方法清除对话历史，并重新运行应用以反映更新。
#     #     if cols[1].button(
#     #             "知识库管理",
#     #             use_container_width=True,
#     #     ):
#     #         knowledge_base_page()  # 将调用chat_box的reset_history方法清除对话历史，并重新运行应用以反映更新。
#
#     with st.sidebar:
#         # st.caption(
#         #     """<p style="
#         #         text-align: center;
#         #         font-size: 24px;
#         #         font-family: 'Microsoft YaHei', SimHei, sans-serif;
#         #         font-weight: 900;
#         #         color: #000;
#         #         text-shadow: 0.5px 0.5px 1px rgba(0,0,0,0.3);
#         #     ">煤矿安全问答系统</p>""",
#         #     unsafe_allow_html=True,
#         # )
#         st.image(
#             os.path.join(
#                 "img",
#                 "煤矿安全模型图标.png"
#             ),
#             use_column_width=True
#         )
#     #     st.caption(
#     #                 f"""<p align="down">当前版本：V1</p>""",
#     #                 unsafe_allow_html=True,
#     #     )
#     #     options = list(pages.keys())  # 直接获取字典的键作为选项列表
#     #
#     #     # 使用st.radio创建单选菜单
#     #     selected_page = st.radio("", options, index=0, horizontal=True)
#     #
#     # if selected_page in pages:
#     #         # 注意：这里直接调用函数，如果需要传递额外参数，请确保它们在作用域内或作为参数传递
#     #     pages[selected_page]["func"](api=api, is_lite=is_lite)
#         st.caption(
#             f"""<p align="right">当前版本：V1</p>""",
#             unsafe_allow_html=True,
#         )
#         options = list(pages)
#         icons = [x["icon"] for x in pages.values()]
#
#         default_index = 0
#         selected_page = option_menu(
#             "",
#             options=options,
#             icons=icons,
#             # menu_icon="chat-quote",
#             default_index=default_index,
#         )
#
#     if selected_page in pages:
#         pages[selected_page]["func"](api=api, is_lite=is_lite)


import streamlit as st
from webui_pages.utils import *
from webui_pages.dialogue.dialogue import dialogue_page, chat_box
from webui_pages.knowledge_base.knowledge_base import knowledge_base_page
import os
import sys
from server.utils import api_address

api = ApiRequest(base_url=api_address())

if __name__ == "__main__":
    is_lite = "lite" in sys.argv

    # 页面基础配置
    st.set_page_config(
        "CoalChat V1",
        os.path.join("img", "煤矿安全模型图标.png"),
        initial_sidebar_state="expanded",
        menu_items={
            'About': f"""欢迎使用 CoalChat V1 ！"""
        }
    )

    # 注入自定义CSS样式
    st.markdown("""
    <style>
    /* 主界面渐变背景 */
    .stApp {
        background: #FFFFFF !important;
        color: #333333;
    }

    /* 侧边栏样式 */
    section[data-testid="stSidebar"] {
        background: #F8F9FA !important;
        border-right: 1px solid #E9ECEF;
    }
    .st-emotion-cache-6qob1r {
        background: #F8F9FA !important;
    }

    /* 聊天消息样式 */
    .user-message {
        background: #E3F2FD !important;
        color: #1A237E !important;
        border-radius: 18px 18px 3px 18px !important;
        padding: 12px !important;
        margin: 8px 0 !important;
        border: 1px solid #BBDEFB;
    }
    .bot-message {
        background: #F5F5F5 !important;
        color: #212121 !important;
        border-radius: 18px 18px 18px 3px !important;
        padding: 12px !important;
        margin: 8px 0 !important;
        border: 1px solid #EEEEEE;
    }

    /* 按钮样式 */
    .stButton > button {
        background: #2196F3 !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        transition: all 0.3s ease !important;
    }
    .stButton > button:hover {
        background: #1976D2 !important;
        box-shadow: 0 2px 8px rgba(33,150,243,0.2);
    }

    /* 输入框样式 */
    .stTextInput input {
        background: white !important;
        border: 1px solid #BBDEFB !important;
        border-radius: 8px !important;
        padding: 10px !important;
    }

    /* 选项卡样式 */
    [data-baseweb="tab-list"] {
        background: #F8F9FA !important;
        border-radius: 10px;
        padding: 8px !important;
        border: 1px solid #E9ECEF;
    }
    [data-baseweb="tab"] {
        color: #666666 !important;
        transition: 0.3s !important;
    }
    [data-baseweb="tab"]:hover {
        color: #2196F3 !important;
        background: rgba(33,150,243,0.1) !important;
    }
    /* 欢迎提示区域样式 */
    .welcome-banner {
        text-align: center; 
        text-justify: inter-word; /* 单词间的间距调整 */
        padding: 10px;
        background: #F8F9FA;
        border-radius: 18px;
        margin-bottom: 20px;
        box-shadow: 0 10px 10px rgba(0, 0, 0, 0.1);
    }
    @media (max-width: 768px) {
        .welcome-banner {
            padding: 10px;
            border-width: 5px;
        }
        
    .welcome-banner h2 {
        color: #1A237E;
        font-size: 28px;
        text-align: center; /* 标题居中对齐 */
    }

    .welcome-banner p {
        color: #666666;
        font-size: 18px;
        line-height: 1.6;
        margin: 10px 0; /* 段落间距 */
    }
    </style>
    """, unsafe_allow_html=True)

    # 页面路由配置
    pages = {
        "LLM对话": {
            "icon": "chat",
            "func": dialogue_page,
        },
        "知识库管理": {
            "icon": "hdd-stack",
            "func": knowledge_base_page,
        },
    }

    # 侧边栏设计
    with st.sidebar:
        # 标题区域
        st.markdown("""
            <div style="
                text-align: center;
                padding: 20px 0;
                background: white;
                border-radius: 12px;
                margin: -10px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            ">
                <h1 style="
                    color: #1A237E;
                    font-size: 26px;
                    margin: 10px 0;
                    font-weight: 600;
                ">煤矿安全规范问答系统</h1>
                <div style="border-bottom: 2px solid #BBDEFB; margin: 0 25px;"></div>
                <p style="
                    color: #666666;
                    font-size: 16px;
                    margin: 10px 0;
                ">当前版本：V1</p>
            </div>
        """, unsafe_allow_html=True)

        # 系统图标
        st.image(
            os.path.join("img", "煤矿安全模型图标.png"),
            use_column_width=True,
            output_format="PNG"
        )

        # 使用st.radio创建单选菜单
        options = list(pages.keys())  # 直接获取字典的键作为选项列表
        selected_page = st.radio("", options, index=0, horizontal=True)
    # # 主页面渲染
    # if selected_page in pages:
    #     pages[selected_page]["func"](api=api, is_lite=is_lite)
    # 主页面渲染
    if selected_page == "LLM对话":
        # # 清空侧边栏的默认内容
        # st.sidebar.markdown("")

        # 欢迎提示和功能说明
        st.markdown("""
                <div class="welcome-banner">
                <h2>欢迎使用煤矿安全规范问答系统</h2>
                <p>
                    这是专为您设计的煤矿安全规范问答系统，可对话获取相关知识解答。
                </p>
                <p>
                    请在下方输入框中输入您的问题，系统会尽力为您提供帮助。
                </p>
            </div>
            """, unsafe_allow_html=True)

        # 调用对话页面函数
        dialogue_page(api=api, is_lite=is_lite)

    elif selected_page == "知识库管理":
        # 知识库管理页面
        knowledge_base_page(api=api, is_lite=is_lite)

