import os
import streamlit as st
import pandas as pd
from llm import LocalLLM
from pandasai.core.prompts.base import BasePrompt
from pandasai.helpers.memory import Memory
from pandasai.agent.base import AgentState


import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

f = [f.name for f in fm.fontManager.ttflist]
print(f)
# plt.rc('font', family='Microsoft Sans Serif')
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

st.set_page_config(page_title="Finance AI", page_icon="💰", layout="centered")
st.title("💰 KTech Finance AI Assistant")

# --- Sidebar: Configuration ---
with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("DeepSeek API Key", type="password")

    role = st.selectbox("请选择一个角色：", ["会计", "财务分析师", "老板"])

    uploaded_files = st.file_uploader(
        "Upload a File", accept_multiple_files=True, type=["xlsx", "csv"]
    )

    st.divider()
    st.markdown("### Model Settings")
    temperature = st.slider(
        "Temperature", 0.0, 1.0, 0.1, help="Lower is better for code generation"
    )


prompt_template = """
您是一名首席财务管家。您的任务是基于提供的财务数据文件和用户的角色和具体问题，进行深入的数据分析和未来预测。请严格遵循以下步骤：
1. 仔细阅读并理解提供的财务数据文件内容: {financial_data}。
2. 请站在用户的角色 {role} 的角度去回答。
3. 准确解析用户提出的具体问题或需求: {user_query}。用户的需求可能涉及历史数据分析、当前财务状况评估或未来趋势预测。
5. 结合文件数据和用户需求，进行以下操作：
   a. 数据解读：识别关键财务指标（如收入、支出、利润、现金流等）及其变化趋势。
   c. 预测生成：若用户需求涉及预测（如下一季度的利润、未来增长趋势等），请基于历史数据模式和合理的财务假设进行推算，并简要说明预测依据。
5. 输出您的分析和预测结果。输出应：
   - 使用清晰、专业的财务语言，但避免过度使用专业术语。
   - 结构化呈现（如使用段落、项目符号或自然的分点叙述），但绝对不要使用任何XML标签。
   - 使用图标的方式进行回答，让用户看起来更直观
6. 输出格式：
    您的角色是： {role}，针对您的问题分析如下：
    具体回答内容
     
"""


# --- Function to setup LLM ---
def get_llm(api_key, temperature):
    return LocalLLM(
        api_base="https://api.deepseek.com/v1",
        model="deepseek-chat",
        api_key=api_key,
        temperature=temperature,
    )


def split_dataframe(df, chunk_size=100):
    return [df[i : i + chunk_size] for i in range(0, df.shape[0], chunk_size)]


if uploaded_files and api_key:
    dfs = []

    for uploaded_file in uploaded_files:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(".xlsx"):
            # Reads the first sheet by default
            df = pd.read_excel(uploaded_file)
        else:
            raise Exception("Only support csv and xlsx")
        dfs.append(df)

    combined_df = pd.concat(dfs, ignore_index=True)

    # 2. Initialize Chat History
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 3. Display Chat History
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            # Check if content is an image path (PandasAI chart)
            if isinstance(message["content"], str) and message["content"].endswith(
                ".png"
            ):
                st.image(message["content"])
            else:
                st.markdown(message["content"])

    # 4. Handle User Input
    if user_query := st.chat_input("Ask something about your data..."):
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # Generate Response
        with st.chat_message("assistant"):
            with st.spinner("DeepSeek is analyzing..."):
                try:
                    llm = get_llm(api_key, temperature)
                    memory = Memory()

                    data_chunks = split_dataframe(combined_df, chunk_size=1000)
                    inputs_for_model = []

                    paused = st.checkbox("Pause Analysis", value=False)
                    for chunk in data_chunks:
                        chunk_text = chunk.to_csv(index=False, sep="\t")

                        props = {
                            "role": role,
                            "financial_data": chunk_text,
                            "user_query": user_query,
                        }
                        BasePrompt.template = prompt_template
                        instruction = BasePrompt(props=props)
                        context = AgentState(memory=memory)

                        while paused:
                            st.write("Analysis paused. Uncheck to continue.")
                            paused = st.checkbox("Pause Analysis", value=True)

                        response = llm.call(instruction, props, context=context)
                        inputs_for_model.append(response)

                    final_analysis_request = "\n".join(inputs_for_model)
                    final_props = {
                        "role": role,
                        "financial_data": "这些是所有数据块的合并分析：\n"
                        + final_analysis_request,
                        "user_query": user_query,
                    }
                    BasePrompt.template = prompt_template
                    final_instruction = BasePrompt(props=final_props)
                    final_response = llm.call(
                        final_instruction,
                        props=final_props,
                        context=AgentState(memory=memory),
                    )

                    # Handle Response Types (Text vs Image)
                    if (
                        isinstance(final_response, str)
                        and os.path.isfile(final_response)
                        and final_response.endswith(".png")
                    ):
                        st.image(final_response)
                        st.session_state.messages.append(
                            {"role": "assistant", "content": final_response}
                        )
                    else:
                        st.markdown(final_response)
                        st.session_state.messages.append(
                            {"role": "assistant", "content": final_response}
                        )

                except Exception as e:
                    st.error(f"Error: {e}")
                    print(f"Error:{e}")
else:
    st.info("Please upload a CSV file and enter your DeepSeek API Key to start.")
