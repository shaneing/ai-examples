import streamlit as st
import pandas as pd
from pandasai import SmartDataframe
from pandasai.config import Config
from pandas_example.llm import LocalLLM
import os

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
f = [f.name for f in fm.fontManager.ttflist]
print(f)
# plt.rc('font', family='Microsoft Sans Serif')
plt.rcParams['font.sans-serif']=[u'SimHei']
plt.rcParams['axes.unicode_minus']=False

st.set_page_config(page_title="DeepSeek Data Chatbot", page_icon="🐼", layout="centered")
st.title("🐼 PandasAI + DeepSeek Chatbot")

# --- Sidebar: Configuration ---
with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("DeepSeek API Key", type="password")
    uploaded_file = st.file_uploader("Upload a File", type=["xlsx", "csv"])

    st.divider()
    st.markdown("### Model Settings")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.1, help="Lower is better for code generation")


# --- Function to setup LLM ---
def get_llm(api_key):
    return LocalLLM(
        api_base="https://api.deepseek.com/v1",
        model="deepseek-chat",
        api_key=api_key,
        temperature=temperature
    )


# --- Main Chat Logic ---
if uploaded_file and api_key:
    # 1. Load Data
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith('.xlsx'):
        # Reads the first sheet by default
        df = pd.read_excel(uploaded_file)
    else:
        raise Exception("Only support csv and xlsx")

    st.write("### Data Preview")
    st.dataframe(df.head(3))

    # 2. Initialize Chat History
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 3. Display Chat History
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            # Check if content is an image path (PandasAI chart)
            if isinstance(message["content"], str) and message["content"].endswith(".png"):
                st.image(message["content"])
            else:
                st.markdown(message["content"])

    # 4. Handle User Input
    if prompt := st.chat_input("Ask something about your data..."):
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate Response
        with st.chat_message("assistant"):
            with st.spinner("DeepSeek is analyzing..."):
                try:
                    llm = get_llm(api_key)
                    c = Config(
                        llm=llm,
                    )
                    # Configure SmartDataframe
                    sdf = SmartDataframe(
                        df,
                        config=c,
                    )

                    # Run Query
                    response = sdf.chat(prompt)

                    # Handle Response Types (Text vs Image)
                    if isinstance(response, str) and os.path.isfile(response) and response.endswith(".png"):
                        st.image(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                    else:
                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})

                except Exception as e:
                    st.error(f"Error: {e}")

else:
    st.info("Please upload a CSV file and enter your DeepSeek API Key to start.")
