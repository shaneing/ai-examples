import os

import pandas as pd
from pandasai import SmartDataframe
from pandasai_openai import OpenAI
from pandasai.config import Config
from pandas_exmaple.llm import LocalLLM

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
f = [f.name for f in fm.fontManager.ttflist]
print(f)
# plt.rc('font', family='Microsoft Sans Serif')
plt.rcParams['font.sans-serif']=[u'SimHei']
plt.rcParams['axes.unicode_minus']=False



from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("API_KEY")

# 1. 读取 Excel 文件
# 假设 Excel 有 'Date', 'Department', 'Category', 'Amount' 等列
df = pd.read_excel("detail_2024.xlsx")

# 2. 配置 LLM (这里以 OpenAI 为例)
# llm = OpenAI(
#     api_token=api_key,
#     api_base="https://api.deepseek.com/chat/completions",
#     model="deepseek-chat"
# )
llm = LocalLLM(
    api_key=api_key,
    api_base="https://api.deepseek.com",
    model="deepseek-chat"
)

c = Config(
    llm=llm,
)

# 3. 创建智能 DataFrame
sdf = SmartDataframe(df, config=c)

# 4. 进行提问
query = "请展示各部门的总开销占比？"
response = sdf.chat(query)
print(f"回答: {response}")

# 你甚至可以让它画图
sdf.chat("请画一个饼图展示各部门的总开销占比")