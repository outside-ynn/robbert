import pandas as pd
import torch
import numpy as np
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from transformers import BertTokenizer, RobertaModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载本地的模型和分词器
model_path = "E:/project_py/transformer/modelscope/hub/dienstag/chinese-roberta-wwm-ext-large"
tokenizer = BertTokenizer.from_pretrained(model_path)
model = RobertaModel.from_pretrained(model_path).to(device)

# 读取 Excel 文件
df = pd.read_excel(r"C:\Users\James\Desktop\1762_all.xlsx")
texts = df['truth'].tolist()

# 生成文本的深层语义表示
def get_embeddings(texts, batch_size=8):
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        all_embeddings.append(outputs.last_hidden_state[:, 0, :].cpu().numpy())
    return np.vstack(all_embeddings)  # 合并批次结果

# def get_embeddings(texts):
#     # 将输入数据移到 GPU
#     inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=512).to(device)
#     with torch.no_grad():
#         outputs = model(**inputs)
#     return outputs.last_hidden_state[:, 0, :].cpu().numpy()  # 移回 CPU 以避免内存问题

# 获取文本的表示
embeddings = get_embeddings(texts)

# 使用 CountVectorizer 提取词频
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)

# 选择最佳主题数量
def compute_perplexity(X, n_components):
    lda = LatentDirichletAllocation(n_components=n_components, random_state=42)
    lda.fit(X)
    return lda.perplexity(X)

# 计算不同主题数量的困惑度
perplexities = []
n_components_range = range(2, 10)  # 假设主题数量范围从2到10
for n_components in n_components_range:
    perplexity = compute_perplexity(X, n_components)
    perplexities.append(perplexity)
    print(f'Number of topics: {n_components}, Perplexity: {perplexity}')

# 选择最佳主题数量（困惑度最低）
best_n_components = n_components_range[perplexities.index(min(perplexities))]
print(f'最佳主题数量: {best_n_components}')

# 使用最佳主题数量进行 LDA 模型训练
lda = LatentDirichletAllocation(n_components=best_n_components, random_state=42)
lda.fit(X)

# 获取每个主题的词
def get_topics(lda_model, vectorizer, n_words=10):
    words = vectorizer.get_feature_names_out()
    topics = {}
    for idx, topic in enumerate(lda_model.components_):
        topics[f'Topic {idx}'] = [words[i] for i in topic.argsort()[-n_words:][::-1]]
    return topics

topics = get_topics(lda, vectorizer)
for topic, words in topics.items():
    print(f'{topic}: {", ".join(words)}')
