import torch
from transformers import BertTokenizer, RobertaModel
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

# 加载本地的RoBERTa模型和分词器
model_path = "E:/project_py/transformer/modelscope/hub/dienstag/chinese-roberta-wwm-ext-large"
tokenizer = BertTokenizer.from_pretrained(model_path)
model = RobertaModel.from_pretrained(model_path)

# 输入文本数据
documents = ["文本数据1", "文本数据2", ...]  # 替换为你的文本数据

# 使用RoBERTa生成文本的深层语义表示
def get_roberta_embeddings(documents):
    embeddings = []
    for doc in documents:
        inputs = tokenizer(doc, return_tensors='pt', padding=True, truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
        # 获取最后一层隐藏状态的平均值作为文本的表示
        embeddings.append(outputs.last_hidden_state.mean(dim=1).squeeze().numpy())
    return np.array(embeddings)

# 获取文本嵌入
roberta_embeddings = get_roberta_embeddings(documents)

# 使用CountVectorizer将文本转换为词频矩阵（用于LDA）
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(documents)

# 选择主题数量
num_topics = 5  # 替换为你想要的主题数量

# 训练LDA模型
lda_model = LatentDirichletAllocation(n_components=num_topics, random_state=42)
lda_model.fit(X)

# 计算困惑度
perplexity = lda_model.perplexity(X)
print(f"困惑度: {perplexity}")

# 输出每个主题的关键词
for idx, topic in enumerate(lda_model.components_):
    print(f"主题 {idx}: {[vectorizer.get_feature_names_out()[i] for i in topic.argsort()[-10:]]}")

# 为每个文本分配主题
topic_distribution = lda_model.transform(X)
for i, doc in enumerate(documents):
    print(f"文本 {i} 的主题分布: {topic_distribution[i]}")
