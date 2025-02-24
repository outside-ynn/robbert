import torch
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans

from transformers import BertTokenizer, RobertaModel

# 加载本地的模型和分词器
model_path = "E:/project_py/transformer/modelscope/hub/dienstag/chinese-roberta-wwm-ext-large"
tokenizer = BertTokenizer.from_pretrained(model_path)
model = RobertaModel.from_pretrained(model_path)


# 准备要分析的文本数据
texts = [
    "今天的天气非常好，我和朋友们一起去公园散步。",
    "股票市场最近波动很大，许多投资者都在观望。",
    "人工智能技术正在快速发展，尤其是在自动驾驶领域。",
    "昨天我看了一场精彩的足球比赛，比赛非常激烈。",
    "中国古代文化源远流长，孔子提出的儒家思想影响深远。",
    "最近我在学习Python编程语言，感觉它非常强大且易于使用。"
]


# 使用分词器对文本进行编码,分词器就是把句子分成一个个的词语，然后把词语给上索引
'''
“今天的天气很好。”对于这个句子
使用分词器，可能会得到如下的分词结果：
["今天", "的", "天气", "很好", "。"]
每个词会被分配一个对应的索引，例如：
{"今天": 0, "的": 1, "天气": 2, "很好": 3, "。": 4}

'''
def encode_texts(texts):
    inputs = [tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512) for text in texts]
    return inputs


# 获取RoBERTa模型的嵌入向量,嵌入向量就是将每个词增添特征维度的元素，这个函数遍历每个编码后的文本输入，通过RoBERTa模型生成嵌入向量（深层语义表示）。
'''
{
  "今天": [0.1, 0.2, ..., 0.128],
  "的": [0.05, 0.1, ..., 0.128],
  "天气": [0.3, 0.4, ..., 0.128],
  "很好": [0.2, 0.5, ..., 0.128],
  "。": [0.0, 0.0, ..., 0.128]
}
'''
def get_embeddings(inputs):
    embeddings = []
    for input_data in inputs:
        with torch.no_grad():
            outputs = model(**input_data)
        embeddings.append(outputs.last_hidden_state.mean(dim=1).squeeze().numpy())
    return embeddings


# LDA主题建模（基于词频矩阵）
def lda_topic_modeling(texts, n_topics=5):  #用困惑度去解决这个topic
    # 使用 CountVectorizer 创建词频矩阵
    vectorizer = CountVectorizer()
    text_matrix = vectorizer.fit_transform(texts)#这个text_matrix就是一个词频矩阵，行是句子（文档），列是词语

    # 进行 LDA 主题建模
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(text_matrix) #训练主题和词的关系

    return lda, vectorizer


# KMeans 聚类（基于 RoBERTa 嵌入向量）
def kmeans_topic_modeling(embeddings, n_clusters=5):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(embeddings)
    return kmeans


def main():
    # 使用词频矩阵进行 LDA 主题建模
    lda_model, vectorizer = lda_topic_modeling(texts, n_topics=5)

    # 打印每个主题的前10个词
    feature_names = vectorizer.get_feature_names_out()
    for idx, topic in enumerate(lda_model.components_):
        print(f"主题 {idx+1}:")
        print([feature_names[i] for i in topic.argsort()[:-11:-1]])

    # 编码文本并获取 RoBERTa 嵌入向量
    inputs = encode_texts(texts)
    embeddings = get_embeddings(inputs)

    # 使用 KMeans 聚类 RoBERTa 嵌入向量
    kmeans_model = kmeans_topic_modeling(embeddings, n_clusters=5)

    # 打印每个文本的聚类标签
    print("\nKMeans 聚类结果：")
    for i, label in enumerate(kmeans_model.labels_):
        print(f"文本 {i+1} 属于聚类 {label}")


if __name__ == "__main__":
    main()
