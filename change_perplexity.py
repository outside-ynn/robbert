import pandas as pd
import torch
import numpy as np
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import BertTokenizer, RobertaModel
from sklearn.model_selection import train_test_split
import math

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
    return np.vstack(all_embeddings)


# 获取文本的表示
embeddings = get_embeddings(texts)

# 使用 TfidfVectorizer 提取特征
tf_vectorizer = TfidfVectorizer(strip_accents='unicode', max_features=500, max_df=0.99, min_df=0.002)
documents = df['truth'].values.tolist()
doc_train, doc_test = train_test_split(documents, test_size=0.2)

# 训练集和测试集的特征矩阵
doc_train_matrix = tf_vectorizer.fit_transform(doc_train)
doc_test_matrix = tf_vectorizer.transform(doc_test)


# 计算 LDA 模型的困惑度
def docperplexity(ldamodel, testset, dictionary, size_dictionary, num_topics):
    print('模型信息：')
    print('主题数量: %s' % num_topics)

    prob_doc_sum = 0.0
    testset_word_num = 0
    doc_topics_list = ldamodel.transform(testset)  # 使用 transform 获取主题分布

    topic_word_list = []
    for topic_id in range(num_topics):
        topic_word = ldamodel.components_[topic_id]
        topic_word_list.append({dictionary[i]: prob for i, prob in enumerate(topic_word)})

    for i, doc in enumerate(testset):
        prob_doc = 0.0
        doc_word_num = 0

        for word_id, num in dict(doc).items():
            prob_word = 0.0
            doc_word_num += num
            word = tf_vectorizer.get_feature_names_out()[word_id]
            for topic_id in range(num_topics):
                prob_topic = doc_topics_list[i][topic_id]
                prob_topic_word = topic_word_list[topic_id].get(word, 0)
                prob_word += prob_topic * prob_topic_word

            prob_doc += math.log(prob_word) if prob_word > 0 else 0  # 避免负无穷
        prob_doc_sum += prob_doc
        testset_word_num += doc_word_num

    perplexity_value = math.exp(-prob_doc_sum / testset_word_num) if testset_word_num > 0 else float('inf')
    print("模型困惑度的值为 : %s" % perplexity_value)
    return perplexity_value


# 计算不同主题数量的困惑度
perplexities = []
n_components_range = range(2, 21, 2)  # 假设主题数量范围从2到20
for n_components in n_components_range:
    LDA = LatentDirichletAllocation(n_components=n_components, random_state=42)
    LDA.fit(doc_train_matrix)
    perplexity_value = docperplexity(LDA, doc_test_matrix, tf_vectorizer.get_feature_names_out(), 500, n_components)
    perplexities.append(perplexity_value)

# 选择最佳主题数量（困惑度最低）
best_n_components = n_components_range[perplexities.index(min(perplexities))]
print(f'最佳主题数量: {best_n_components}')

# 使用最佳主题数量进行 LDA 模型训练
lda = LatentDirichletAllocation(n_components=best_n_components, random_state=42)
lda.fit(doc_train_matrix)


# 获取每个主题的词
def get_topics(lda_model, vectorizer, n_words=10):
    words = vectorizer.get_feature_names_out()
    topics = {}
    for idx, topic in enumerate(lda_model.components_):
        topics[f'Topic {idx}'] = [words[i] for i in topic.argsort()[-n_words:][::-1]]
    return topics


topics = get_topics(lda, tf_vectorizer)
for topic, words in topics.items():
    print(f'{topic}: {", ".join(words)}')
