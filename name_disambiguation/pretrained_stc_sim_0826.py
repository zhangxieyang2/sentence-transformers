# -*- encoding: utf-8 -*-
"""
    @author: weili
    @time: 
    @des: 用sentence-transformers预训练好的模型，计算两个文本的相似性，设置阈值确定label。主要用于走通流程
"""
from sentence_transformers import SentenceTransformer, util
import numpy as np
from sklearn.cluster import KMeans
import json
import os
from itertools import combinations
import torch

embedder = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')
valid_dat_dir = r'/home/weili/wp_c/NameDisambiguation/data/valid'


def gen_doc_pair_for_pred():
    """
    同名作者的论文列表，枚举所有论文pair；每个论文用其关键信息拼接的stc表示
    :return:
    """
    author_name2doc_ids = json.load(open(os.path.join(valid_dat_dir, 'sna_valid_author_raw.json')))
    doc_id2info = json.load(open(os.path.join(valid_dat_dir, 'sna_valid_pub.json')))
    doc_id2meta_stc = {}
    for doc_id, doc_info in doc_id2info.items():
        meta_stc = gen_meta_stc_of_doc(doc_info)
        doc_id2meta_stc[doc_id] = meta_stc
    # 枚举论文pair
    doc_id_pairs = []
    for aut_name, doc_id_list in author_name2doc_ids.items():
        doc_id_pairs.extend(combinations(doc_id_list, 2))
    return doc_id2meta_stc, doc_id_pairs


def cal_stc_sim(doc_id2meta_stc, doc_id_pairs=None, num_clusters=5, sim_thresh=0):
    # doc_id_list, meta_stc_list = list(doc_id2meta_stc.keys()), list(doc_id2meta_stc.values())
    # # 得到句子表示
    # corpus_embeddings = embedder.encode(meta_stc_list, convert_to_tensor=True)
    # doc_id2emb = {}
    # for doc_id, doc_emb in zip(doc_id_list, corpus_embeddings):
    #     doc_id2emb[doc_id] = doc_emb
    # 计算相似性
    author_name2doc_ids = json.load(open(os.path.join(valid_dat_dir, 'sna_valid_author_raw.json')))
    author_name2doc_clusters = {}
    for aut_name, doc_id_list in author_name2doc_ids.items():
        stc_list = [doc_id2meta_stc[doc_id] for doc_id in doc_id_list]
        # emb_list = embedder.encode(stc_list, convert_to_tensor=True)
        emb_list = embedder.encode(stc_list)
        cos_scores = util.pytorch_cos_sim(emb_list, emb_list)  # 应该是个N*N的矩阵吧
        # TODO 相似度计算出来了，下一步是用聚类算法呢，还是先判断两个doc是否是一个人写的（利用thresh硬分割），然后手动聚起来？
        clustering_model = KMeans(n_clusters=num_clusters)
        clustering_model.fit(emb_list)
        cluster_assignment = clustering_model.labels_

        doc_id_clusters = [[] for i in range(num_clusters)]
        clustered_sentences = [[] for i in range(num_clusters)]
        for sentence_idx, cluster_id in enumerate(cluster_assignment):
            doc_id = doc_id_list[sentence_idx]
            doc_id_clusters[cluster_id].append(doc_id)
            clustered_sentences[cluster_id].append(doc_id2meta_stc[doc_id])
        author_name2doc_clusters[aut_name] = doc_id_clusters
    return author_name2doc_clusters  # 按照提交格式的


def gen_meta_stc_of_doc(doc_info):
    meta_stc = ''
    for thek, thev in doc_info.items():
        if thek in {'id', 'year'}:
            continue
        if thek == 'authors':
            thev = '; '.join(['{}, {}'.format(vv['name'], vv['org']) for vv in thev])
        elif type(thev) == list:
            thev = '; '.join(thev)
        if thev:
            meta_stc += ' {}: {}.'.format(thek.title(), thev)  # 字段值也加上
    meta_stc = meta_stc.strip()
    return meta_stc


def json_dump(obj, fp, with_indent=False):
    """
    :param obj:
    :param fp
    :param with_indent
    :return:
    """
    if with_indent:
        fp.write(json.dumps(obj, ensure_ascii=False, indent=4).encode('utf-8'))
    else:
        fp.write(json.dumps(obj, ensure_ascii=False).encode('utf-8'))
    fp.close()


def gen_pesudo_stc_pair():
    """
    同一个作者的不同论文的meta_stc之间，互为pair
    用这些pair去训练sentence transformer
    两两生成pair数量太大了，可以随机选
    :return:
    """
    dat_dir = r'/home/weili/wp_c/disambiguation/my_data'
    authors_train = json.load(open(os.path.join(dat_dir, 'authors_train.json')))
    authors_valid = json.load(open(os.path.join(dat_dir, 'authors_valid.json')))
    pubs_train = json.load(open(os.path.join(dat_dir, 'pubs_train.json'), encoding='utf-8'))
    from itertools import combinations
    from tqdm import tqdm
    from random import sample as rand_sample
    stc_ids_1, stc_ids_2, stc1_list, stc2_list, label_list = [], [], [], [], []
    import csv
    os.makedirs('./data', exist_ok=True)
    writer = csv.writer(open('./data/meta_stc_pair.tsv', 'w', encoding='utf-8'), delimiter='\t')
    writer.writerow(['qid1', 'qid2', 'question1', 'question2', 'is_duplicate'])
    for aut_name, cluster_id2pub_list in tqdm(authors_train.items()):
        for pub_id_list in cluster_id2pub_list.values():
            pub_id_pairs = list(combinations(pub_id_list, 2))
            sample_num = len(pub_id_pairs)
            if sample_num > 60 * 30:
                sample_num = int(sample_num / 60)
                pub_id_pairs = rand_sample(pub_id_pairs, sample_num)
            # print(len(pub_id_list), len(pub_id_pairs))
            for id1, id2 in pub_id_pairs:
                # stc_ids_1.append(id1)
                # stc_ids_2.append(id2)
                if id1 not in pubs_train or id2 not in pubs_train:
                    print(id1, id2)
                    continue
                stc1 = gen_meta_stc_of_doc(pubs_train[id1])
                stc2 = gen_meta_stc_of_doc(pubs_train[id2])
                writer.writerow([id1, id2, stc1, stc2, 1])
        break
    print(len(stc_ids_1))  # 63025502
    print()


if __name__ == '__main__':
    # # 1.最基本的，用预训练的sentence transformer，计算论文meta_stc的相似度，并聚类。结果0.39713
    # doc_id2meta_stc_, _ = gen_doc_pair_for_pred()
    # author_name2doc_clusters_ = cal_stc_sim(doc_id2meta_stc_)
    # out_f = open('/home/weili/wp_c/NameDisambiguation/pred_res/basic_cluster.json', 'wb')
    # out_f.write(json.dumps(author_name2doc_clusters_, ensure_ascii=False, indent=4).encode('utf-8'))

    # 2.用论文的meta_stc生成“伪同义句对”，训练sentence transformer，然后同1。
    gen_pesudo_stc_pair()
