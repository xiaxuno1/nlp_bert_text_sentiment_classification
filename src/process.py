#!/usr/bin/env python
# coding=utf-8
"""
@author: xiaxu
@license: MIT
@file: process.py
@date: 2025/10/23 15:20
@desc: 处理原始的jsonl数据
"""
from datasets import load_dataset
from transformers import AutoTokenizer

from src import config


def process():
    """
    读取、处理、划分、tokenizer
    :return：train_dataset.jsonl；test_dataset.jsonl
    """
    dataset = load_dataset('csv', data_files=str(config.RAW_DATA_DIR/'online_shopping_10_cats.csv'))['train']
    dataset = dataset.remove_columns(['cat'])
    dataset = dataset.filter(lambda x:x['review'] is not None and x['review'].strip() != '')

    #划分测试集和数据集
    dataset = dataset.class_encode_column('label') #转换为ClassLabel，stratify_by_column只支持此
    dataset_dict = dataset.train_test_split(test_size=0.2,stratify_by_column='label')

    #tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)

    #encode
    def batch_encode(batch_seq):
        inputs = tokenizer(batch_seq['review'],padding="max_length",truncation=True,max_length = config.SEQ_LEN) #长度对齐，大的截断，小的填充
        inputs['label'] = batch_seq['label'] #添加一列
        return inputs
    #map
    dataset_dict = dataset_dict.map(batch_encode,batched=True,remove_columns=['review','label'])

    #save
    dataset = dataset_dict.save_to_disk(config.PROCESSED_DATA_DIR)


if __name__ == '__main__':
    process()