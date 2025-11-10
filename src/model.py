#!/usr/bin/env python
# coding=utf-8
"""
@author: xiaxu
@license: MIT
@file: model.py
@date: 2025/11/10 16:37
@desc:bert预训练模型
"""
import torch
from torch import nn
from transformers import AutoModel

from src import config


class TextSentimentAnalyze(nn.Module):
    def __init__(self)
        super().__init__()
        self.bert = AutoModel.from_pretrained(config.PRE_TRAINED_DIR/config.MODEL_NAME)
        self.linear = nn.Linear(self.bert.config.hidden_size,1) #输出一个特征值用于分类

    def forward(self,inputs_ids,attention_mask,token_type_ids):
        #[B,S]
        output = self.bert(input_ids=inputs_ids,attention_mask=attention_mask,token_type_ids=token_type_ids)
        # dict:last_hidden_state,pooler_state
        last_hidden_state = output.last_hidden_state

        cls_hidden_state = last_hidden_state[:,0,:]

        output = self.linear(cls_hidden_state).squeeze(-1)
        #[B]
        return output
