from openai import OpenAI
import jsonlines
import requests
import os
import time
# For prerequisites running the following sample, visit https://help.aliyun.com/document_detail/611472.html
from http import HTTPStatus
from dashscope import Application
import pandas as pd
import http.client
import json

client = OpenAI(
    base_url="",
    api_key="",
)

def get_response(messages, model='gpt-4o', json_format=False):
    if json_format:
        if model == 'gpt-4o':
            message = client.chat.completions.create(
                messages=messages,
                model=model,
                response_format={"type": "json_object"},
            )
        else:
            message = client.chat.completions.create(
                messages=messages,
                model=model,
                response_format={"type": "json_object"}
            )
    else:
        if model == 'gpt-4o':
            message = client.chat.completions.create(
                messages=messages,
                model=model,
            )
        else:
            message = client.chat.completions.create(
                messages=messages,
                model=model,
            )
    return message.choices[0].message.content

