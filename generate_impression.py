# *-* coding: utf-8 *-*
"""
因为输出可能超过模型最大限制，我们使用多轮对话生成印象
"""

import pandas as pd
import re
import numpy as np
import time
import csv
import tqdm
import os
from openai import OpenAI
import qianfan
import logging

# log保存到本地:错误
import logging

# 创建logger对象
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # 设置logger的总体日志级别
# 创建文件处理器，用于错误日志
file_handler = logging.FileHandler('impression_generation.log', mode='a')
file_handler.setLevel(logging.INFO)  # 设置file_handler的日志级别
# 创建流处理器，用于控制台输出
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)  # 设置stream_handler的日志级别
# 创建格式化器
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
# 将格式化器添加到处理器
file_handler.setFormatter(formatter)
stream_handler.setFormatter(formatter)
# 将处理器添加到logger
logger.addHandler(file_handler)
logger.addHandler(stream_handler)


class ImpressionGenerator:

    def __init__(self, which_api):
        self.which_api = which_api
        if self.which_api == "chatkore":
            # chatkore
            self.url = 'https://api.chatkore.com/v1/chat/completions'
            self.base_url = 'https://api.chatkore.com/v1'
            self.api_key = 'YOUR_API_KEY'  #=======需要修改的参数=======
        elif self.which_api == "uiuiapi":
            self.url = 'https://uiuiapi.com/v1/chat/completions'
            self.base_url = 'https://uiuiapi.com/v1'
            self.api_key = '' #=======需要修改的参数，建议使用这个api=======
        elif self.which_api == "moonshot":
            self.url = 'https://api.moonshot.cn/v1/files'
            self.base_url = 'https://api.moonshot.cn/v1'
            self.api_key = 'YOUR_API_KEY' #=======需要修改的参数=======
        elif self.which_api == "gptw.top":
            self.base_url = 'https://api.pro365.top/v1'
            self.api_key = 'YOUR_API_KEY' #=======需要修改的参数=======
        elif self.which_api == "gptgod":
            self.base_url = 'https://api.gptgod.online/v1'
            self.api_key = 'YOUR_API_KEY' #=======需要修改的参数=======
        elif self.which_api == "qianfan":
            self.base_url = None
            self.api_key = None
        elif self.which_api == "qianwen":
            self.base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
            self.api_key = '' #=======需要修改的参数=======

        # Openai
        self.client = OpenAI(
            api_key=self.api_key,
            base_url= self.base_url,
        )

        # Qianfan
        # 动态获取最新模型列表依赖 IAM Access Key 进行鉴权，使用应用 AK 鉴权时不支持该功能
        os.environ["QIANFAN_ACCESS_KEY"] = "dcdef043d7b04c1bbd07eecc37519190"
        os.environ["QIANFAN_SECRET_KEY"] = "8d20c79edb4343b1b08b62e8cab7f774"

        self.get_responese = {
            "chatkore": self.get_responese_using_openai,
            "uiuiapi": self.get_responese_using_openai,
            "moonshot": self.get_responese_using_openai,
            "gptw.top": self.get_responese_using_openai,
            "gptgod": self.get_responese_using_openai,
            "qianwen": self.get_responese_using_openai,
            "qianfan": self.get_responese_using_qianfan,
        }
        self.get_content = {
            "chatkore": self.get_content_using_openai,
            "uiuiapi": self.get_content_using_openai,
            "moonshot": self.get_content_using_openai,
            "gptw.top": self.get_content_using_openai,
            "gptgod": self.get_content_using_openai,
            "qianwen": self.get_content_using_openai,
            "qianfan": self.get_content_using_qianfan,
        }

    def load_prompt(self, prompt_path):
        with open(prompt_path, "r", encoding="utf-8") as f:
            prompt = f.read()
        
        prompts = prompt.split("#==========SEPARATION==========#")
        return prompts
    
    def load_findings(self, 
                      excel_path=None, 
                      outfile=None, 
                      uid_col=None, 
    ):
        # # 如何确定编码格式
        # with open(excel_path, 'rb') as f:
        #     result = chardet.detect(f.read())
        #     encoding = result['encoding']

        if excel_path.endswith(".csv") or excel_path.endswith(".CSV"):
            data = pd.read_csv(excel_path, encoding="gbk")
        elif excel_path.endswith(".xlsx"):
            data = pd.read_excel(excel_path)
        elif excel_path.endswith(".xls"):
            data = pd.read_excel(excel_path)
        elif excel_path.endswith("txt"):
            with open(excel_path, "r", encoding="utf-8") as f:
                data = f.read()
            data = pd.DataFrame([data])
            data[uid_col] = np.arange(0, len(data))
            # data[impression_col] = ""
        
        # 如果有out_file，则提取extract data 和outfile不同的uid
        if outfile and os.path.exists(outfile):
            if outfile.endswith(".csv"):
                exist_data = pd.read_csv(outfile, encoding="gbk")
            elif outfile.endswith(".xlsx"):
                exist_data = pd.read_excel(outfile)
            elif outfile.endswith(".xls"):
                exist_data = pd.read_excel(outfile)
            
            exist_data = exist_data.dropna(subset=[uid_col])
            data = data.dropna(subset=[uid_col])

            # 去掉重复的uid
            exist_data = exist_data.drop_duplicates(subset=[uid_col])
            data = data.drop_duplicates(subset=[uid_col])

            exist_uid = exist_data[uid_col].astype(np.int64).astype(str).tolist()
            data_uid = data[uid_col].astype(np.int64).astype(str).tolist()
            remain_uid = np.in1d(data_uid, exist_uid, invert=True)  # invert=True表示取不在exist_uid中的元素
            # 用其它方法找到不在exist_uid中的元素
            remain_uid = [uid not in exist_uid for uid in data_uid]
            if sum(remain_uid) == 0:
                print("All data has been processed!")
                return None, None, None
            data = data[remain_uid]

        pid = data[uid_col].tolist()
        return pid, data
    
    def get_responese_using_openai(self, messages, model, stream, max_tokens, temperature, top_p):
        response = self.client.chat.completions.create(
            messages=messages,
            model=model,
            max_tokens=max_tokens,
            stream=stream,
            temperature=temperature,
            # top_p=top_p,
        )
        return response
    
    def get_responese_using_qianfan(self, messages, model, stream, max_tokens, temperature, top_p):
        chat_comp = qianfan.ChatCompletion(model=model)
        response = chat_comp.do(
            messages=messages,
            model=model,
            max_output_tokens=max_tokens,
            stream=stream,
            temperature=temperature,
            # top_p=0.5,
        )
        return response
    
    def get_content_using_openai(self, response, stream):
        if stream:
            content = ''
            print("\n")
            for chunk in response:
                if len(chunk.choices) > 0:
                    print(chunk.choices[0].delta.content or "", end="")
                    content += chunk.choices[0].delta.content or ""
        else:
            try:
                content = response.choices[0].message.content
                print("\nAI输出:")
                print(content)
            except:
                content = response.text

        totle_tokens = None
        return content, totle_tokens
    
    def get_content_using_qianfan(self, response, stream):
        if stream:
            content = ""
            tokens = {'prompt_tokens': 0, 'completion_tokens': []}
            print("\n")
            for line in response:
                text = line['result']
                tokens['prompt_tokens'] = line.get('usage').get('prompt_tokens')
                tokens['completion_tokens'].append(line.get('usage').get('completion_tokens'))
                print(text, end='')
                content += text
        else:
            content = response["result"]
            tokens = {'prompt_tokens': response.get('usage').get('prompt_tokens'), 'completion_tokens': response.get('usage').get('completion_tokens')}
            print("\nAI输出:")
            print(content)

        totle_tokens = tokens['prompt_tokens'] + tokens['completion_tokens'][-1]
        return content, totle_tokens

    def extract_json(self, content):
        """
        如果有{}，则提取所有{}中的内容，返回最后一个{}中的内容
        如果没有完整的{}，则返回content
        """
        # 反转字符串
        reversed_text = content[::-1]

        # 在反转后的字符串中匹配第一个 '}' 和 '{' 之间的内容
        match_impression = re.search(r'\}(.*?)\{', reversed_text, re.DOTALL)
        
        if match_impression:
            # 反转匹配到的内容以恢复原始顺序
            impression = match_impression.group(1)[::-1]
        else:
            impression = content

        # # 提取治疗方式
        # match_treatment = re.search(r'\{(.*?)\}', content, re.DOTALL)
        # if match_treatment:
        #     treatment = match_treatment.group(1)[::-1]
        # else:
        #     treatment = "None"
        return impression

    def main(self, 
             prompt_path=None,
             findings_path=None,
             include_info=None,
             uid_col=None,
             model=None,
             stream=None,
             max_tokens=None,
             temperature=None,
             top_p=None,
             outfile=None,
    ):
        prompts = self.load_prompt(prompt_path)
        
        pid, data = self.load_findings(findings_path, outfile, uid_col)
        data_include = data[include_info]

        headers = ["检查号", "患者资料", "AI输出", "耗时(s)","Prompt", "Response"]
        if outfile and (os.path.exists(outfile) is False):  # 如果给定了输出文件且文件不存在，则创建文件并写入表头
            with open(outfile, 'w', newline='', encoding='gbk') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(headers)

        for i in tqdm.tqdm(range(len(pid))):
            try:
                start_time = time.time()
                messages = []
                messages.append({"role": "user", "content": prompts[0]})
                messages.append({"role": "assistant", "content": "收到，我将严格按照您的要求进行处理。"})
                info = data_include.iloc[i].to_dict()
                messages.append({"role": "user", "content": f"患者资料如下：[{info}]，请等指令"})
                messages.append({"role": "assistant", "content":"我已收到患者资料，等待您的指令..."})
                response_contents = ""
                for prompt in prompts[1:]:
                    messages.append({"role": "user", "content": f"你继续完成如下步骤：[{prompt}]"})
                    response = self.get_responese[self.which_api](messages, model, stream, max_tokens, temperature, top_p)
                    response_content, tokens = self.get_content[self.which_api](response, stream)
                    logger.info(f"Model: {model};Tokens: {tokens}")
                    messages.append({"role": "assistant", "content":response_content})
                    response_contents += "\n" + response_content

                end_time = time.time()
                time_cost = end_time - start_time
                print(f"\nTime cost: {time_cost}")

                # save
                impression_from_ai = self.extract_json(response_contents)
                if outfile:
                    with open(outfile, 'a', newline='', encoding='gbk') as csvfile:
                        csv_writer = csv.writer(csvfile)
                        csv_writer.writerow([pid[i], info, impression_from_ai, time_cost, messages, response_contents])
            
            except Exception as e:
                print(f"API:[{self.which_api}];Error occurs in [{pid[i]}] with error: [{e}]\n")
                logger.error(f"Error: {e}")


if __name__ == '__main__':
    #=======需要修改的参数=======
    prompt_path = "./提示词.txt"
    findings_path = './例子.CSV'
    include_info = ["性别", "年龄", "影像所见", "既往史", "个人史", "家族史", "结节倍增时间"]
    uid_col='姓名'
    which_api = "uiuiapi" #"qianfan"
    model=  "claude-3-5-sonnet-20240620"  # "ERNIE-4.0-Turbo-8K"
    time_str = time.strftime("%Y%m%d%H%M%S", time.localtime())
    outfile = './response_' + time_str + '.csv'
    stream = True
    max_tokens = None
    temperature = 0
    #=======需要修改的参数=======

    ig = ImpressionGenerator(which_api)
    ig.main(prompt_path,findings_path, include_info, uid_col, model, stream, max_tokens, temperature, outfile)
