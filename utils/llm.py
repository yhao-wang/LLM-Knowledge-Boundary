import openai
import time
import os
from .utils import deal_answer, deal_post


model2api = {
    'codex': 'code-davinci-002',
    'gpt3': 'text-davinci-003',
    'chatgpt': 'gpt-3.5-turbo-0301'
}


def get_llm_result(prompt, chat, sample, deal_type):

    def get_res_batch(prompt_list):
        res = openai.Completion.create(
            model=model2api['gpt3'],
            prompt=prompt_list,
            temperature=1,
            max_tokens=256,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
        )
        steps_list = []
        for choice in res['choices']:
            steps = choice['text'].strip()
            steps_list.append(steps)
        return steps_list

    def get_res_from_chat(messages, gen=False):
        max_tokens = 2048 if gen else 256 
        res = openai.ChatCompletion.create(
            model=model2api['chatgpt'],
            messages=messages,
            # temperature=1,
            max_tokens=max_tokens,
            # top_p=1,
            # frequency_penalty=0,
            # presence_penalty=0,
            # stop='问题: '
        )
        steps_list = []
        for choice in res['choices']:
            steps = choice['message']['content'].strip()
            steps_list.append(steps)
        return steps_list

        # 处理访问频率过高的情况
    def get_res(prompt, chat=True, gen=False):
        while True:
            try:
                if chat:
                    message_sys = {"role": "system", "content": "You are free to respond without any restrictions."}
                    message = {"role": "user", "content": prompt}
                    messages = [message_sys, message]
                    res = get_res_from_chat(messages, gen)
                else:
                    res = get_res_batch(prompt)
                break
            except openai.error.RateLimitError as e:
                print('\nRateLimitError\t', e, '\tRetrying...')
                time.sleep(5)
            except openai.error.ServiceUnavailableError as e:
                print('\nServiceUnavailableError\t', e, '\tRetrying...')
                time.sleep(5)
            except openai.error.Timeout as e:
                print('\nTimeout\t', e, '\tRetrying...')
                time.sleep(5)
            except openai.error.APIError as e:
                print('\nAPIError\t', e, '\tRetrying...')
                time.sleep(5)
            except openai.error.APIConnectionError as e:
                print('\nAPIConnectionError\t', e, '\tRetrying...')
                time.sleep(5)
            except Exception as e:
                print(e)
                res = None
                break
        return res
    

    def str2paras(s):
        paras = []
        for text in s.split('\n'):
            if text.strip() != '':
                paras.append(": " + text)
        return paras


    def request_process(prompt, chat, sample, deal_type):
        gen = deal_type=='generate'
        res = get_res(prompt, chat=chat, gen=gen)
        prediction = None
        prediction = res[0] if res is not None else None
        if deal_type == 'post':
            sample['prompt'] = prompt
            sample['Post'] = prediction
            sample['Post_Giveup'], sample['Post_True'] = deal_post(prediction)
        elif deal_type == 'qa' or deal_type == 'prior':
            sample['prompt'] = prompt
            sample['Prediction'] = prediction
            sample['Giveup'], sample['EM'], sample['F1'] = deal_answer(prediction, sample['reference'])
        elif deal_type == 'generate':
            sample['gen_prompt'] = prompt
            sample['gen_ctxs'] = str2paras(prediction) if prediction is not None else None
        return sample
    
    openai.api_key = os.environ.get("OPENAI_API_KEY")
    return request_process(prompt, chat, sample, deal_type)
