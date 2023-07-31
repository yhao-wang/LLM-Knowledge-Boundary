prompt_dict = {
    'qa': {
        'none': 'Answer the following question based on your internal knowledge with one or few words.\nQuestion: {question}{paras}{prediction}',
        'ra': 'Given the following information: \n{paras}\nAnswer the following question based on the given information or your internal knowledge with one or few words without the source.\nQuestion: {question}{prediction}',
        'tail': '\nAnswer: ',
    },
    'prior': {
        'none': 'Are you sure to accurately answer the following question based on your internal knowledge, if yes, you should give a short answer with one or few words, if no, you should answer \"Unknown\"\nQuestion: {question}{paras}{prediction}',
        'ra': 'Given the following information: \n{paras}\nCan you answer the following question based on the given information or your internal knowledge, if yes, you should give a short answer with one or few words, if no, you should answer \"Unknown\".\nQuestion: {question}{prediction}',
        'tail': '\nAnswer: ',
    },
    'post': {
        'none': 'Can you judge if the following answer about the question is correct based on your internal knowledge, if yes, you should answer True or False, if no, you should answer \"Unknown\".\nQuestion: {question}{paras}\nAnswer: {prediction}',
        'ra': 'Given the following information: \n{paras}\nCan you judge the if the following answer about the question is correct based on the given information or your internal knowledge, if yes, you should answer True or False, if no, you should answer \"Unknown\".\nQuestion: {question}\nAnswer: {prediction}',
        'tail': '\nJudgement is: ',
    },
    'generate': {
        'none': 'I want you to act as a Wikipedia page. I will give you a question, and you will provide related passages in the format of a Wikipedia page which contains 10 paragraphs split by \"\n\n\". Your summary should be informative and factual, covering the key phrases that could answer the following question.\nQuestion: {question}{paras}{prediction}',
        'ra': '',
        'tail': '',
    }
}


def get_prompt(sample, args):
    paras = ""
    prompt = prompt_dict[args.type]['none']
    if args.ra != 'none':
        ra_dict = args.ra
        i = 0
        doc = []
        for k, v in ra_dict.items():
            v = min(v, len(sample[k]))
            for j in range(v):
                doc.append(("Passage-%d" % i) + sample[k][j])
                i += 1
        paras = '\n'.join(doc)
        prompt = prompt_dict[args.type]['ra']
    tail = prompt_dict[args.type]['tail'] if not args.usechat else ""
    prediction = sample['Prediction'] if args.type == 'post' else ""
    prompt = prompt.format(question=sample['question'], paras=paras, prediction=prediction) + tail
    return prompt
