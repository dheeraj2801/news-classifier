from .imports import re, nlp, torch, MAX_LENGTH, np, vocab
def preprocess(sentence):
        sentence = sentence.lower()
        sentence = re.sub(r'https?:\/\/.*[\r\n]*', '', str(sentence)) #removing urls
        sentence = re.sub(r'^RT[\s]+', '', str(sentence)) #removing retweets
        sentence = re.sub('@[^\s]+','',str(sentence)) #removing handlers
        sentence = re.sub(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]+',
        '',str(sentence)) #removing emojis
        sentence = re.sub("[.,/\\{}~`!@#$%^&*()_=+-;:?><]","",sentence) #removing symbols
        sentence = re.sub(r'\s+',' ',str(sentence)) #removing extra-spaces
        sentence = sentence.rstrip() #removing trailing spaces
        sentence = sentence.lstrip() #removing leading spaces
        return sentence
def tokenizer(sentence):
    tokens = []
    sentence = preprocess(sentence)
    for token in nlp(sentence):
        if token.text == "nt":
            tokens.append("not")
        elif not (token.is_punct | token.is_digit | 
        token.is_space | token.is_quote | token.is_stop | token.like_num |
        token.like_email | token.like_url | token.is_left_punct |
        token.is_right_punct | token.is_space):
            tokens.append(str(token))
    return tokens
def data_maker(sentence):
    sentence = preprocess(sentence)
    tokens = tokenizer(sentence)
    tokens = [2]+vocab(tokens)
    size = min(len(tokens), MAX_LENGTH)
    ids = np.zeros((MAX_LENGTH), dtype=np.int64)
    masks =  np.ones((MAX_LENGTH), dtype=np.bool_)
    ids[:size] = tokens[:size]
    ids[size-1 if size>=MAX_LENGTH else size] = 3
    masks[:size if size>=MAX_LENGTH else size+1] = False
    return torch.from_numpy(ids).view(1,-1),torch.from_numpy(masks).view(1,-1)