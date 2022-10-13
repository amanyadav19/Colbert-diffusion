from transformers import AutoTokenizer

class QuestionTokenizer():
    def __init__(self, total_maxlen=128, base="bert-base-uncased"):
        self.total_maxlen = total_maxlen
        self.tok = AutoTokenizer.from_pretrained(base)

    def encode(self, questions):
        assert type(questions) in [list, tuple], type(questions)

        encoding = self.tok(questions, padding='longest', truncation='longest_first',
                            return_tensors='pt', max_length=self.total_maxlen, add_special_tokens=True)

        return encoding
