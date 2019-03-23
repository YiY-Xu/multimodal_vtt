import numpy as np
import re
def tokenizeDoc ( cur_doc ):
    return re. findall ( '\\w+', cur_doc )

def compute_text_prob (text, context_prob):
# This function output the new probability for each sentence condition on context
    text_prob = []
    for line in text:
        prob = 1
        words = np.array(tokenizeDoc(line))
        for word in words:
            if word not in context_prob.keys():
                prob = prob * 0.01
            else:
                prob = prob * context_prob[word]

        text_prob.append(prob)

    return np.array(text_prob)

def compute_context_prob (context):
# This function compute the naive bayes probability for every word in context
    word_counter = dict()
    for line in context:
        for word in np.array(tokenizeDoc(line)):
            if word not in word_counter.keys():
                word_counter[word] = 1
            else:
                word_counter[word] += 1

    word_prob = dict()
    word_sum = sum(word_counter.values())
    for key in word_counter.keys():
        word_prob[key] = word_counter[key] / word_sum

    return word_prob

# test case
output_text = ['I like golf.', 'I like gulf.', 'I like girl.']
training_context = ['today I play golf', 'I like golf']

b = compute_context_prob (training_context)
print ('Dict of probability:\n', b)

c = compute_text_prob (output_text, b)
print ('Prediced prob condition on context:\n', c)
