import numpy as np
import pandas as pd
from glob import glob
import PyPDF2
import textract
import math
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import PunktSentenceTokenizer
import re
import re
import string
import fitz
from operator import itemgetter
from os import path
import os
from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse

def textpreProcessing():
    fileNames = []
    pdfNames = []
    imagenames = []
    wordNames = []
    # print(os.listdir('.'))
    # print(os.path.realpath(__file__))
    for nam in (glob(path.join(os.path.realpath(os.path.dirname(__file__)),"*.pdf"))):
        
        fileNames.append(nam)
        pdfNames.append(nam)
        # print(nam, '\t', type(nam))

    #     print('All word files')
    for nam in glob('*.doc'):
        fileNames.append(nam)
        wordNames.append(nam)
    #     print(nam, '\t', type(nam))
    for nam in glob('*.docx'):
        wordNames.append(nam)
        fileNames.append(nam)
    #     print(nam, '\t', type(nam))

        # print('All image files')
    for nam in glob('*.jpg'):
        imagenames.append(nam)
        fileNames.append(nam)
    #     print(nam, '\t', type(name))
    for nam in glob('*.jpeg'):
        imagenames.append(nam)
        fileNames.append(nam)
    #     print(nam, '\t', type(name))
    for nam in glob('*.png'):
        imagenames.append(nam)
        fileNames.append(nam)
    #     print(nam, '\t', type(nam))

    # print(pdfNames, '-----------------------')
    nltk.download('stopwords')
    nltk.download('punkt')


    # print('All pdfs')
    # print(glob.glob('*.pdf'),'\n','-----------------------------------------')

    def fonts(doc, granularity=False):
        styles = {}
        font_counts = {}

        for page in doc:
    #         print(type(page))
            blocks = page.get_text("dict")["blocks"]
            for b in blocks:  # iterate through the text blocks
                if b['type'] == 0:  # block contains text
                    for l in b["lines"]:  # iterate through the text lines
                        for s in l["spans"]:  # iterate through the text spans
                            if granularity:
                                identifier = "{0}_{1}_{2}_{3}".format(s['size'], s['flags'], s['font'], s['color'])
                                styles[identifier] = {'size': s['size'], 'flags': s['flags'], 'font': s['font'],
                                                    'color': s['color']}
                            else:
                                identifier = "{0}".format(s['size'])
                                styles[identifier] = {'size': s['size'], 'font': s['font']}

                            font_counts[identifier] = font_counts.get(identifier, 0) + 1  # count the fonts usage

        font_counts = sorted(font_counts.items(), key=itemgetter(1), reverse=True)

        if len(font_counts) < 1:
            raise ValueError("Zero discriminating fonts found!")

        return font_counts, styles

    def font_tags(font_counts, styles):
        p_style = styles[font_counts[0][0]]  # get style for most used font by count (paragraph)
        p_size = p_style['size']  # get the paragraph's size

        # sorting the font sizes high to low, so that we can append the right integer to each tag 
        font_sizes = []
        for (font_size, count) in font_counts:
            font_sizes.append(float(font_size))
        font_sizes.sort(reverse=True)

        # aggregating the tags for each font size
        idx = 0
        size_tag = {}
        for size in font_sizes:
            idx += 1
            if size == p_size:
                idx = 0
                size_tag[size] = '<p>'
            if size > p_size:
                size_tag[size] = '<h{0}>'.format(idx)
            elif size < p_size:
                size_tag[size] = '<s{0}>'.format(idx)

        return size_tag

    # size_tag = font_tags(fontCount,styles)

    def headers_para(doc, size_tag):
        header_para = []  # list with headers and paragraphs
        first = True  # boolean operator for first header
        previous_s = {}  # previous span

        for page in doc:
            blocks = page.get_text("dict")["blocks"]
            for b in blocks:  # iterate through the text blocks
                if b['type'] == 0:  # this block contains text

                    # REMEMBER: multiple fonts and sizes are possible IN one block

                    block_string = ""  # text found in block
                    for l in b["lines"]:  # iterate through the text lines
                        for s in l["spans"]:  # iterate through the text spans
                            if s['text'].strip():  # removing whitespaces:
                                if first:
                                    previous_s = s
                                    first = False
                                    block_string = size_tag[s['size']] + s['text']
                                else:
                                    if s['size'] == previous_s['size']:

                                        if block_string and all((c == "|") for c in block_string):
                                            # block_string only contains pipes
                                            block_string = size_tag[s['size']] + s['text']
                                        if block_string == "":
                                            # new block has started, so append size tag
                                            block_string = size_tag[s['size']] + s['text']
                                        else:  # in the same block, so concatenate strings
                                            block_string += " " + s['text']

                                    else:
                                        header_para.append(block_string)
                                        block_string = size_tag[s['size']] + s['text']

                                    previous_s = s

                        # new block started, indicating with a pipe
                        block_string += "|"

                    header_para.append(block_string)

        return header_para

    text_df = pd.DataFrame(
        columns=['Text', 'Clean'] # paragraph id, actual text and cleaned text respectively
    )
    #  df.loc[len(df.index)] = ['Amy', 89, 93]
    # text_df
    unique = set()
    regP = r'<p>'
    regPipe = r'\|+'
    regNo = r'\d'
    regNo1 = r'[^A-Za-z]'
    count = 0

    for pdfName in pdfNames:
        doc = fitz.open(pdfNames[0])
        # type(doc)
        fontCount, styles = fonts(doc,granularity=False)
        size_tag = font_tags(fontCount,styles)
        header_para = headers_para(doc,size_tag)

        
        for text in header_para:
            if re.match(regP, text):
                count +=1
                text = re.sub(regP,"",text)
                newText = re.split(regPipe, text)
                text = ' '.join(newText)
                # create a stemmer object
                stemmer = PorterStemmer()

                # create a sentence tokenizer object
                sentence_tokenizer = PunktSentenceTokenizer()

                # tokenize the text string into sentences
                sentences = sentence_tokenizer.tokenize(text)

                # create a set of stopwords
                stop_words = set(stopwords.words('english'))

                # loop through each sentence and tokenize, stem, and remove stop words
                stemmed_tokens = []
                clean_tokens = []
                for sentence in sentences:
                    # tokenize sentence into words
                    words = nltk.word_tokenize(sentence)
                    # stem each word and remove stop words
                    for word in words:
                        # stem the word
                        stemmed_word = stemmer.stem(word)
                        # check if the stemmed word is a stop word
                        if stemmed_word not in stop_words:
                            # add the stemmed word to the list of stemmed tokens
                            stemmed_tokens.append(stemmed_word)

                # print the original text and the stemmed tokens
        #         print("Original Text: ", text)
        #         print("Stemmed Tokens: ", stemmed_tokens)
                for w in stemmed_tokens:
                    if not re.match(regNo1,w):
                        unique.add(w)
                        clean_tokens.append(w)
                text_df.loc[len(text_df.index)] = [text, clean_tokens]

    # print(unique)
    # print(count)
    # print(len(sorted(unique)))
    # print('------------------------------------------------------------------------------------------------------------------')
    # print(sorted(unique))
    # print('------------------------------------------------------------------------------------------------------------------')

    tf_matrix = pd.DataFrame(columns=sorted(unique))
    for i in range(len(text_df.index)):
        tf_matrix.loc[len(tf_matrix.index)] = [0 for i in range(len(unique))]
    # tf_matrix

    count = 0
    for i in range(len(text_df.index)):
        l = text_df.loc[i]['Clean']
        
        for j in l:
    #         print(j)
            tf_matrix.loc[i][j] +=1
    #         print(tf_matrix.loc[i][j])
            count +=1


    df_matrix = pd.DataFrame(columns=sorted(unique))
    # print(df_matrix)
    df_matrix.loc[0] = [0.0 for i in range(len(unique))]

    # print(df_matrix)

    for i in range(len(text_df.index)):
        l = text_df.loc[i]['Clean']
        for j in l:
            df_matrix.loc[0][j] +=1

    N = len(text_df.index)
    # print(N)
    for i in sorted(unique):
    #     print(N/df_matrix.loc[0][i], i, math.log2(N/df_matrix.loc[0][i]))
        if df_matrix.loc[0][i] != 0:
            df_matrix.loc[0][i] = math.log2(N/(df_matrix.loc[0][i]))

    text_df.to_excel('Text.xlsx')
    tf_matrix.to_excel('tf.xlsx')
    df_matrix.to_excel('df.xlsx')
    # print(text_df)
    # print(tf_matrix)
    # print(df_matrix)

def Retrieval(query):
    #pass
    # pass
    text_df = pd.DataFrame(pd.read_excel('Text.xlsx'))
    tf_matrix = pd.DataFrame(pd.read_excel('tf.xlsx'))
    df_matrix = pd.DataFrame(pd.read_excel('df.xlsx'))
    
    stemmer = PorterStemmer()
    sentence_tokenizer = PunktSentenceTokenizer()
    stop_words = set(stopwords.words('english'))
    stemmed_tokens = [] #contains the stemmed tokens of query
    # for text in query:
    query = nltk.word_tokenize(query)
    #     print(text)
    for word in query:
        stemmed_word = stemmer.stem(word)
        if stemmed_word not in stop_words:
            stemmed_tokens.append(stemmed_word)

    
    Scores = dict()
    count = 0
    for i in range(len(tf_matrix.index)):
        Scores[i] = 0
        for j in stemmed_tokens:
    #         print(j)/
            if j in tf_matrix.columns and j in df_matrix.columns:
    #             print(tf_matrix.loc[i][j], df_matrix.loc[0][j])
                Scores[i] += ((tf_matrix.loc[i][j])*(df_matrix.loc[0][j]))
                count+=1
    # count
    def get_keys_of_largest_values(d):
        # Get the items of the dictionary and sort them by value in descending order
        sorted_items = sorted(d.items(), key=lambda x: x[1], reverse=True)
        # Extract the keys of the first three items
        keys = [k for k, v in sorted_items[:3]]
        l = [text_df.loc[k]['Text'] for k in keys]
        return l
    return get_keys_of_largest_values(Scores)
    
app = Flask(__name__)

@app.route("/")
def hello():
    return "Hello, World!"

@app.route("/sms", methods=['POST'])
def sms_reply():
    """Respond to incoming calls with a simple text message."""
    # Fetch the message
    msg = request.form.get('Body')

    # Create reply
    query = msg
    print(msg)
    retText = Retrieval(query)
    print(retText)
    resp = MessagingResponse()
    resp.message("{}".format(retText))
    return str(resp)

if __name__ == '__main__':
    print("starting preprocessing")
    textpreProcessing()
    print("done with preprocessing")
    app.run(debug=True)
    