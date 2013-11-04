# David Zbarsky: dzbarsky@wharton.upenn.edu
# Yaou Wang: yaouwang@wharton.upenn.edu

from nltk.corpus import PlaintextCorpusReader
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem.porter import PorterStemmer
from xml.etree import ElementTree
import math
import string
import random
import fileinput
import os
import itertools
import subprocess
import xml.etree.ElementTree as ET

'''
homework 3 by David Zbarsky and Yaou Wang
'''

#from hw1: functions for translating files/directory into tokens
def get_all_files(directory):
    # We assume that a filename with a . is always a file rather than a directory
    # IF we were passed a file, just return that file.  This simplifies representing documents because they need to handle single files.
    if directory.find('.') < 0:
        return PlaintextCorpusReader(directory, '.*').fileids()
    #if directory is a file return the file in a list
    return [directory]

def load_file_sentences(filepath):
    index = filepath.rfind('/')
    dir = filepath[:index]
    filepath = filepath[index + 1:]
    return sent_tokenize(PlaintextCorpusReader(dir, filepath).raw().lower())

def load_file_tokens(filepath):
    tokens = []
    for sentence in load_file_sentences(filepath):
        tokens.extend(word_tokenize(sentence))
    return tokens

def load_collection_tokens(directory):
    tokens = []
    for file in get_all_files(directory):
        tokens.extend(load_file_tokens(directory + '/' + file))
    return tokens
#end of functions from hw1

#returns a list of all words occurring >= 5 times in the directory
def extract_top_words(directory):
    tokens = load_collection_tokens(directory)
    freq = dict()
    top_words = []
    for token in tokens:
        if token in freq.keys():
            freq[token] += 1
        else:
            freq[token] = 1
    for word in freq.keys():
        if freq[word] >= 5:
            top_words.append(word)
    return top_words

def unigram_map_entry(filename, top_words):
    tokens = load_file_tokens(filename)
    output_list = [0] * len(top_words)
    for token in tokens:
        try:
            i = top_words.index(token)
            output_list[i] += 1
        except ValueError:
            pass
    return output_list

def get_mpqa_lexicon(lexicon_path):
    words = dict()
    for line in open(lexicon_path):
      type = line[5:line.find(' ')]
      line = line[line.find('word1'):]
      word = line[6:line.find(' ')]
      line = line[line.find('polarity'):]
      polarity = line[9:line.find(' ')]
      if word not in words:
          words[word] = [(type, polarity)]
      else:
          words[word].append((type, polarity))
    return words

def get_mpqa_features(file, dictionary):
    tokens = load_file_tokens(file)
    l = [0, 0, 0]
    for token in tokens:
        if token in dictionary.keys():
            for t in dictionary[token]:
                if t[1] == 'positive':
                    l[0] += 1
                elif t[1] == 'negative':
                    l[1] += 1
                elif t[1] == 'neutral':
                    l[2] += 1
    return l

def get_mpqa_features_wordtype(file, dictionary):
    tokens = load_file_tokens(file)
    l = [0, 0, 0, 0, 0, 0]
    for token in tokens:
        if token in dictionary.keys():
            for t in dictionary[token]:
                if t[0] == 'strongsubj':
                    if t[1] == 'positive':
                        l[0] += 1
                    elif t[1] == 'negative':
                        l[1] += 1
                    elif t[1] == 'neutral':
                        l[2] += 1
                if t[0] == 'weaksubj':
                    if t[1] == 'positive':
                        l[3] += 1
                    elif t[1] == 'negative':
                        l[4] += 1
                    elif t[1] == 'neutral':
                        l[5] += 1
    return l

def get_geninq_lexicon(lexicon_path):
    words = dict()
    for line in open(lexicon_path):
        word = line[:line.find('\t')]
        positive = 1 if (line.find('Pstv') is not -1 or line.find('Pos') is not -1) else 0
        negative = 1 if (line.find('Ngtv') is not -1 or line.find('Neg') is not -1) else 0
        strong = 1 if line.find('Strng') is not -1 else 0
        weak = 1 if line.find('Weak') is not -1 else 0
        words[word] = (positive, negative, strong, weak)
    return words

def get_geninq_features(filename, geninq_dict):
    l = [0, 0, 0, 0]
    for token in load_file_tokens(filename):
        if token in geninq_dict:
            l = [a + b for a, b in zip(l, geninq_dict[token])]
    return l

#helper to remove adjacent duplicates
def remove_adj_dup(l):
    for i in range(len(l) - 1, 0, -1):
        if l[i] == l[i - 1]:
            l[i] = '0'
    return l

def extract_named_entities(xml_file_name):
    try:
        tree = ET.parse(xml_file_name)
        root = tree.getroot()
        l = []
        names = [0, 0, 0, 0, 0]
        for token in root.iter('token'):
            l.append(token.find('NER').text)
        l = remove_adj_dup(l)
        for ner in l:
            if ner == 'ORGANIZATION':
                names[0] += 1
            if ner == 'PERSON':
                names[1] += 1
            if ner == 'LOCATION':
                names[2] += 1
            if ner == 'MONEY':
                names[3] += 1
            if ner == 'DATE':
                names[4] += 1
        return names
    except:
        return [0, 0, 0, 0, 0]

def extract_pos(training_xml_path, pos_list):
    words = []
    if training_xml_path.find('.xml') is not -1:
        paths = [training_xml_path]
    else:
        paths = [training_xml_path + '/' + file for file in get_all_files(training_xml_path)]

    for path in paths:
        try:
            tree = ElementTree.parse(path)
            for token in tree.getroot().iter('token'):
                if token.find('POS').text in pos_list:
                    words.append(token.find('word').text)
        except:
            pass
    return words

def extract_adjectives(training_xml_path):
    return extract_pos(training_xml_path, ['JJ', 'JJR', 'JJS'])

def extract_verbs(training_xml_path):
    return extract_pos(training_xml_path, ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'])

def map_adjectives(xml_filename, adj_list):
    adjectives = extract_adjectives(xml_filename)
    return [1 if t in adjectives else 0 for t in adj_list]

def map_verbs(xml_filename, verb_list):
    verbs = extract_verbs(xml_filename)
    return [1 if t in verbs else 0 for t in verb_list]

def extract_verb_dependencies(xml_path):
    dep_dict = dict()
    #finds the list of verb dependencies
    verb_deps = load_file_tokens('/home1/c/cis530/hw3/verb_deps.txt')
    if xml_path.find('.xml') is not -1:
        paths = [xml_path]
    else:
        paths = [xml_path + '/' + file for file in get_all_files(xml_path)]
    for path in paths:
        try:
            tree = ElementTree.parse(path)
            for basic_dep in tree.getroot().iter('basic-dependencies'):
                for dep in basic_dep.findall('dep'):
                    name = dep.get('type')
                    if name in verb_deps:
                        t = (name, dep.find('governor').text, dep.find('dependent').text)
                        if t in dep_dict.keys():
                            dep_dict[t] += 1
                        else:
                            dep_dict[t] = 1
        except:
           pass
    dep_list = []
    for dep in dep_dict.keys():
        if dep_dict[dep] >= 5:
            dep_list.append(dep)
    return dep_list

def map_verb_dependencies(xml_filename, dependency_list):
    array = [0] * len(dependency_list)
    tree = ElementTree.parse(xml_filename)
    for basic_dep in tree.getroot().iter('basic-dependencies'):
        for dep in basic_dep.findall('dep'):
            try:
                t = (dep.get('type'), dep.find('governor').text, dep.find('dependent').text)
                i = dependency_list.index(t)
                array[i] += 1
            except:
                pass
    return array



def main():
    #top_words = extract_top_words('/home1/c/cis530/hw3/data')
    #print unigram_map_entry('/home1/c/cis530/hw3/data/6285515.txt', top_words)
    #dic = get_mpqa_lexicon('/home1/c/cis530/hw3/mpqa-lexicon/subjclueslen1-HLTEMNLP05.tff')
    #print get_mpqa_features('/home1/c/cis530/hw3/data/6285515.txt', dic)
    #print get_mpqa_features_wordtype('/home1/c/cis530/hw3/data/6285515.txt', dic)
    #print get_mpqa_lexicon('/home1/c/cis530/hw3/mpqa-lexicon/subjclueslen1-HLTEMNLP05.tff')['mean']
    #gi_dict = get_geninq_lexicon('/home1/c/cis530/hw3/gi-lexicon/inquirerTags.txt')
    #print gi_dict["make"]
    #print gi_dict["malady"]
    #print get_geninq_features('data/2067818.txt', gi_dict)
    #command line call to run CoreNLP
    #os.system('java -cp stanford-corenlp-2012-07-09.jar:stanford-corenlp-2012-07-06-models.jar:xom.jar:joda-time.jar -Xmx3g edu.stanford.nlp.pipeline.StanfordCoreNLP -annotators tokenize,ssplit,pos,lemma,ner,parse -filelist datafilelist.txt -outputDirectory data_result')
    #os.system('java -cp stanford-corenlp-2012-07-09.jar:stanford-corenlp-2012-07-06-models.jar:xom.jar:joda-time.jar -Xmx3g edu.stanford.nlp.pipeline.StanfordCoreNLP -annotators tokenize,ssplit,pos,lemma,ner,parse -filelist test_datafilelist.txt -outputDirectory test_data_result')
    #print extract_named_entities('data_result/71964.txt.xml')
    #print extract_adjectives('data_result')
    dictionary = extract_verb_dependencies('data_result')
    print dictionary
    print map_verb_dependencies('data_result/334701.txt.xml', dictionary)
    #print map_adjectives('data_result/71964.txt.xml', ['big', 'small', 'public'])

if __name__ == "__main__":
    main()
