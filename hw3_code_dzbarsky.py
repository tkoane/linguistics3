# David Zbarsky: dzbarsky@wharton.upenn.edu
# Yaou Wang: yaouwang@wharton.upenn.edu

from nltk.corpus import PlaintextCorpusReader
from nltk.tokenize import sent_tokenize, word_tokenize
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
    if index < 0:
        sents = sent_tokenize(PlaintextCorpusReader('.', filepath).raw().lower())
    else:
        sents = sent_tokenize(PlaintextCorpusReader(filepath[:index], filepath[index+1:]).raw().lower())
    return sents

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
      word = line[6:line.find(' ')].lower()
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
        word = line[:line.find('\t')].lower()
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
            tree = ET.parse(path)
            for token in tree.getroot().iter('token'):
                if token.find('POS').text in pos_list:
                    words.append(token.find('word').text.lower())
        except:
            pass
    return list(set(words))

def extract_adjectives(training_xml_path):
    return extract_pos(training_xml_path, ['JJ', 'JJR', 'JJS'])

def extract_verbs(training_xml_path):
    return extract_pos(training_xml_path, ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'])

# This is for part 7
def extract_nouns(training_xml_path):
    return extract_pos(training_xml_path, ['NN', 'NNS', 'NNP', 'NNPS'])

def map_adjectives(xml_filename, adj_list):
    adjectives = extract_adjectives(xml_filename)
    return [1 if t in adjectives else 0 for t in adj_list]

def map_verbs(xml_filename, verb_list):
    verbs = extract_verbs(xml_filename)
    return [1 if t in verbs else 0 for t in verb_list]

# This is for part 7
def map_nouns(xml_filename, noun_list):
    nouns = extract_nouns(xml_filename)
    return [1 if t in nouns else 0 for t in noun_list]

def extract_verb_dependencies(xml_path):
    dep_dict = dict()
    #finds the list of verb dependencies
    verb_deps = load_file_tokens('/home1/c/cis530/hw3/verb_deps.txt')
    print verb_deps
    if xml_path.find('.xml') is not -1:
        paths = [xml_path]
    else:
        paths = [xml_path + '/' + file for file in get_all_files(xml_path)]
    for path in paths:
        try:
            tree = ET.parse(path)
            for basic_dep in tree.getroot().iter('basic-dependencies'):
                for dep in basic_dep.findall('dep'):
                    name = dep.get('type')
                    if name in verb_deps:
                        t = (name, dep.find('governor').text.lower(), dep.find('dependent').text.lower())
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
    try:
        tree = ET.parse(xml_filename)
        for basic_dep in tree.getroot().iter('basic-dependencies'):
            for dep in basic_dep.findall('dep'):
                try:
                    t = (dep.get('type'), dep.find('governor').text.lower(), dep.find('dependent').text.lower())
                    i = dependency_list.index(t)
                    array[i] += 1
                except:
                    pass
    except:
        pass
    return array

#from hw2: helper that gets the group of high vs low return files
def get_files_listed(corpusroot, filelist):
    lowd = dict()
    highd = dict()
    files = get_all_files(corpusroot)
    index = filelist.rfind('/')
    if index < 0:
        tokens = word_tokenize(PlaintextCorpusReader('.', filelist).raw())
    else:
        tokens = word_tokenize(PlaintextCorpusReader(filelist[:index], filelist[index+1:]).raw())
    i = 0
    while i < len(tokens):
        if float(tokens[i+1]) <= -5.0 and tokens[i] in files:
            lowd[tokens[i]] = float(tokens[i+1])
        if float(tokens[i+1]) >= 5.0 and tokens[i] in files:
            highd[tokens[i]] = float(tokens[i+1])
        i += 2

    return (lowd, highd)

def write_features(f, label, v):
    f.write(str(label))
    for i in range(len(v)):
        if v[i] != 0:
            f.write(' ' + str(i + 1) + ':' + str(v[i]))
    f.write('\n')

def process_corpus(txt_dir, xml_dir, feature_mode):
    if txt_dir.find('test') < 0:
        flag = 'train'
    else:
        flag = 'test'
    (lowd, highd) = get_files_listed(txt_dir, '/home1/c/cis530/hw3/xret_tails.txt')
    #here we want lexical features
    if feature_mode == 1:
        top_words = extract_top_words(txt_dir)
        f = open(flag + '_1_lexical.txt', 'w')
        for file in get_all_files(txt_dir):
            v = unigram_map_entry(txt_dir + '/' + file, top_words)
            if file in lowd:
                label = -1
            else:
                label = 1
            write_features(f, label, v)
            
    elif feature_mode == 2:
        mpqa_dict = get_mpqa_lexicon('/home1/c/cis530/hw3/mpqa-lexicon/subjclueslen1-HLTEMNLP05.tff')
        gi_dict = get_geninq_lexicon('/home1/c/cis530/hw3/gi-lexicon/inquirerTags.txt')
        f = open(flag + '_2_sentiment.txt', 'w')
        for file in get_all_files(txt_dir):
            v = get_mpqa_features(txt_dir + '/' + file, mpqa_dict)
            v.extend(get_geninq_features(txt_dir + '/' + file, gi_dict))
            if file in lowd:
                label = -1
            else:
                label = 1
            write_features(f, label, v)
        
    elif feature_mode == 3:
        f = open(flag + '_3_named_entity.txt', 'w')
        for file in get_all_files(xml_dir):
            v = extract_named_entities(xml_dir + '/' + file)
            if file[:file.find('.xml')] in lowd:
                label = -1
            else:
                label = 1
            write_features(f, label, v)

    elif feature_mode == 4:
        f = open(flag + '_4_postags.txt', 'w')
        adj_list = extract_adjectives(xml_dir)
        verb_list = extract_verbs(xml_dir)
        for file in get_all_files(xml_dir):
            v = map_adjectives(xml_dir + '/' + file, adj_list)
            v.extend(map_verbs(xml_dir + '/' + file, verb_list))
            if file[:file.find('.xml')] in lowd:
                label = -1
            else:
                label = 1
            write_features(f, label, v)

    elif feature_mode == 5:
        f = open(flag + '_5_dependency.txt', 'w')
        dep_list = extract_verb_dependencies(xml_dir)
        for file in get_all_files(xml_dir):
            v = map_verb_dependencies(xml_dir + '/' + file, dep_list)
            if file[:file.find('.xml')] in lowd:
                label = -1
            else:
                label = 1
            write_features(f, label, v)
            
    elif feature_mode == 6:
        f = open(flag + '_6_all.txt', 'w')
        top_words = extract_top_words(txt_dir)
        mpqa_dict = get_mpqa_lexicon('/home1/c/cis530/hw3/mpqa-lexicon/subjclueslen1-HLTEMNLP05.tff')
        gi_dict = get_geninq_lexicon('/home1/c/cis530/hw3/gi-lexicon/inquirerTags.txt')
        adj_list = extract_adjectives(xml_dir)
        verb_list = extract_verbs(xml_dir)
        dep_list = extract_verb_dependencies(xml_dir)
        for file in get_all_files(txt_dir):
            v = unigram_map_entry(txt_dir + '/' + file, top_words)
            v.extend(get_mpqa_features(txt_dir + '/' + file, mpqa_dict))
            v.extend(get_geninq_features(txt_dir + '/' + file, gi_dict))
            v.extend(extract_named_entities(xml_dir + '/' + file + '.xml'))
            v.extend(map_adjectives(xml_dir + '/' + file + '.xml', adj_list))
            v.extend(map_verbs(xml_dir + '/' + file + '.xml', verb_list))
            v.extend(map_verb_dependencies(xml_dir + '/' + file, dep_list))
            if file in lowd:
                label = -1
            else:
                label = 1
            write_features(f, label, v)

    elif feature_mode == 7:
        f = open(flag + '_7_own.txt', 'w')
        noun_list = extract_nouns(xml_dir)
        for file in get_all_files(xml_dir):
            v = map_nouns(xml_dir + '/' + file, noun_list)
            if file[:file.find('.xml')] in lowd:
                label = -1
            else:
                label = 1
            write_features(f, label, v)

def compute_performance(test_file, output_file):
    results_high = 0
    results_low = 0
    tests_high = 0
    tests_low = 0
    intersect_high = 0
    intersect_low = 0
    test_lines = open(test_file).readlines()
    output_lines = open(output_file).readlines()
    for i in range(len(test_lines)):
        test = int(word_tokenize(test_lines[i])[0])
        result = int(word_tokenize(output_lines[i])[0])
        if test == result:
            if test == 1:
                intersect_high += 1
            else:
                intersect_low += 1
        if test == -1:
            tests_low += 1
        elif test == 1:
            tests_high += 1
        if result == -1:
            results_low += 1
        elif result == 1:
            results_high += 1
    pos_pres = intersect_high/float(results_high) if results_high > 0 else 'infinity'
    pos_recall = intersect_high/float(tests_high) if tests_high > 0 else 'infinity'
    if results_high > 0 and tests_high > 0:
        pos_f_measure = 2 * pos_pres * pos_recall / (pos_pres + pos_recall)
    else:
        pos_f_measure = 'NA'
    neg_pres = intersect_low/float(results_low) if results_low > 0 else 'infinity'
    neg_recall = intersect_low/float(tests_low) if tests_low > 0 else 'infinity'
    if results_low > 0 and tests_low > 0:
        neg_f_measure = 2 * neg_pres * neg_recall / (neg_pres + neg_recall)
    else:
        neg_f_measure = 'NA'

    return (pos_pres, pos_recall, pos_f_measure, neg_pres, neg_recall, neg_f_measure)

'''
Part 6.3

Here are our results of precision, recall, and f measure with respect to
the different models:



'''

'''
Part 7

We added the functions extract_nouns and map_nouns which uses nouns as 
features. The intuition here is that certain nouns such as 'process,' 
'improvement,' etc. are positive and hence should lead to positive stock
reaction. Furthermore, many capital structure changes are correlated with
stock movements. Investors like 'share buyback,' 'dividends,' and 'spin-offs'
but generally dislike 'acquisition' or 'issuance' of debt or stock. Hence
by extracting nouns we should see these features correlate with stock movements.



'''
            
def main():
    #test functions for Part 1 & 2
    #top_words = extract_top_words('/home1/c/cis530/hw3/data')
    #print top_words
    #print unigram_map_entry('/home1/c/cis530/hw3/data/6285515.txt', top_words)
    #dic = get_mpqa_lexicon('/home1/c/cis530/hw3/mpqa-lexicon/subjclueslen1-HLTEMNLP05.tff')
    #print dic
    #print get_mpqa_features('/home1/c/cis530/hw3/data/6285515.txt', dic)
    #print get_mpqa_features_wordtype('/home1/c/cis530/hw3/data/6285515.txt', dic)
    #print get_mpqa_lexicon('/home1/c/cis530/hw3/mpqa-lexicon/subjclueslen1-HLTEMNLP05.tff')['mean']
    #gi_dict = get_geninq_lexicon('/home1/c/cis530/hw3/gi-lexicon/inquirerTags.txt')
    #print gi_dict
    #print gi_dict["make"]
    #print gi_dict["malady"]
    #print get_geninq_features('data/2067818.txt', gi_dict)

    
    #command line call to run CoreNLP
    #os.system('java -cp stanford-corenlp-2012-07-09.jar:stanford-corenlp-2012-07-06-models.jar:xom.jar:joda-time.jar -Xmx3g edu.stanford.nlp.pipeline.StanfordCoreNLP -annotators tokenize,ssplit,pos,lemma,ner,parse -filelist datafilelist.txt -outputDirectory data_result')
    #os.system('java -cp stanford-corenlp-2012-07-09.jar:stanford-corenlp-2012-07-06-models.jar:xom.jar:joda-time.jar -Xmx3g edu.stanford.nlp.pipeline.StanfordCoreNLP -annotators tokenize,ssplit,pos,lemma,ner,parse -filelist test_datafilelist.txt -outputDirectory test_data_result')

    #test functions for Part 3, 4, & 5
    #print extract_named_entities('data_result/71964.txt.xml')
    #print extract_adjectives('data_result')
    #print map_adjectives('data_result/71964.txt.xml', ['big', 'small', 'public'])
    #dictionary = extract_verb_dependencies('data_result')
    #print dictionary
    #print map_verb_dependencies('data_result/334701.txt.xml', dictionary)

    '''
    #generating training files for Part 6.2
    txt_dir = 'data'
    xml_dir = 'data_result'
    process_corpus(txt_dir, xml_dir, 1)
    process_corpus(txt_dir, xml_dir, 2)
    process_corpus(txt_dir, xml_dir, 3)
    process_corpus(txt_dir, xml_dir, 4)
    process_corpus(txt_dir, xml_dir, 5)
    process_corpus(txt_dir, xml_dir, 6)
    
    
    #call on svm to train files
    #we use -t 0 to change training into a linear model
    os.system('svm-train -t 0 train_1_lexical.txt 1_model.model')
    os.system('svm-train -t 0 train_2_sentiment.txt 2_model.model')
    os.system('svm-train -t 0 train_3_named_entity.txt 3_model.model')
    os.system('svm-train -t 0 train_4_postags.txt 4_model.model')
    os.system('svm-train -t 0 train_5_dependency.txt 5_model.model')
    os.system('svm-train -t 0 train_6_all.txt 6_model.model')
    
    
    #generating testing files for Part 6.3
    txt_dir = 'test_data'
    xml_dir = 'test_data_result'
    process_corpus(txt_dir, xml_dir, 1)
    process_corpus(txt_dir, xml_dir, 2)
    process_corpus(txt_dir, xml_dir, 3)
    process_corpus(txt_dir, xml_dir, 4)
    process_corpus(txt_dir, xml_dir, 5)
    process_corpus(txt_dir, xml_dir, 6)
    
    
    #call on svm to predict files
    os.system('svm-predict test_1_lexical.txt 1_model.model 1_result')
    os.system('svm-predict test_2_sentiment.txt 2_model.model 2_result')
    os.system('svm-predict test_3_named_entity.txt 3_model.model 3_result')
    os.system('svm-predict test_4_postags.txt 4_model.model 4_result')
    os.system('svm-predict test_5_dependency.txt 5_model.model 5_result')
    os.system('svm-predict test_6_all.txt 6_model.model 6_result')
    '''

    
    #computes precision, recall and f-score
    print compute_performance('test_1_lexical.txt', '1_result')
    print compute_performance('test_2_sentiment.txt', '2_result')
    print compute_performance('test_3_named_entity.txt', '3_result')
    print compute_performance('test_4_postags.txt', '4_result')
    print compute_performance('test_5_dependency.txt', '5_result')
    print compute_performance('test_6_all.txt', '6_result')
    
    
    #Part 7 calculations
    process_corpus('data', 'data_result', 7)
    os.system('svm-train -t 0 train_7_own.txt 7_model.model')
    process_corpus('test_data', 'test_data_result', 7)
    os.system('svm-predict test_7_own.txt 7_model.model 7_result')
    
    print compute_performance('test_7_own.txt', '7_result')
    
    
    #finds invalid xml files
    '''
    xml_dir = 'data_result'
    for file in get_all_files(xml_dir):
        try:
            tree = ET.parse(xml_dir + '/' + file)
        except Exception as e:
            print e
            print file
   '''

if __name__ == "__main__":
    main()
