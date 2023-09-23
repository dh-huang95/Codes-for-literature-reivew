import os, shutil, random

import pd as pd
from numpy import array
from string import punctuation
from os import listdir
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from pattern.text.en import singularize

from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
import pandas as pd
from pandas import DataFrame
from matplotlib import pyplot

from cleandata import cluster_list
from corefunctions import load_doc, clean_doc, add_doc_to_vocab, doc_to_line, process_docs, prepare_data, save_list, \
    change_file_name, create_vocabs
from corefunctions import process_train_docs
from corefunctions import evaluate_mode

source_list = ['1_recoveryRecycling', '2_cePrep', '3_facPlanning', '4_cescDesign', '5_ceCooperation',
               '6_mfgOptimisation', '7_cescOptimisation', '8_productBmInnovation']

for j in range(8):
    tested_cluster = source_list[j]

    os.makedirs(r'F:/test-file/results/', exist_ok=True)

    edf = pd.DataFrame(index=range(1), columns=range(4))
    edf.columns = ['binary', 'count', 'tfidf', 'freq']
    edf.to_csv(r'F:/test-file/results/' + tested_cluster + "out.csv", index=False)

    for i in range(10): # random sampling abstracts to check robust

        results = DataFrame()
        # name all folders

        # path source directory
        src_dir = r'F:/test-file/'+tested_cluster
        num_dir_t = len(os.listdir(src_dir))

        # path to destination directory
        destn_dir = r'F:/test-file/temp'

        # check whether there exists such a file
        if os.path.isdir(destn_dir):
            shutil.rmtree(destn_dir)
            os.mkdir(destn_dir)
        else:
            os.mkdir(destn_dir)

        # get all the files in the source directory
        files = os.listdir(src_dir)
        shutil.copytree(src_dir, destn_dir, dirs_exist_ok=True)

        # path to destination directory
        file_dir_nt = r'F:/test-file/temp_0'

        # check whether there exists such a file
        if os.path.isdir(file_dir_nt):
            shutil.rmtree(file_dir_nt)
            os.mkdir(file_dir_nt)
        else:
            os.mkdir(file_dir_nt)

        counter = 0
        for cluster_name in source_list:
            if cluster_name == tested_cluster:
                continue
            else:
                name_dir_nt = r'F:/test-file/'+cluster_name
                for filename in os.listdir(name_dir_nt):
                    name_source_nt = name_dir_nt+'/'+filename
                    print(name_source_nt)
                    counter += 1
                    name_des_nt = r'F:/test-file/temp_0/'+'ceANDai'+str(counter)
                    shutil.copyfile(name_source_nt, name_des_nt)

        random_file = random.sample(os.listdir(r'F:/test-file/temp_0/'), num_dir_t)

        # check whether there exists such a file to contain the random un targeted sample
        random_file_dir_nt = r'F:/test-file/temp_1'
        if os.path.isdir(random_file_dir_nt):
            shutil.rmtree(random_file_dir_nt)
            os.mkdir(random_file_dir_nt)
        else:
            os.mkdir(random_file_dir_nt)

        for random_nt in random_file:
            random_dir_nt = r'F:/test-file/temp_0/'+ random_nt
            random_dir_nt_t1 = r'F:/test-file/temp_1/'+random_nt
            shutil.copyfile(random_dir_nt, random_dir_nt_t1)

        # sort out the last 15% tested samples to change the name
        # get the list
        name_file_t = sorted(os.listdir(r'F:/test-file/temp/'), key=len)
        name_file_nt = sorted(os.listdir(r'F:/test-file/temp_1/'), key=len)
        test_set_number = int(len(name_file_t)*0.15)
        training_set_number = len(name_file_t) - test_set_number

        name_file_test_t = name_file_t[-test_set_number:]
        name_file_test_nt = name_file_nt[-test_set_number:]

        # change the last 15% names in the list
        change_file_name(name_file_test_t, r'F:/test-file/temp/')
        change_file_name(name_file_test_nt, r'F:/test-file/temp_1/')

        # define vocab
        vocab = Counter()
        # add all docs to vocab
        process_docs(r'F:/test-file/temp', vocab)
        process_docs(r'F:/test-file/temp_1', vocab)
        # print the size of the vocab
        print(len(vocab))
        # print the top words in the vocab
        print(vocab.most_common(50))

        # keep tokens with a minium occurrence
        min_occurrence = 1
        tokens = [k for k, c in vocab.items() if c >= min_occurrence]
        tokens_fre = [[k, c] for k, c in vocab.items() if c >= min_occurrence]
        print(len(tokens))
        print(len(tokens_fre))

        # save tokens to a vocabulary file
        cluster_name_df = pd.DataFrame(tokens_fre, columns=['token', 'size'])
        cluster_name_df.to_csv(r"F:/test-file/cluster_vocab/" + tested_cluster + '.csv')
        save_list(tokens, r'F:/test-file/vocab_temp.txt')

        # load the vocabulary
        vocab_filename = r'F:/test-file/vocab_temp.txt'
        vocab = load_doc(vocab_filename)
        vocab = vocab.split()
        vocab = set(vocab)

        positive_lines = process_train_docs(r'F:/test-file/temp', vocab, True)
        negative_lines = process_train_docs(r'F:/test-file/temp_1', vocab, True)
        train_docs = positive_lines + negative_lines

        positive_lines_test = process_train_docs(r'F:/test-file/temp', vocab, False)
        negative_lines_test = process_train_docs(r'F:/test-file/temp_1', vocab, False)
        test_docs = positive_lines_test + negative_lines_test

        y_train = array([1 for _ in range(training_set_number)] + [0 for _ in range(training_set_number)])
        y_test = array([1 for _ in range(test_set_number)] + [0 for _ in range(test_set_number)])

        modes = ['binary', 'count', 'tfidf', 'freq']
        for mode in modes:
            x_train, x_test = prepare_data(train_docs, test_docs, mode)
            results[mode] = evaluate_mode(x_train, y_train, x_test, y_test)
        results.to_csv(r'F:/test-file/results/' + tested_cluster + 'out0.csv', index=False)

        df = pd.read_csv(r'F:/test-file/results/' + tested_cluster + 'out0.csv')
        # df.drop(columns=df.columns[0], axis=1, inplace=True)

        df1 = pd.read_csv(r'F:/test-file/results/' + tested_cluster + 'out.csv')
        # df1.drop(columns=df.columns[0], axis=1, inplace=True)

        summary = pd.concat([df, df1], ignore_index=True)
        summary.to_csv(r'F:/test-file/results/' + tested_cluster + "out.csv", index=False)
        os.remove(r'F:/test-file/results/' + tested_cluster + "out0.csv")





