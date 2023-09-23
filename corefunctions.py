import os
from os import listdir
from string import punctuation

import pandas as pd
from keras.layers import Dense
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from pattern.text.en import singularize


def change_file_name(file_name_list, dir):
    # function for changing names in the folder
    for old_name in file_name_list:
        new_name = old_name.replace("AND", "WITH")
        full_old_name = dir + old_name
        full_new_name = dir + new_name
        os.rename(full_old_name, full_new_name)


def load_doc(filename):
    # open the file as read only
    file = open(filename, 'r', encoding='unicode_escape')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text


def clean_doc(doc):
    # turn a doc into clean tokens
    # split into tokens by space
    doc = doc.replace('_', '')
    doc = doc.replace('/', '')
    doc = doc.replace('© 2000', ' ')
    doc = doc.replace('© 2004', ' ')
    doc = doc.replace('© 2005', ' ')
    doc = doc.replace('© 2006', ' ')
    doc = doc.replace('© 2007', ' ')
    doc = doc.replace('© 2008', ' ')
    doc = doc.replace('© 2009', ' ')
    doc = doc.replace('© 2010', ' ')
    doc = doc.replace('© 2011', ' ')
    doc = doc.replace('© 2012', ' ')
    doc = doc.replace('© 2013', ' ')
    doc = doc.replace('© 2014', ' ')
    doc = doc.replace('© 2015', ' ')
    doc = doc.replace('© 2016', ' ')
    doc = doc.replace('© 2017', ' ')
    doc = doc.replace('© 2018', ' ')
    doc = doc.replace('© 2019', ' ')
    doc = doc.replace('© 2020', ' ')
    doc = doc.replace('© 2021', ' ')
    doc = doc.replace('© 2022', ' ')
    doc = doc.replace('© 2023', ' ')
    doc = doc.replace('3R principle', 'reduce,reuse and recycle')
    doc = doc.replace('BMS', 'battery management system')
    doc = doc.replace('BOL', 'beginning of life')
    doc = doc.replace('BGP', 'biogas plants')
    doc = doc.replace('BAT', 'boundary aware Transformer')
    doc = doc.replace('3PL', 'third party reverse logistics')
    doc = doc.replace('3PRLP', 'third party reverse logistics provider')
    doc = doc.replace('BB', 'buyback')
    doc = doc.replace('BFGS', 'Broyden Fletcher Goldfarb Shanno')
    doc = doc.replace('BIM', 'Building Information Modelling')
    doc = doc.replace('CBM', 'circular business model')
    doc = doc.replace('CCSs', 'candidate concept schemes')
    doc = doc.replace('C–D centers', 'collection distribution centers')
    doc = doc.replace('CE', 'Circular Economy')
    doc = doc.replace('CH', 'Canova Hansen')
    doc = doc.replace('CLMEDIM', 'closed loop multi echelon distribution inventory supply chain model')
    doc = doc.replace('CLSC ND', 'closed loop supply chain network design')
    doc = doc.replace('CLSCND', 'closed loop supply chain network design')
    doc = doc.replace('CLSCN', 'closed loop supply chain network')
    doc = doc.replace('CLSC', 'Closed loop supply chain')
    doc = doc.replace('CMfg', 'Cloud manufacturing')
    doc = doc.replace('CMS', 'circular manufacturing systems')
    doc = doc.replace('CM', 'Circular Manufacturing')
    doc = doc.replace('CPP', 'Capacitated Production Planning')
    doc = doc.replace('CPQ', 'certainty of product quality')
    doc = doc.replace('CRMP', 'circular rubber manufacturing problem')
    doc = doc.replace('CRM', 'critical raw materials')
    doc = doc.replace('CRTX', 'Circular Textile Intelligence')
    doc = doc.replace('CSC', 'circular supply chain')
    doc = doc.replace('CUP', 'collected used products')
    doc = doc.replace('C VSM', 'Circular Value Stream Mapping')
    doc = doc.replace('DA', 'deconstructability assessment')
    doc = doc.replace('dBOM', 'disassembly Bill Of Material')
    doc = doc.replace('D CLSC', 'discounted closed loop supply chain')
    doc = doc.replace('DCP', 'Decision Choices Procedure')
    doc = doc.replace('DCs', 'distribution centres')
    doc = doc.replace('DDL', 'direct delivery')
    doc = doc.replace('DEMATEL', 'Decision Making Trial Evaluation and Laboratory')
    doc = doc.replace('DES', 'discrete event simulation')
    doc = doc.replace('DE', 'differential evolution')
    doc = doc.replace('DFCDSC', 'Design for circular digital Supply chain')
    doc = doc.replace('DfX', 'Design for Disassembly and Recovery')
    doc = doc.replace('DMs', 'decision makers')
    doc = doc.replace('DTD', 'door to door')
    doc = doc.replace('DPS', 'disproportionate profit sharing')
    doc = doc.replace('DP', 'Disassembly planning')
    doc = doc.replace('DSP', 'direct shipment')
    doc = doc.replace('DSS', 'Decision Support System')
    doc = doc.replace('DST', 'Decision Support Tool')
    doc = doc.replace('DWM', 'demolition waste management')
    doc = doc.replace('EEE', 'electrical and electronic equipment')
    doc = doc.replace('EI', 'environmental impacts')
    doc = doc.replace('ELVs', 'end of life vehicles')
    doc = doc.replace('EOL', 'end of life')
    doc = doc.replace('EOU', 'end of use')
    doc = doc.replace('EPR', 'Extended Producer Responsibility')
    doc = doc.replace('EU', 'European Union')
    doc = doc.replace('EVRS', 'EV renewal system')
    doc = doc.replace('EVSM', 'extended value stream mapping')
    doc = doc.replace('EVs', 'electric vehicles')
    doc = doc.replace('EWC', 'European Waste Catalogue')
    doc = doc.replace('FISOFin', 'Formal Industrial Symbiosis Opportunity Filtering method')
    doc = doc.replace('FLMEDIM', 'forward logistics multi echelon distribution inventory supply chain model')
    doc = doc.replace('FL', 'forward logistics')
    doc = doc.replace('FN', 'food neophobia')
    doc = doc.replace('FTN', 'food technology neophobia')
    doc = doc.replace('FW', 'food waste')
    doc = doc.replace('GCLSC', 'green closed loop supply chain')
    doc = doc.replace('GCL SCS', 'green closed loop supply chain system')
    doc = doc.replace('GCM', 'greedy clustering method')
    doc = doc.replace('GDM', 'Group Decision Making')
    doc = doc.replace('GL', 'Green logistics')
    doc = doc.replace('GSCM', 'supply chain management')
    doc = doc.replace('GW', 'glass waste')
    doc = doc.replace('HDDs', 'hard disk drives')
    doc = doc.replace('HDPE', 'high density polyethene')
    doc = doc.replace('HHS', 'household and similar waste')
    doc = doc.replace('HMRC', 'hazardous materials recovery centres')
    doc = doc.replace('HP', 'hazardous products')
    doc = doc.replace('HRS', 'Heat Recovery System')
    doc = doc.replace('HW', 'household waste')
    doc = doc.replace('ICS', 'initial conceptual scheme')
    doc = doc.replace('ICT', 'information and communications technology')
    doc = doc.replace('ID', 'influence diagram')
    doc = doc.replace('ILNO', 'integrated logistics network optimization')
    doc = doc.replace('IS', 'industrial symbiosis')
    doc = doc.replace('JIT', 'Just In Time')
    doc = doc.replace('KPIs', 'key performance indicators')
    doc = doc.replace('LAP', 'location allocation problem')
    doc = doc.replace('LCA', 'life cycle analysis')
    doc = doc.replace('LCI', 'Life Cycle Inventory')
    doc = doc.replace('LDPE', 'low density polyethene')
    doc = doc.replace('LIBS', 'laser induced breakdown spectroscopy')
    doc = doc.replace('LRPTW', 'location routing problem with time window')
    doc = doc.replace('LRP', 'location routing problem')
    doc = doc.replace('LSTM', 'long short term memory')
    doc = doc.replace('LS', 'local search')
    doc = doc.replace('MCDM', 'multi criteria decision making')
    doc = doc.replace('MCGDM', 'Multiple criteria group decision making')
    doc = doc.replace('MCME LRP RLN', 'multi cycle and multi echelon LRP in reverse logistics network')
    doc = doc.replace('MDVRPTW', 'multi depot vehicle routing problems with time windows')
    doc = doc.replace('ME', 'maximum entropy')
    doc = doc.replace('MFA', 'multi facility allocation')
    doc = doc.replace('MIIDAS', 'Multicriteria Interactive Intelligence Decision Aiding System')
    doc = doc.replace('MPAORLSE',
                      'Model of Performance Appraising & Optimizing of the Reverse Logistics System for Enterprises')
    doc = doc.replace('MRFs', 'Material Recovery Facilities')
    doc = doc.replace('MRO', 'maintenance,repair,and overhaul')
    doc = doc.replace('MSW', 'municipal solid waste')
    doc = doc.replace('NB', 'new batteries')
    doc = doc.replace('NDEA', 'network data envelopment analysis')
    doc = doc.replace('NDL', 'normal delivery')
    doc = doc.replace('NFSP', 'neural fictitious self play')
    doc = doc.replace('NIR', 'Near infrared')
    doc = doc.replace('NPS', 'number of Pareto solutions')
    doc = doc.replace('NPV', 'Net Present Value')
    doc = doc.replace('NS ', 'neighborhood search')
    doc = doc.replace('O2O', 'online to offline')
    doc = doc.replace('OCP', 'optimal control problem')
    doc = doc.replace('ODCPR', 'order driven component and product recovery')
    doc = doc.replace('OQN', 'open queueing network')
    doc = doc.replace('PaaS', 'Product as a Service')
    doc = doc.replace('PC', 'part consolidation')
    doc = doc.replace('PCW', 'paper and cardboard waste')
    doc = doc.replace('PEM', 'parallel enumeration method')
    doc = doc.replace('PFs', 'Production Facilities')
    doc = doc.replace('PLD', 'product line design')
    doc = doc.replace('POs', 'performance outcomes')
    doc = doc.replace('PPS', 'proportional profit sharing')
    doc = doc.replace('PRP', 'Pollution routing problems')
    doc = doc.replace('PSP', 'product sustainability performance')
    doc = doc.replace('PSS', 'Product Service System')
    doc = doc.replace('PUV', 'Product utilisation value')
    doc = doc.replace('PW', 'plastic waste')
    doc = doc.replace('REEs', 'rare earth elements')
    doc = doc.replace('RFID', 'Radio Frequency Identification')
    doc = doc.replace('RFKD', 'requirement function knowledge deployment')
    doc = doc.replace('RHM', 'residual hazardous materials')
    doc = doc.replace('RLN', 'reverse logistics network')
    doc = doc.replace('rLNP', 'reverse logistics network problem')
    doc = doc.replace('RLSC', 'reverse logistics supply chain')
    doc = doc.replace('RLS', 'reverse logistics system')
    doc = doc.replace('RL', 'reverse logistics')
    doc = doc.replace('RMRP', 'Reverse Material Requirement Planning')
    doc = doc.replace('RM', 'raw materials')
    doc = doc.replace('rMSW', 'residual municipal solid waste')
    doc = doc.replace('RO', 'robust optimization')
    doc = doc.replace('RSC', 'Reverse supply chain')
    doc = doc.replace('RSM', 'Response Surface Methodology')
    doc = doc.replace('SAAWM', 'sample average approximation based weighting method')
    doc = doc.replace('SB', 'scrap batteries')
    doc = doc.replace('SCLSC', 'sustainable closed loop supply chain')
    doc = doc.replace('SCM', 'supply chain management')
    doc = doc.replace('SCND', 'supply chain network design')
    doc = doc.replace('SCN', 'supply chain network')
    doc = doc.replace('SCSS', 'sustainable circular supplier selection')
    doc = doc.replace('SC', 'Supply Chain')
    doc = doc.replace('SDLB', 'single objective sequence dependent disassembly line balancing problem')
    doc = doc.replace('SDPC', 'simultaneous delivery and pick up problem with constrained capacity')
    doc = doc.replace('SEO', 'Social Engineering Optimizer')
    doc = doc.replace('SI', 'social impacts')
    doc = doc.replace('SPDTW', 'simultaneous pickup and delivery problems with time windows')
    doc = doc.replace('MDM', ' multiple decision makers')
    doc = doc.replace('SR', 'stabilized residual')
    doc = doc.replace('STPI', 'spot to point inflation')
    doc = doc.replace('TIDL', 'taxonomic development level index')
    doc = doc.replace('TSC', 'traditional supply chain')
    doc = doc.replace('TSDP', 'travelling salesman delivery and pick up problems')
    doc = doc.replace('SNE', 'distributed Stochastic Neighbour Embedding')
    doc = doc.replace('TSP', 'travelling salesman problem')
    doc = doc.replace('TS', 'tabu search')
    doc = doc.replace('TW', 'textile waste')
    doc = doc.replace('UPs', 'used products')
    doc = doc.replace('URD', 'uncertainty remanufacturing demand')
    doc = doc.replace('VE  ', 'Virtual Enterprise')
    doc = doc.replace('VMI', 'vendor managed inventory')
    doc = doc.replace('VNS', 'Variable Neighborhood Search')
    doc = doc.replace('VRPSDP', 'VehicleRoutingProblem with simultaneous deliveries and pickups')
    doc = doc.replace('VRP sPD', 'vehicleRoutingProblem with simultaneous pickup and delivery')
    doc = doc.replace('VRPSPD', 'vehicle routing problem with simultaneous pickups and deliveries')
    doc = doc.replace('VRP', 'VehicleRoutingProblem')
    doc = doc.replace('VU', 'value uncaptured')
    doc = doc.replace('WAP', 'weight assessment through prioritization method')
    doc = doc.replace('WD', 'waste diversion')
    doc = doc.replace('WEEE', 'waste of electrical and electronic equipment')
    doc = doc.replace('WMX', 'weight mapping crossover')
    doc = doc.replace('WTV', 'waste to value')
    doc = doc.replace('WWT', 'Wastewater treatment')
    doc = doc.replace('WW', 'wood waste')
    doc = doc.replace('genetic algorithm', 'GeneticAlgorithm')
    doc = doc.replace('neural network', 'NeuralNetwork')
    doc = doc.replace('Support Vector Machine', 'SupportVectorMachine')
    doc = doc.replace('Decision Support System', 'DecisionSupportSystem')
    doc = doc.replace('Constraint Logic Programming', 'ConstraintLogicProgramming')
    doc = doc.replace('Optimisation Algorithm', 'OptimisationAlgorithm')
    doc = doc.replace('Agent-based', 'AgentBased')
    doc = doc.replace('Ant Colony', 'AntColony')
    doc = doc.replace('Bee Colony', 'BeeColony')
    doc = doc.replace('Swarm Optimization', 'SwarmOptimization')
    doc = doc.replace('Swarm Intelligence', 'SwarmIntelligence')
    doc = doc.replace('computer vision', 'ComputerVision')
    doc = doc.replace('combinatorial optimisation', 'CombinatorialOptimisation')
    doc = doc.replace('deep learning', 'DeepLearning')
    doc = doc.replace('firefly algorithm', 'FireflyAlgorithm')
    doc = doc.replace('multi-objective', 'MultiObjective')
    doc = doc.replace('fuzzy ', 'fuzzy')
    doc = doc.replace('Fuzzy ', 'fuzzy')
    doc = doc.replace('machine learning', 'MachineLearning')
    doc = doc.replace('linear programming', 'LinearProgramming')
    doc = doc.replace('integer programming', 'IntegerProgramming')
    doc = doc.replace('Natural Language Processing', 'NaturalLanguageProcessing')
    doc = doc.replace('Frequency Identification', 'FrequencyIdentification')
    doc = doc.replace('simulated annealing', 'SimulatedAnnealing')
    # split into tokens by white space
    tokens = doc.split()
    # remove punctuation from each token
    table = str.maketrans('', '', punctuation)
    tokens = [w.translate(table) for w in tokens]
    # remove remaining tokens that are not alphabetic
    tokens = [word for word in tokens if word.isalpha()]
    # convert to lower case
    tokens = [word.lower() for word in tokens]
    # filter out stop words
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if not w in stop_words]
    # simple sense for verb
    tokens = [WordNetLemmatizer().lemmatize(word, 'v') for word in tokens]
    # convert plural to singular
    tokens = [singularize(word) for word in tokens]
    # remove meaningless tokens
    tokens = [w for w in tokens if
              not w in {'the', 'and', 'paper', 'must','could', 'literature', 'research', 'study', 'find', 'use', 'elsevier'}]
    # filter out short tokens
    tokens = [word for word in tokens if len(word) > 1]
    return tokens


def add_doc_to_vocab(filename, vocab):
    # load .txt file and add to vocab
    # load doc
    doc = load_doc(filename)
    # clean doc
    tokens = clean_doc(doc)
    # update counts
    vocab.update(tokens)


def process_docs(directory, vocab):
    # load all docs in a directory
    # walk through all files in the folder
    for filename in listdir(directory):
        # skip any reviews in the test set
        if filename.startswith("ceWITH"):
            continue
        # create the full path of the file to open
        path = directory + '/' + filename
        # add doc to vocab
        add_doc_to_vocab(path, vocab)


def save_list(lines, filename):
    # save list to file
    # convert lines to a single blob of test
    data = '\n'.join(lines)
    # open file
    file = open(filename, 'w', encoding='unicode_escape')
    # write text
    file.write(data)
    # close file
    file.close()


def create_vocabs(cluster_folder_dir, vocab, cluster_name):
    # create vocabs for each cluster
    # add all docs to vocab
    process_docs(cluster_folder_dir, vocab)
    # print the size of the vocab
    print(len(vocab))
    # print the top words in the vocab
    print(vocab.most_common(50))
    # keep tokens with a min occurrence
    min_occurrence = 1
    tokens = [k for k, c in vocab.items() if c >= min_occurrence]
    tokens_fre = [[k, c] for k, c in vocab.items() if c >= min_occurrence]
    print(len(tokens_fre))
    # Color = ['Default' for i in range(1279)]
    cluster_name_df = pd.DataFrame(tokens_fre, columns=['token', 'size'])
    cluster_name_df.to_csv(r"F:/test-file/cluster_vocab/" + cluster_name + '.csv')
    # print(len(tokens))
    # save tokens to a vocabulary file
    save_list(tokens, r"F:/test-file/cluster_vocab/" + cluster_name + '.txt')


def doc_to_line(filename, vocab):
    # load abstracts, clean and return line of tokens
    # load the doc
    doc = load_doc(filename)
    # clean doc
    tokens = clean_doc(doc)
    # filter by vocab
    tokens = [w for w in tokens if w in vocab]
    return " ".join(tokens)


# load all docs in a directory
def process_train_docs(directory, vocab, is_train):
    lines = list()
    for filename in listdir(directory):
        # walk through all the files in the folder
        if is_train and filename.startswith('ceWITH'):
            continue
        if not is_train and not filename.startswith('ceWITH'):
            continue
        # create the full path of the file to open
        path = directory + '/' + filename
        # load and clean the doc
        line = doc_to_line(path, vocab)
        # add to list
        lines.append(line)
    return lines


def prepare_data(train_abs, test_abs, mode):
    # prepare bag of words encoding of abstracts
    # create the tokenizer
    tokenizer = Tokenizer()
    # fit the tokenizer on the documents
    tokenizer.fit_on_texts(train_abs)
    # encoding training data set
    x_train = tokenizer.texts_to_matrix(train_abs, mode=mode)
    # encoding the test data set
    x_test = tokenizer.texts_to_matrix(test_abs, mode=mode)
    return x_train, x_test


def evaluate_mode(x_train, y_train, x_test, y_test):
    # evaluate the performance of the neural network model
    scores = list()
    n_repeats = 30 # the number of repeats for checking robust
    n_words = x_test.shape[1]
    for i in range(n_repeats):
        # define network base on the project
        model = Sequential()
        model.add(Dense(50, input_shape=(n_words,), activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(x_train, y_train, epochs=50, verbose=2)
        loss, acc = model.evaluate(x_test, y_test, verbose=0)
        scores.append(acc)
        print('%d accuracy is %s' % ((i+1), acc))
    return scores


def predict_sentiment(abstract, vocab, tokenizer, model):
    tokens = clean_doc(abstract)
    tokens = [w for w in tokens if w in vocab]
    line = ' '.join(tokens)
    encoded = tokenizer.texts_to_matrix([line], mode='freq')
    yhat = model.predict(encoded, verbose=0)
    return round(yhat[0,0])
