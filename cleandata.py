import csv
import pandas as pd
import os


def create_dir(dir):
    # creating empty folders
    if not os.path.exists(dir):
        os.makedirs(dir)
        print("Create directory:", dir)
    else:
        print("Directory already existed:", dir)
    return dir


def csv_txt(filename, cluster_list, folder_list):
    df = pd.read_csv(filename)
    df = df.fillna('')
    print(df.shape)

    for cluster in cluster_list:
        posit = cluster_list.index(cluster)
        classified_df = df[df.cluster_name == cluster]
        classified_df = classified_df.reset_index()
        classified_df = classified_df[['tittle', 'keywords', 'abstract']]
        classified_df['text'] = classified_df['tittle'] + '. ' + classified_df['abstract'] + '. ' + classified_df[
            'keywords']
        textNames = [i for i in range(classified_df.shape[0])]
        print(classified_df.shape)

        for textName in textNames:
            f = open(r"F:/test-file/" + folder_list[posit] + "/" + "ceANDai" + str(textName) + ".txt",
                     'w', encoding="utf-8")
            f.write(classified_df['text'][textName])
            f.close()


folder_list = ['1_recoveryRecycling', '2_cePrep', '3_facPlanning', '4_cescDesign', '5_ceCooperation',
               '6_mfgOptimisation', '7_cescOptimisation', '8_productBmInnovation']
for name in folder_list:
    path = r'F:/test-file/' + name
    create_dir(path)
filename = r'F:/test-file/datasample.csv'
cluster_list = ['recoveryRecycling', 'cePrep', 'facPlanning', 'cescDesign', 'ceCooperation', 'mfgOptimisation',
                'cescOptimisation', 'productBmInnovation']
csv_txt(filename, cluster_list, folder_list)
