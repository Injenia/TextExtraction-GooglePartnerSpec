from gensim.models import Doc2Vec
import pandas as pd
import pickle
import json
import embedding as em
from parallelize import parallelize

parallel_embed_document = parallelize(em.embed_document_p)

def group_to_list_label(g):
    label = next(iter(g["label"]))
    sents = list(g["sentence"])
    return sents, label

def build_dataset2(model, df, permitted_words):
    l = [x for x in df.groupby("filename").apply(group_to_list_label)]
    docs = [e[0] for e in l]
    labels = [e[1] for e in l]
    print("Starting to embed")
    embedded_docs = parallel_embed_document(docs, model, permitted_words)
    return embedded_docs, labels

if __name__ == '__main__':
    csv_filename = '/notebooks/dev/infocamere/atti2.csv'
    model_filename = '../models/gensim_5000_model_with_verb.d2v'
    permitted_words_filename = '../first_5000_words_with_verb.json'
    dataset_filename = "embedded_docs_with_verb.p"
    
    df = pd.read_csv(csv_filename, encoding='utf-8')
    
    size_nc = len(df.loc[df['label'] == 'non_costitutivo'].groupby('filename'))

    grouped = df.loc[df['label'] == 'costitutivo'].groupby(df["filename"])
    dfs = [g[1] for g in list(grouped)[:size_nc]]

    grouped_nc = df.loc[df['label'] == 'non_costitutivo'].groupby(df["filename"])
    dfs_nc = [g[1] for g in list(grouped_nc)]

    df_balanced = pd.concat(dfs + dfs_nc)
    
    print("Balanced dataset")
    
    print("Freeing memory")
    del df
    del grouped
    del dfs
    del grouped_nc
    del dfs_nc
    
    print("Loading models")
    model =  Doc2Vec.load(model_filename)
    
    with open(permitted_words_filename) as o:
        permitted_words = set(json.load(o))
        
    print("Starting workers")
    docs, labels = build_dataset2(model, df_balanced, permitted_words)
    print("Embedding completed!")
    
    label_map = {'costitutivo':1, 'non_costitutivo':0}
    labels_n = [label_map[l] for l in labels]
    
    with open(dataset_filename, "w") as fout:
        pickle.dump([docs, labels_n], fout)    
    