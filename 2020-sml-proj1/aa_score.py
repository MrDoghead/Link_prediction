import csv
import math
import pickle as pk
import sys

def load_data_from_txt(path):
    rows = []
    with open(path,'r') as f:
        for line in f.readlines():
            data = line.strip().split()
            rows.append(data)
    return rows

def load_data_from_csv(path):
    with open(path,'r') as f:
        reader = csv.reader(f)
        rows = [row for row in reader]
    return rows[1:]

def common_neighbors(Graph, Node1, Node2):
    if (Node1 in list(Graph.keys())) and (Node2 in list(Graph.keys())):
        N1_neis = Graph[Node1]
        N2_neis = Graph[Node2]
        common = list(set(N1_neis).intersection(N2_neis))
        return common
    else:
        return []

def AA(Graph, Node1, Node2):
    sim = 0.0
    common_neighors = common_neighbors(Graph, Node1, Node2)
    for node in common_neighors:
        if node in list(Graph.keys()):
            #sim = (1. / math.log(len(Graph[node])))
            sim += (1. / math.log(len(Graph[node])+2))
    return sim

def predict(test_set,Graph,method):
    res  = []
    for data in test_set:
        ind,source,sink = data
        sim = AA(Graph,int(source),int(sink))
        res.append([ind,sim])
    return res

def save_as_csv(results,path):
    headers = ['id','Predicted']
    with open(path,'w') as f:
        f_csv = csv.writer(f)
        f_csv.writerow(headers)
        f_csv.writerows(results)

def save_as_txt(results,path):
    with open(path,'w') as f:
        for row in results:
            f.write(row)

if __name__ == '__main__':
    # load data
    train_data_path = './data/train.txt'
    train_data = load_data_from_txt(train_data_path)
    test_data_path = './data/test-public.txt'
    test_data = load_data_from_txt(test_data_path)
    test_data = test_data[1:]
    print('train:',len(train_data))
    print(train_data[0])
    print('test:',len(test_data))
    print(test_data[0])
    
    # make nodes and graph
    nodes = []
    graph = {}
    for i in range(len(train_data)):
        nodes_list = [int(n) for n in train_data[i]]
        for node in nodes_list:
            nodes.append(node)
        graph[nodes_list[0]] = nodes_list[1:]
    nodes = set(nodes)
    print('nodes:',len(nodes))

    # prediction
    predictions = predict(test_data,graph,AA)
    print('prediction',len(predictions))
    print(predictions[:10])
    save_path = './AA_score.csv'
    save_as_csv(predictions,save_path)
    print('saved to',save_path)

    



