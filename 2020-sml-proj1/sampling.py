import csv
import random
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

def save_as_csv(results,path):
    headers = ['id','Predicted']
    with open(path,'w') as f:
        f_csv = csv.writer(f)
        f_csv.writerow(headers)
        f_csv.writerows(results)

def save_as_txt(data,path):
    with open(path,'w') as f:
        for row in data:
            source,sink = row
            output = '\t'.join([str(source),str(sink)])
            f.write(output)
            f.write('\n')
    print(path,'saved')

def pos_sampling(source_nodes,graph,save_path):
    pos_samples = []
    for source in source_nodes:
        sinks = graph[source]
        random.shuffle(sinks)
        for sink in sinks[:10]:
            pos_samples.append((source,sink))
    print('positive samples:',len(pos_samples))
    save_as_txt(pos_samples,pos_sample_path)

def neg_sampling(nodes,source_nodes,graph,save_path):
    neg_samples = []
    for source in source_nodes:
        true_sinks = set(graph[source])
        if len(neg_samples) % 1000 == 0:
            print(len(neg_samples),'neg_samples produced...')
        sinks = set()
        enough = False
        while not enough:
            rd_nodes = random.sample(nodes,30)
            for node in rd_nodes:
                if node not in true_sinks:
                    sinks.add(node)
                if len(sinks) >= 10:
                    enough = True
                    break
        for sink in sinks:
            neg_samples.append((source,sink))
    print('negtive sample:',len(neg_samples))
    save_as_txt(neg_samples,neg_sample_path)

def sampling(all_data_path,pos_sample_path,neg_sample_path):
    # load data
    all_data = load_data_from_txt(all_data_path)
    print('all data:',len(all_data))
    print(all_data[0])
    
    # make nodes and graph
    nodes = []
    graph = {}
    for i in range(len(all_data)):
        nodes_list = [int(n) for n in all_data[i]]
        for node in nodes_list:
            nodes.append(node)
        graph[nodes_list[0]] = nodes_list[1:]
    nodes = set(nodes)
    print('nodes:',len(nodes))
    candidate_nodes = list(graph.keys())
    print('candidate nodes:',len(candidate_nodes))

    # choose candidates
    source_nodes = []
    for node in candidate_nodes:
        if len(graph[node]) >= 10:
            source_nodes.append(node)
    random.shuffle(source_nodes) # shuffle data
    #print(len(source_nodes))
    source_nodes = source_nodes[:18000]
    assert len(source_nodes) == 18000

    # make positive samples
    #pos_sampling(source_nodes,graph,pos_sample_path)

    # make negative samples
    neg_sampling(nodes,source_nodes,graph,neg_sample_path)


if __name__ == '__main__':
    all_data_path = './data/train.txt'
    pos_sample_path = './data/pos_samples.txt'
    neg_sample_path = './data/neg_samples.txt'
    sampling(all_data_path,pos_sample_path,neg_sample_path)

    

    



