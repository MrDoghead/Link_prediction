import math
import json
import sys

def load_data(path):
    data = []
    with open(path,'r') as f:
        for line in f.readlines():
            nodes = line.strip().split()
            data.append(nodes)
    return data

def save_data(data,path):
    with open(path,'w') as f:
        json.dump(data,f)
    print('saved at',path)

if __name__ == '__main__':
    train_path = './data/train.txt'
    train_data = load_data(train_path)

    all_nodes = []
    outbound = {}   # outbound neighbors
    inbound = {}    # inbound neighbors
    for node_list in train_data:
        all_nodes.extend(node_list)
        source = node_list[0]
        sink = node_list[1:]
        outbound[source] = set(sink)
        for n in sink:
            if n not in inbound:
                inbound[n] = set()
            inbound[n].add(source)
    all_nodes = set(all_nodes)

    basic_info = {}
    for i,node in enumerate(all_nodes):
        if i % 1000 == 0:
            print(i,'nodes processed...')
        in_neighbors = inbound.get(node,set())
        num_in_neighbors = len(in_neighbors)
        out_neighbors = outbound.get(node,set())
        num_out_neighbors = len(out_neighbors)
        all_neighbors = in_neighbors | out_neighbors
        num_all_neighbors = len(all_neighbors)
        assert num_all_neighbors == len(set(all_neighbors))
        log_all_neighbors = math.log(num_all_neighbors)

        basic_info[node] = [list(in_neighbors),
                            num_in_neighbors,
                            list(out_neighbors),
                            num_out_neighbors,
                            list(all_neighbors),
                            num_all_neighbors,
                            log_all_neighbors]

    # save info
    data_path = './data/basic_info.json'
    save_data(basic_info,data_path)
