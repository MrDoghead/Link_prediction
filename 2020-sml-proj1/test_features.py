import pickle
import json
import math
from tqdm import tqdm

def load_data_from_json(path):
    with open(path,'r') as f:
        data = json.load(f)
    return data

def load_test_from_txt(path):
    data = []
    with open(path,'r') as f:
        for line in f.readlines():
            cols = line.strip().split()
            data.append(cols)
    return data[1:]

def read_from_pkl(path):
    with open(path,'rb') as f:
        data = pickle.load(f)
    return data

def get_adamic_adar_index(node1,node2):
    node1_neighbors = set(basic_info[node1][4])
    node2_neighbors = set(basic_info[node2][4])
    common_neighbors = node1_neighbors & node2_neighbors
    score = 0.0
    for cm_node in common_neighbors:
        score += (1. / math.log(basic_info[cm_node][5]+2))
    return score

def get_cosine_sim(node1,node2):
    vec1 = [basic_info[node1][i] for i in [1,3,5,6]]
    vec2 = [basic_info[node2][i] for i in [1,3,5,6]]
    doc_product = 0.0
    for i in range(len(vec1)):
        doc_product += (vec1[i] * vec2[i])
    sim = doc_product / (len(vec1) * len(vec2))
    return sim

def get_jaccard_coefficient(node1,node2):
    node1_neighbors = set(basic_info[node1][4])
    node2_neighbors = set(basic_info[node2][4])
    insertion = node1_neighbors & node2_neighbors
    union = node1_neighbors | node2_neighbors
    if not union:
        return 0.0
    coe = len(insertion) / len(union)
    return coe

def get_salton_sim(node1,node2):
    node1_neighbors = set(basic_info[node1][4])
    node2_neighbors = set(basic_info[node2][4])
    common_neighbors = node1_neighbors & node2_neighbors
    node1_out_degree = basic_info[node1][3]
    node2_in_degree = basic_info[node2][1]
    if not common_neighbors:
        return 0.0
    sim = len(common_neighbors) / math.sqrt(node1_out_degree * node2_in_degree)
    return sim

def get_preferential_attachment(node1,node2):
    node1_degree = basic_info[node1][5]
    node2_degree = basic_info[node2][5]
    score = 1.0 * node1_degree * node2_degree
    return score

def get_fridencs_measure(node1,node2):
    node1_neighbors = set(basic_info[node1][4])
    node2_neighbors = set(basic_info[node2][4])
    score = 0.0
    for n1 in node1_neighbors:
        n1_neighbors = basic_info[n1][4]
        for n2 in node2_neighbors:
            if n2 in n1_neighbors:
                score += 1
    return score

def save_as_pkl(data,path):
    with open(path,'wb') as f:
        pickle.dump(data,f,protocol=pickle.HIGHEST_PROTOCOL)
    print(path,'saved')

def get_resource_allocation(node1,node2):
    node1_neighbors = set(basic_info[node1][4])
    node2_neighbors = set(basic_info[node2][4])
    common_neighbors = node1_neighbors & node2_neighbors
    score = 0.0
    for cm_node in common_neighbors:
        score += 1.0 / basic_info[cm_node][5]
    return score

if __name__ == '__main__':
    test_data = load_test_from_txt('./data/test-public.txt')
    basic_info = load_data_from_json('./data/basic_info.json')
    edges = read_from_pkl('./data/edges.pkl')
    direct_edges = edges['forward']
    direct_edges = set(direct_edges)
    all_edges = edges['forward'] + edges['backward']
    all_edges = set(all_edges)
    print('all data loaded')
    print('test size:',len(test_data))

    # make test features
    print('start producing test features...')
    test_features = {}
    for sample in tqdm(test_data):
        test_id,source,sink = sample
        num_of_neighbors_source = basic_info[source][5]
        num_of_in_neighbors_source = basic_info[source][1]
        num_of_out_neighbors_source = basic_info[source][3]
        num_of_neighbors_sink = basic_info[sink][5]
        num_of_in_neighbors_sink = basic_info[sink][1]
        num_of_out_neighbors_sink = basic_info[sink][3]
        num_of_neighbors_sum = num_of_neighbors_source + num_of_neighbors_sink
        num_of_in_neighbors_sum = num_of_in_neighbors_source + num_of_in_neighbors_sink
        num_of_out_neighbors_sum = num_of_out_neighbors_source + num_of_out_neighbors_sink
        adamic_adar_index = get_adamic_adar_index(source,sink)
        cosine_sim = get_cosine_sim(source,sink)
        jaccard_coefficient = get_jaccard_coefficient(source,sink)
        salton_sim = get_salton_sim(source,sink)
        preferential_attachment = get_preferential_attachment(source,sink)
        friends_measure = get_fridencs_measure(source,sink)
        resource_allocation = get_resource_allocation(source,sink)

        
        test_features[test_id] = ['',
                                    source,
                                    sink,
                                    num_of_neighbors_source,
                                    num_of_in_neighbors_source,
                                    num_of_out_neighbors_source,
                                    num_of_neighbors_sink,
                                    num_of_in_neighbors_sink,
                                    num_of_out_neighbors_sink,
                                    num_of_neighbors_sum,
                                    num_of_in_neighbors_sum,
                                    num_of_out_neighbors_sum,
                                    adamic_adar_index,
                                    cosine_sim,
                                    jaccard_coefficient,
                                    salton_sim,
                                    preferential_attachment,
                                    friends_measure,
                                    resource_allocation,
                                    #adamic_adar_index_out,
                                    #cosine_sim_out,
                                    #jaccard_coefficient_out,
                                    #salton_sim_out,
                                    #preferential_attachment_out,
                                    #resource_allocation_out
                                    ]

    # save test features
    save_test_path = './data/test_19_features.pkl'
    save_as_pkl(test_features,save_test_path)






