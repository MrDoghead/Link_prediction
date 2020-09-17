import pickle
import json
import math
import time
from tqdm import tqdm

def load_data_from_json(path):
    with open(path,'r') as f:
        data = json.load(f)
    return data

def load_train_from_txt(path,label):
    data = []
    with open(path,'r') as f:
        for line in f.readlines():
            cols = line.strip().split()
            cols.append(label)
            data.append(cols)
    return data

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
    #print('aa:',score)
    return score

def get_cosine_sim(node1,node2):
    vec1 = [basic_info[node1][i] for i in [1,3,5,6]]
    vec2 = [basic_info[node2][i] for i in [1,3,5,6]]
    doc_product = 0.0
    for i in range(len(vec1)):
        doc_product += (vec1[i] * vec2[i])
    sim = doc_product / (len(vec1) * len(vec2))
    #print('cos:',sim)
    return sim

def get_cosine_score(node1,node2):
    node1_neighbors = set(basic_info[node1][4])
    node2_neighbors = set(basic_info[node2][4])
    common_neighbors = node1_neighbors & node2_neighbors
    if not common_neighbors:
        return 0.0
    else:
        return (0.0 + len(common_neighbors)) / (basic_info[node1][5] * basic_info[node2][5])

def get_jaccard_coefficient(node1,node2):
    node1_neighbors = set(basic_info[node1][4])
    node2_neighbors = set(basic_info[node2][4])
    insertion = node1_neighbors & node2_neighbors
    union = node1_neighbors | node2_neighbors
    if not union:
        return 0.0
    coe = len(insertion) / len(union)
    #print('jaccard:',coe)
    return coe

def get_salton_sim(node1,node2):
    node1_neighbors = set(basic_info[node1][4])
    node2_neighbors = set(basic_info[node2][4])
    common_neighbors = node1_neighbors & node2_neighbors
    node1_out_degree = basic_info[node1][3]
    node2_in_degree = basic_info[node2][1]
    if not common_neighbors:
        return 0.0
    try:
        sim = len(common_neighbors) / math.sqrt(node1_out_degree * node2_in_degree)
    except:
        return 0.0
    #print('salton:',sim)
    return sim

def get_preferential_attachment(node1,node2):
    node1_degree = basic_info[node1][5]
    node2_degree = basic_info[node2][5]
    score = 1.0 * node1_degree * node2_degree
    #print('pa:',score)
    return score

def get_fridencs_measure(node1,node2):
    node1_neighbors = set(basic_info[node1][4])
    node2_neighbors = set(basic_info[node2][4])
    score = 0.0
    for n1 in node1_neighbors:
        for n2 in node2_neighbors:
            if (n1,n2) in all_edges:
                score += 1
    #print('fm:',score)
    return score

def get_resource_allocation(node1,node2):
    node1_neighbors = set(basic_info[node1][4])
    node2_neighbors = set(basic_info[node2][4])
    common_neighbors = node1_neighbors & node2_neighbors
    score = 0.0
    for cm_node in common_neighbors:
        score += 1.0 / basic_info[cm_node][5]
    #print('ra:',score)
    return score

# how similar are the outbound neighbors of source to sink
def get_outbound_similarity_score(source, sink, metric):
    # get the outbound_node of source
    outbound_node_for_source_set = set(basic_info[source][2])
    summation = 0
    for outbound_node_for_source in outbound_node_for_source_set:
        summation =summation + metric(sink,outbound_node_for_source)
    if len(outbound_node_for_source_set) == 0:
        return 0
    score = 1/len(outbound_node_for_source_set)*summation
    return score

def get_inbound_similarity_score(source, sink, metric):
    # get the inbound_node of sink
    inbound_node_for_sink_set = set(basic_info[sink][0])
    summation = 0
    for inbound_node_for_sink in inbound_node_for_sink_set:
        summation =summation + metric(source,inbound_node_for_sink)
    if len(inbound_node_for_sink_set) == 0:
        return 0
    score = 1/len(inbound_node_for_sink_set)*summation
    return score

def save_as_pkl(data,path):
    with open(path,'wb') as f:
        pickle.dump(data,f,protocol=pickle.HIGHEST_PROTOCOL)
    print(path,'saved')

if __name__ == '__main__':
    pos_train = load_train_from_txt('./data/pos_samples.txt',1)
    neg_train = load_train_from_txt('./data/neg_samples.txt',0)
    basic_info = load_data_from_json('./data/basic_info.json')
    edges = read_from_pkl('./data/edges.pkl')
    direct_edges = edges['forward']
    direct_edges = set(direct_edges)
    all_edges = edges['forward'] + edges['backward']
    all_edges = set(all_edges)
    print('all data loaded')
    assert len(pos_train) == len(neg_train)

    # make train features
    print('start producing train features...')
    all_train = pos_train + neg_train
    train_features = {}
    ind = 0
    for sample in tqdm(all_train):
        train_id = 'train-' + str(ind)
        source,sink,label = sample

        cosine_score = get_cosine_score(source,sink)

        adamic_adar_index_in = get_inbound_similarity_score(source,sink,get_adamic_adar_index)
        cosine_score_in = get_inbound_similarity_score(source,sink,get_cosine_score)
        jaccard_coefficient_in = get_inbound_similarity_score(source,sink,get_jaccard_coefficient)
        salton_sim_in = get_inbound_similarity_score(source,sink,get_salton_sim)
        preferential_attachment_in = get_inbound_similarity_score(source,sink,get_preferential_attachment)
        #friends_measure_in = get_inbound_similarity_score(source,sink,get_fridencs_measure)
        resource_allocation_in = get_inbound_similarity_score(source,sink,get_resource_allocation)

        adamic_adar_index_out = get_outbound_similarity_score(source,sink,get_adamic_adar_index)
        cosine_score_out = get_outbound_similarity_score(source,sink,get_cosine_score)
        jaccard_coefficient_out = get_outbound_similarity_score(source,sink,get_jaccard_coefficient)
        salton_sim_out = get_outbound_similarity_score(source,sink,get_salton_sim)
        preferential_attachment_out = get_outbound_similarity_score(source,sink,get_preferential_attachment)
        #friends_measure_out = get_outbound_similarity_score(source,sink,get_fridencs_measure)
        resource_allocation_out = get_outbound_similarity_score(source,sink,get_resource_allocation)

        train_features[train_id] = [
                                    source,
                                    sink,
                                    cosine_score,

                                    adamic_adar_index_in,
                                    cosine_score_in,
                                    jaccard_coefficient_in,
                                    salton_sim_in,
                                    preferential_attachment_in,
                                    #friends_measure_in,
                                    resource_allocation_in,

                                    adamic_adar_index_out,
                                    cosine_score_out,
                                    jaccard_coefficient_out,
                                    salton_sim_out,
                                    preferential_attachment_out,
                                    #friends_measure_in,
                                    resource_allocation_out
                                    ]
        ind += 1

    # save train features
    save_train_path = './data/train_13_features.pkl'
    save_as_pkl(train_features,save_train_path)






