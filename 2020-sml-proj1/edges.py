import pickle as pkl

forward = []
backward = []
count = 0
with open('./data/train.txt','r') as f:
    for line in f.readlines():
        count += 1
        if count % 1000 == 0:
            print(count,'...')
        cols = line.strip().split()
        source = cols[0]
        sinks = cols[1:]
        for sink in sinks:
            forward.append((source,sink))
            backward.append((sink,source))

print('size',len(forward))

edges = {
        'forward':forward,
        'backward':backward
        }

with open('./data/edges.pkl','wb') as f:
    pkl.dump(edges,f,protocol=pkl.HIGHEST_PROTOCOL)

