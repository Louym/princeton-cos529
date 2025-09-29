import pickle
import numpy as np
import argparse
import networkx as nx


def load_graph(filename):
    'load the graphical model (DO NOT MODIFY)'
    return pickle.load(open(filename, 'rb'))

def decode(num, K, N, i=-1):
    assert num < K**N
    if i==-1:#decode all
        assignment = []
        for _ in range(N):
            assignment.append(num % K)
            num = num // K
        return assignment
    else:
        return (num // K**i) % K
            
def check_equal(G1, G2):
   # Check if two soluations get the same results
    assert G1.graph['K'] == G2.graph['K']
    for v in G1.nodes:
        assert np.allclose(G1.nodes[v]['marginal_prob'], G2.nodes[v]['marginal_prob'])
        assert np.allclose(G1.nodes[v]['gradient_unary_potential'], G2.nodes[v]['gradient_unary_potential'])
    for e in G1.edges:
        assert np.allclose(G1.edges[e]['gradient_binary_potential'], G2.edges[e]['gradient_binary_potential'])
    assert np.allclose(G1.graph['v_map'], G2.graph['v_map'])
    
def inference_brute_force(G):
    '''
    Perform probabilistic inference on graph G, and compute the gradients of the log-likelihood
    Input: 
        G: 
            a Graph object in the NetworkX library (version 2.2, https://networkx.github.io/)
        K: 
            G.graph['K']
        unary potentials: 
            G.nodes[v]['unary_potential'], a 1-d numpy array of length K
        binary potentials: 
            G.edges[u, v]['binary_potential'], a 2-d numpy array of shape K x K
        assignment for computing the gradients: 
            G.nodes[v]['assignment'], an integer within [0, K - 1]
    Output:
        G.nodes[v]['marginal_prob']: 
            the marginal probability distribution for v, a 1-d numpy array of length K
        G.graph['v_map']: 
            the MAP assignment, a 1-d numpy arrary of length n, where n is the number of vertices
        G.nodes[v]['gradient_unary_potential']: 
            the gradient of the log-likelihood w.r.t. the unary potential of vetext v, a 1-d numpy array of length K
        G.edges[u, v]['gradient_binary_potential']: 
            the gradient of the log-likelihood w.r.t. the binary potential of edge (u, v), a 2-d numpy array of shape K x K
    '''
    # initialize the output buffers
    for v in G.nodes:
        G.nodes[v]['marginal_prob'] = np.zeros(G.graph['K'])
        G.nodes[v]['gradient_unary_potential'] = np.zeros(G.graph['K'])
    for e in G.edges:
        G.edges[e]['gradient_binary_potential'] = np.zeros((G.graph['K'], G.graph['K']))
    G.graph['v_map'] = np.zeros(len(G.nodes))
    
    # YOUR CODE STARTS
    # Calculate overall probabilities
    P_overall=np.zeros(G.graph['K']**len(G.nodes))
    for i in range(G.graph['K']**len(G.nodes)):
        P=1
        nodes_value=decode(i, G.graph['K'], len(G.nodes))
        for j in G.nodes:
            P*=G.nodes[j]['unary_potential'][nodes_value[j]]
        for e in G.edges:
            u,v=e
            P*=G.edges[u,v]['binary_potential'][nodes_value[u],nodes_value[v]]
                    
        P_overall[i]=P
    P_overall=P_overall/np.sum(P_overall)
    # MAP
    G.graph['v_map']=np.array(decode(np.argmax(P_overall), G.graph['K'], len(G.nodes)))
    # Calculate marginal probabilities
    for i in G.nodes:
        for j in range(G.graph['K']):
            P_marginal=0
            for k in range(G.graph['K']**len(G.nodes)):
                if decode(k, G.graph['K'], len(G.nodes), i)==j:
                    P_marginal+=P_overall[k]
            G.nodes[i]['marginal_prob'][j]=P_marginal
    # Calculate gradients for nodes
    for i in G.nodes:
        for j in range(G.graph['K']):
            if j==G.nodes[i]['assignment']:
                G.nodes[i]['gradient_unary_potential'][j]=(1-G.nodes[i]['marginal_prob'][j])/G.nodes[i]['unary_potential'][j]
            else:
                G.nodes[i]['gradient_unary_potential'][j]+=-G.nodes[i]['marginal_prob'][j]/G.nodes[i]['unary_potential'][j]
    # Calculate gradients for edges
    Pij=np.zeros((G.graph['K'], G.graph['K']))
    
    for e in G.edges:
        u,v=e
        for i in range(G.graph['K']):
            for j in range(G.graph['K']):
                Pij=0
                for index in range(G.graph['K']**len(G.nodes)):
                    if decode(index, G.graph['K'], len(G.nodes), u)==i and decode(index, G.graph['K'], len(G.nodes), v)==j:
                        Pij+=P_overall[index]                                    
                if i==G.nodes[u]['assignment'] and j==G.nodes[v]['assignment']:
                    G.edges[e]['gradient_binary_potential'][i,j]= (1-Pij)/G.edges[e]['binary_potential'][i,j]
                else:
                    G.edges[e]['gradient_binary_potential'][i,j]+=-Pij/G.edges[e]['binary_potential'][i,j] 

def inference(G):
    '''
    Perform probabilistic inference on graph G, and compute the gradients of the log-likelihood
    Input: 
        G: 
            a Graph object in the NetworkX library (version 2.2, https://networkx.github.io/)
        K: 
            G.graph['K']
        unary potentials: 
            G.nodes[v]['unary_potential'], a 1-d numpy array of length K
        binary potentials: 
            G.edges[u, v]['binary_potential'], a 2-d numpy array of shape K x K
        assignment for computing the gradients: 
            G.nodes[v]['assignment'], an integer within [0, K - 1]
    Output:
        G.nodes[v]['marginal_prob']: 
            the marginal probability distribution for v, a 1-d numpy array of length K
        G.graph['v_map']: 
            the MAP assignment, a 1-d numpy arrary of length n, where n is the number of vertices
        G.nodes[v]['gradient_unary_potential']: 
            the gradient of the log-likelihood w.r.t. the unary potential of vetext v, a 1-d numpy array of length K
        G.edges[u, v]['gradient_binary_potential']: 
            the gradient of the log-likelihood w.r.t. the binary potential of edge (u, v), a 2-d numpy array of shape K x K
    '''
    # initialize the output buffers
    for v in G.nodes:
        G.nodes[v]['marginal_prob'] = np.zeros(G.graph['K'])
        G.nodes[v]['gradient_unary_potential'] = np.zeros(G.graph['K'])
    for e in G.edges:
        G.edges[e]['gradient_binary_potential'] = np.zeros((G.graph['K'], G.graph['K']))
    G.graph['v_map'] = np.zeros(len(G.nodes))
    
    # YOUR CODE STARTS
    # initialize messages
    for v in G.nodes:
        G.nodes[v]['msg']=np.zeros((G.degree[v], G.graph['K']))
        G.nodes[v]['map_msg']=np.zeros((G.degree[v], G.graph['K']))
        G.nodes[v]['map_value']=np.zeros((G.degree[v], G.graph['K']))
    # up message passing
    # set node 0 as root and transfer G to a tree
    T = nx.bfs_tree(G, 0)
    # Using tree traversal to get the passing orders
    postorder = list(nx.dfs_postorder_nodes(T, source=0))
    preorder = list(nx.dfs_preorder_nodes(T, source=0))
    del T
    for i in postorder:
        if i==0:
            break
        for j in G.neighbors(i):
            if preorder.index(j)>preorder.index(i): 
                continue
            # Get the message from i to j
            for valuej in range(G.graph['K']):
                # calculate the message from i to j when xj=valuej
                msg=0
                map_msg=0
                for valuei in range(G.graph['K']):
                    # add one item of xj=valuej, xi=valuei to the msg
                    # sum-product Message
                    if i<j:
                        partial_msg=G.nodes[i]['unary_potential'][valuei]*G.edges[i,j]['binary_potential'][valuei, valuej]
                    else:
                        partial_msg=G.nodes[i]['unary_potential'][valuei]*G.edges[j,i]['binary_potential'][valuej, valuei]
                    # max-product message
                    if i<j:
                        partial_map_msg=G.nodes[i]['unary_potential'][valuei]*G.edges[i,j]['binary_potential'][valuei, valuej]
                    else:
                        partial_map_msg=G.nodes[i]['unary_potential'][valuei]*G.edges[j,i]['binary_potential'][valuej, valuei]
                    # multiply the messages from i's other neighbors to i
                    for k in G.neighbors(i):
                        if k!=j:
                            assert G.nodes[i]['msg'][list(G.neighbors(i)).index(k), valuei]!=0, f"{k}->{i} message not ready, j={j}"
                            partial_msg*=G.nodes[i]['msg'][list(G.neighbors(i)).index(k), valuei]
                            partial_map_msg*=G.nodes[i]['map_msg'][list(G.neighbors(i)).index(k), valuei]
                    if map_msg<partial_map_msg:
                        map_msg=partial_map_msg
                        G.nodes[i]['map_value'][list(G.neighbors(i)).index(j), valuej] = valuei
                    msg+=partial_msg
                G.nodes[j]['msg'][list(G.neighbors(j)).index(i), valuej]=msg
                G.nodes[j]['map_msg'][list(G.neighbors(j)).index(i), valuej]=map_msg
            G.nodes[j]['msg'][list(G.neighbors(j)).index(i), :]/=np.sum(G.nodes[j]['msg'][list(G.neighbors(j)).index(i), :])
            G.nodes[j]['map_msg'][list(G.neighbors(j)).index(i), :]/=np.sum(G.nodes[j]['map_msg'][list(G.neighbors(j)).index(i), :])
    
    
    # MAP for root
    G.graph['v_map'][0]=np.argmax(np.prod(G.nodes[0]['map_msg'],axis=0)*G.nodes[0]['unary_potential'])
    # down message passing
    for i in preorder:
        # MAP for other nodes
        if i!=0:
            count=0
            for k in G.neighbors(i):
                # Find parent
                if preorder.index(k)<preorder.index(i):
                    G.graph['v_map'][i]=G.nodes[i]['map_value'][list(G.neighbors(i)).index(k), int(G.graph['v_map'][k])]
                    count+=1
            assert count==1, f"Node {i} has {count} parents"
        if G.degree[i]==1 and i!=0:
            continue
        for j in G.neighbors(i):
            if preorder.index(j)<preorder.index(i): 
                continue
            # Get the message from i to j
            for valuej in range(G.graph['K']):
                # calculate the message from j to i when xi=valuei
                msg=0
                for valuei in range(G.graph['K']):
                    # add one item of xj=valuej, xi=valuei to the msg
                    if i<j:
                        partial_msg=G.nodes[i]['unary_potential'][valuei]*G.edges[i,j]['binary_potential'][valuei, valuej]
                    else:
                        partial_msg=G.nodes[i]['unary_potential'][valuei]*G.edges[j,i]['binary_potential'][valuej, valuei]
                    # multiply the messages from i's other neighbors to i
                    for k in G.neighbors(i):
                        if k!=j:
                            assert G.nodes[i]['msg'][list(G.neighbors(i)).index(k), valuei]!=0, f"{k}->{i} message not ready, j={j}"
                            partial_msg*=G.nodes[i]['msg'][list(G.neighbors(i)).index(k), valuei]
                    msg+=partial_msg
                G.nodes[j]['msg'][list(G.neighbors(j)).index(i), valuej]=msg
            G.nodes[j]['msg'][list(G.neighbors(j)).index(i), :]/=np.sum(G.nodes[j]['msg'][list(G.neighbors(j)).index(i), :])
    # Marginalize and Normalize
    for v in G.nodes:
        G.nodes[v]['marginal_prob']=G.nodes[v]['unary_potential']*np.prod(G.nodes[v]['msg'], axis=0)
        G.nodes[v]['marginal_prob']/=np.sum(G.nodes[v]['marginal_prob'])
        # Calculate the gradients for nodes
        G.nodes[v]['gradient_unary_potential']=-G.nodes[v]['marginal_prob']/G.nodes[v]['unary_potential']
        G.nodes[v]['gradient_unary_potential'][G.nodes[v]['assignment']]+=1/G.nodes[v]['unary_potential'][G.nodes[v]['assignment']]
    # Calculate the gradients for edges
    for e in G.edges:
        u,v=e
        # Calculate Puv
        index_v=list(G.neighbors(u)).index(v)
        index_u=list(G.neighbors(v)).index(u)
        msg_u=np.concatenate((G.nodes[u]['msg'][:index_v], G.nodes[u]['msg'][index_v+1:]), axis=0)
        msg_v=np.concatenate((G.nodes[v]['msg'][:index_u], G.nodes[v]['msg'][index_u+1:]), axis=0)
        msg_u=np.prod(msg_u, axis=0).reshape(1,-1)
        msg_v=np.prod(msg_v, axis=0).reshape(1,-1)
        G.edges[e]['gradient_binary_potential']=msg_u.T @ msg_v
        G.edges[e]['gradient_binary_potential']=G.nodes[u]['unary_potential'].reshape(-1,1)*G.edges[e]['gradient_binary_potential']*G.nodes[v]['unary_potential'].reshape(1,-1)
        G.edges[e]['gradient_binary_potential']=G.edges[e]['gradient_binary_potential']*G.edges[e]['binary_potential']
        G.edges[e]['gradient_binary_potential']/=np.sum(G.edges[e]['gradient_binary_potential'])
        # Calculate the gradients for edges use Puv
        G.edges[e]['gradient_binary_potential']=-G.edges[e]['gradient_binary_potential']/G.edges[e]['binary_potential']
        G.edges[e]['gradient_binary_potential'][G.nodes[u]['assignment'], G.nodes[v]['assignment']]+=1/G.edges[e]['binary_potential'][G.nodes[u]['assignment'], G.nodes[v]['assignment']]
        
        
    # Transform into integers
    G.graph['v_map'] = G.graph['v_map'].astype(int)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str, help='The input graph')
    args = parser.parse_args()
    G = load_graph(args.input)
    inference(G)
    # inference_brute_force(G)
    ### For debugging only: check if two solutions are the same
    # G2=G.copy()
    # assert len(G.nodes)<=30
    # assert G2.graph["K"]**len(G2.nodes)<=2**30
    # inference_brute_force(G2)
    # check_equal(G, G2)
    ### End of debugging
    pickle.dump(G, open('results_' + args.input, 'wb'))