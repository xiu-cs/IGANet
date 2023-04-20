import numpy as np

class Graph():
    """ The Graph to model the skeletons of human body/hand

    Args:
        strategy (string): must be one of the follow candidates
        - spatial: Clustered Configuration

        layout (string): must be one of the follow candidates
        - 'hm36_gt' same with ground truth structure of human 3.6 , with 17 joints per frame

        max_hop (int): the maximal distance between two connected nodes
        dilation (int): controls the spacing between the kernel points

    """

    def __init__(self,
                 layout, 
                 strategy,
                 pad=0,
                 max_hop=1,
                 dilation=1):

        self.max_hop = max_hop # 1
        self.dilation = dilation # 1
        self.seqlen = pad  # 1
        self.get_edge(layout)
        self.hop_dis = get_hop_distance(self.num_node, self.edge, max_hop=max_hop) # [17,17], 相邻位1，本身为0，其他为inf

        # get distance of each node to center
        self.dist_center = self.get_distance_to_center(layout)  # dist_center 各个节点到joint7的距离s
        self.get_adjacency(strategy)

    def get_distance_to_center(self,layout): 
        """
        :return: get the distance of each node to center (joint 7)
        """
        dist_center = np.zeros(self.num_node)
        if layout == 'hm36_gt':
            for i in range(self.seqlen):
                index_start = i*self.num_node_each
                dist_center[index_start+0 : index_start+7] = [1, 2, 3, 4, 2, 3, 4]
                dist_center[index_start+7 : index_start+11] = [0, 1, 2, 3]
                dist_center[index_start+11 : index_start+17] = [2, 3, 4, 2, 3, 4]
        return dist_center

    def __str__(self):
        return self.A

    def graph_link_between_frames(self,base):
        """
        calculate graph link between frames given base nodes and seq_ind
        :param base:
        :return:
        """
        return [((front) + i*self.num_node_each, (back)+ i*self.num_node_each) for i in range(self.seqlen) for (front, back) in base] # 把每一帧的关节点都连接起来


    def basic_layout(self,neighbour_base, sym_base):
        """
        for generating basic layout time link selflink etc.
        neighbour_base: neighbour link per frame
        sym_base: symmetrical link(for body) or cross-link(for hand) per frame

        :return: link each node with itself
        """
        self.num_node = self.num_node_each * self.seqlen
        time_link = [(i * self.num_node_each + j, (i + 1) * self.num_node_each + j) for i in range(self.seqlen - 1) # for single frame, this is null
                     for j in range(self.num_node_each)]
        self.time_link_forward = [(i * self.num_node_each + j, (i + 1) * self.num_node_each + j) for i in
                                  range(self.seqlen - 1) 
                                  for j in range(self.num_node_each)]
        self.time_link_back = [((i + 1) * self.num_node_each + j, (i) * self.num_node_each + j) for i in
                               range(self.seqlen - 1)
                               for j in range(self.num_node_each)]

        self_link = [(i, i) for i in range(self.num_node)]

        self.neighbour_link_all = self.graph_link_between_frames(neighbour_base)

        self.sym_link_all = self.graph_link_between_frames(sym_base)

        return self_link, time_link

    def get_edge(self, layout):
        """
        get edge link of the graph
        la,ra: left/right arm
        ll/rl: left/right leg
        cb: center bone
        """
        if layout == 'hm36_gt':
            self.num_node_each = 17

            neighbour_base = [(0, 1), (2, 1), (3, 2), (4, 0), (5, 4), (6, 5), 
                              (7, 0), (8, 7), (9, 8), (10, 9), (11, 8),
                              (12, 11), (13, 12), (14, 8), (15, 14), (16, 15)
                              ]
                        
            sym_base = [(6, 3), (5, 2), (4, 1), (11, 14), (12, 15), (13, 16)]  

            self_link, time_link = self.basic_layout(neighbour_base, sym_base) # self_link: node itself; time_link: 

            self.la, self.ra =[11, 12, 13], [14, 15, 16] # left and right arm
            self.ll, self.rl = [4, 5, 6], [1, 2, 3] # left and right leg
            self.cb = [0, 7, 8, 9, 10] # center bone
            self.part = [self.la, self.ra, self.ll, self.rl, self.cb]

            self.edge = self_link + self.neighbour_link_all + self.sym_link_all + time_link # len=39

            # center node of body/hand
            self.center = 8 - 1
        else:
            raise ValueError("Do Not Exist This Layout.")

    def get_adjacency(self, strategy):
        valid_hop = range(0, self.max_hop + 1, self.dilation) # [0, 1]
        adjacency = np.zeros((self.num_node, self.num_node)) 
        for hop in valid_hop:
            adjacency[self.hop_dis == hop] = 1
        normalize_adjacency = normalize_digraph(adjacency) 

        if strategy == 'spatial':
            A = []
            for hop in valid_hop:
                a_root = np.zeros((self.num_node, self.num_node))
                a_close = np.zeros((self.num_node, self.num_node))
                a_further = np.zeros((self.num_node, self.num_node))
                a_sym = np.zeros((self.num_node, self.num_node))
                a_forward = np.zeros((self.num_node, self.num_node))
                a_back = np.zeros((self.num_node, self.num_node))
                for i in range(self.num_node):
                    for j in range(self.num_node):
                        if self.hop_dis[j, i] == hop: # 0:diagonal; 1:adjacent point
                            if (j,i) in self.sym_link_all or (i,j) in self.sym_link_all: # symmetrical node
                                a_sym[j, i] = normalize_adjacency[j, i]
                            elif (j,i) in self.time_link_forward:
                                a_forward[j, i] = normalize_adjacency[j, i]
                            elif (j,i) in self.time_link_back:
                                a_back[j, i] = normalize_adjacency[j, i]
                            elif self.dist_center[j] == self.dist_center[i]: 
                                a_root[j, i] = normalize_adjacency[j, i]
                            elif self.dist_center[j] > self.dist_center[i]:
                                a_close[j, i] = normalize_adjacency[j, i]
                            else:
                                a_further[j, i] = normalize_adjacency[j, i] 

                if hop == 0:
                    A.append(a_root)
                else:
                    A.append(a_close)
                    A.append(a_further)
                    A.append(a_sym)
                    if self.seqlen > 1: 
                        A.append(a_forward)
                        A.append(a_back)

            A = np.stack(A)
            self.A = A

        else:
            raise ValueError("Do Not Exist This Strategy")
            
def get_hop_distance(num_node, edge, max_hop=1): # 建立邻接矩阵,相邻则置0   
    A = np.zeros((num_node, num_node))
    for i, j in edge:
        A[j, i] = 1
        A[i, j] = 1
    # compute hop steps
    hop_dis = np.zeros((num_node, num_node)) + np.inf
    transfer_mat = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]# GET [I,A]; matrix_power计算矩阵次方  0次方对角线全1，1次方不动
    arrive_mat = (np.stack(transfer_mat) > 0) # [2,17,17]
    for d in range(max_hop, -1, -1): # preserve A(i,j) = 1 while A(i,i) = 0  相邻为1 对角为0
        hop_dis[arrive_mat[d]] = d
    return hop_dis

def normalize_digraph(A):
    Dl = np.sum(A, 0) 
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-1)
    AD = np.dot(A, Dn)
    return AD

def normalize_undigraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node)) # 17,17 
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-0.5)
    DAD = np.dot(np.dot(Dn, A), Dn)
    return DAD

if __name__=="__main__":
    graph = Graph('hm36_gt', 'spatial', 1)
    print(graph.A.shape)
    # print(graph)