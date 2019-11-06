'''
convert binary parse tree to a list of distance
'''

# lili's code

def get_position(parsing, idx):
    if type(parsing) == type(''):
        return idx

    idxleft = get_position(parsing[0], idx)
    if idx == idxleft:
        parsing[0] = idx
    idxright = get_position(parsing[1], idxleft+1)
    if idxright == idxleft + 1:
        parsing[1] = idxright
    return idxright

def get_len(parsing):
    return str(parsing).count('[') + 1

def rec_order(parsing, order, begin, num):
    if type( parsing ) == type(''):
        return num

    lenleft = get_len(parsing[0])
    try:
        order[begin + lenleft-1] = num
    except:
        print(parsing[0])
        print( begin, lenleft, len(order) )
        exit(2)
    num=num+1
    num = rec_order(parsing[0], order, begin, num)
    num = rec_order(parsing[1], order, begin+lenleft, num)

    return num

def compute_order(parse):
    order = [0] * (get_len(parse) - 1)
    rec_order(parse, order, 0, 0)
    order = [max(order) + 1. - x for x in order]

    return order

# bowen's code

class my_node(object):
    def __init__(self):
        self.l_child = None
        self.r_child = None
        self.parent = None
        self.data = None

    def set_children(self, x, y):
        self.l_child = x
        self.r_child = y
        x.set_parent(self)
        y.set_parent(self)
    
    def set_parent(self, x):
        self.parent = x

    def get_parent(self):
        return self.parent

    def set_data(self, x):
        self.data = x

    def set_depth(self, d):
        self.depth = d

    def __str__(self):
        return self.data

class my_tree(object):
    def __init__(self, parse_list):
        self.leave_nodes = []
        self.root = my_node()
        self.root.set_depth(1)
        if len(parse_list) < 2:
            self.root.set_data(parse_list[0])
        else:
            self.build_tree(self.root, parse_list[0], parse_list[1], 2)
    
    def build_tree(self, parent, l, r, d):
        l_child = my_node()
        l_child.set_depth(d)
        r_child = my_node()
        r_child.set_depth(d)
        parent.set_children(l_child, r_child)
        if isinstance(l, str):
            l_child.set_data(l)
        else:
            self.build_tree(l_child, l[0], l[1], d+1)
        if isinstance(r, str):
            r_child.set_data(r)
        else:
            self.build_tree(r_child, r[0], r[1], d+1)

    def get_leave_nodes(self):
        self.leave_nodes = []
        stack = [self.root]
        while stack:
            cur_node = stack[0]
            stack.pop(0)
            if cur_node.data is not None:
                self.leave_nodes.append(cur_node)
            else:
                stack.insert(0, cur_node.r_child)
                stack.insert(0, cur_node.l_child)

    def get_depth_of_common_ancestor(self, x, y):
        (x, y) = (x, y) if x.depth < y.depth else (y, x)    # x - shallow one
        for i in range(y.depth - x.depth):
            y = y.get_parent()
        while x is not y:
            x = x.get_parent()
            y = y.get_parent()
        assert x.depth == y.depth
        return x.depth

    def get_distance(self):
        if not self.leave_nodes:
            self.get_leave_nodes()
        dist = []
        for i in range(1, len(self.leave_nodes)):
            x = self.leave_nodes[i-1]
            y = self.leave_nodes[i]
            dist.append(self.get_depth_of_common_ancestor(x, y))
        max_depth = max([x.depth for x in self.leave_nodes]) + 0.
        return [max_depth - x for x in dist]

def compute_order_by_denition(parse):
    parse_tree = my_tree(parse)
    return parse_tree.get_distance()

if __name__ == '__main__':
    parse = ['a', [['b', 'c'],['d', ['e', 'f']]]]
    print compute_order_by_denition(parse)
    print compute_order(parse)
else:
    pass