from collections import deque  # module for working with deque
from typing import List, Dict  # module for working with annotations
from datetime import datetime  # module for working with time
start_time = datetime.now()  # marking the time of beginning


class Vertex:
    def __init__(self, name):
        self.name = name  # Unique Name of vertex in graph.
        self.index = 0  # Index of a vertex assigned during the DFS.
        self.neighbours = list()  # List of neighbours on graph.
        self.parent = None  # Ancestor of vertex in graph.
        self.lowPoint1 = self  # Variables assigned during the DFS
        self.lowPoint2 = self  # for bucket sorting.
        self.firstPath = None  # denotes the number of the first
        # path containing v

    def add_neighbour(self, v) -> None:
        if v not in self.neighbours:
            self.neighbours.append(v)


class Edge:
    def __init__(self, v, u):
        self.index = 0  # Index of an edge assigned during the DFS.
        self.type = None  # "TREE" or "BACK", None if not visited in DFS
        self.pair = (v, u)  # A pair of vertices defining an edge

    def sort_f(self) -> None:
        """
        Function for bucket sort.
        """
        v = self.pair[0]
        w = self.pair[1]
        if self.type == "BACK":
            return 2 * w.index
        elif self.type == "TREE" and w.lowPoint2 >= v.index:
            return 2 * w.lowPoint1
        elif self.type == "TREE" and w.lowPoint2 < v.index:
            return 2 * w.lowPoint1 + 1


class Path:
    def __init__(self, edge):
        # list of vertices in path
        self.path_vertices = [edge.pair[0], edge.pair[1]]
        # list of edges in path
        self.path_edges = [edge]
        self.index = 0

    def path_append(self, edge) -> None:
        """
        This method adds an edge to the current path.
        """
        self.path_vertices.append(edge.pair[0])
        self.path_vertices.append(edge.pair[1])
        self.path_edges.append(edge)


class Graph:
    vertices = {}  # Dict{key = vertex : value = set of adjacent vertexes}
    edges = {}  # Dict{key = vertex : value = list of adjacent edges}
    bucket = []
    adj_list = {}
    paths = deque()
    global DFCount  # Counter for DFS
    DFCount = 0
    global cur_start
    cur_start = None
    start_vertex: Vertex  # The vertex from which DFS was launched
    global counter
    counter = 0

    def add_vertex(self, v) -> bool:
        if isinstance(v, Vertex) and v.name not in self.vertices:
            self.vertices[v] = set()
            return True
        else:
            return False

    def add_edge(self, u, v) -> bool:
        if u in self.vertices and v in self.vertices and u != v:
            for key, value in self.vertices.items():
                if key == u:
                    u.add_neighbour(v)
                    self.vertices[u].add(v)
                    if self.edges.get(u) is None:
                        self.edges[u] = set()
                    self.edges[u].add(Edge(u, v))
                if key == v:
                    v.add_neighbour(u)
                    self.vertices[v].add(u)
                    if self.edges.get(v) is None:
                        self.edges[v] = set()
                    self.edges[v].add(Edge(v, u))

            return True
        else:
            return False

    def print_adjacency_list_ns(self):
        """
        This method prints adjacency lists
        of vertices in graph (self.vertices).
        """
        for key in list(self.vertices.keys()):
            print(f"{key.name}: [", end=' ')
            for neigh in self.vertices[key]:
                print(neigh.name, end=' ')
            print(']')

    def print_adjacency_list_s(self):
        """
        This method prints adjacency lists
        of edges (key: Vertex) in graph (self.edges).
        """
        for key in list(self.adj_list.keys()):
            print(key.name + ': [', end=' ')
            for neigh in self.adj_list[key]:
                print(neigh.name, end=' ')
            print(']')

    def get_number_of_edges(self) -> int:
        i = 0
        for arr in self.edges.values():
            i += len(arr)
        return i

    def dfs(self, vertex) -> None:
        self.start_vertex = vertex
        self._dfs(vertex)

    def _dfs(self, vertex) -> None:
        global DFCount
        DFCount += 1
        vertex.index = DFCount
        vertex.lowPoint1 = vertex.lowPoint2 = vertex.index
        for v in vertex.neighbours:
            if v.index == 0:
                v.parent = vertex
                self.vertices[v].remove(vertex)  # make edge is oriented
                for e in self.edges[vertex]:  # mark edge as a tree arc and make oriented
                    if e.pair == (vertex, v):
                        e.type = "TREE"
                        e.index += DFCount
                for e_b in self.edges[v]:
                    if e_b.pair == (v, vertex):
                        self.edges[v].remove(e_b)
                        break
                self._dfs(v)

                # Backtracking. Here we calculate lowpoints of vertex
                if v.lowPoint1 < vertex.lowPoint1:  # dummy statement b
                    vertex.lowPoint2 = min(vertex.lowPoint1, v.lowPoint2)
                    vertex.lowPoint1 = v.lowPoint1
                elif v.lowPoint1 == vertex.lowPoint1:
                    vertex.lowPoint2 = min(vertex.lowPoint2, v.lowPoint2)
                else:
                    vertex.lowPoint2 = min(vertex.lowPoint2, v.lowPoint1)

            elif v.index < vertex.index and \
                    v != vertex.parent:  # to avoid exploring an edge in both directions
                self.vertices[v].remove(vertex)  # make edge oriented
                for e in self.edges[vertex]:  # mark edge as a back edge and make oriented
                    if e.pair == (vertex, v):
                        e.type = "BACK"
                        e.index += DFCount
                for e_b in self.edges[v]:
                    if e_b.pair == (v, vertex):
                        self.edges[v].remove(e_b)
                        break
                # Calculating lowpoints
                if v.index < vertex.lowPoint1:
                    vertex.lowPoint2 = vertex.lowPoint1
                    vertex.lowPoint1 = v.index
                elif v.index > vertex.lowPoint1:
                    vertex.lowPoint2 = min(vertex.lowPoint2, v.index)

    def bucket_sort(self):
        """
        This function sorts the vertices in
        the dictionary(self.adj_list) in a specific
        way for further method self.embed().
        """
        for i in range(2 * len(self.vertices.keys()) + 1):
            self.bucket.append(list())
        for e_set in self.edges.values():
            for e in e_set:
                self.bucket[e.sort_f()].append(e)
        for v in self.vertices.keys():
            self.adj_list[v] = list()
        for i in range(2 * len(self.vertices.keys()) + 1):
            for e in self.bucket[i]:
                self.adj_list[e.pair[0]].append(e.pair[1])

    def embed(self) -> bool:
        """
        This is the main function for checking whether all graph
        segments can be embedded on a plane. Before calling it,
        you need to call self.DFS() and sort adjacency list by
        self.bucket_sort().
        """

        def pathfinder(vertex: Vertex):
            nonlocal Stack, Next, \
                Block, path_counter, path_start, free
            for w in self.adj_list[vertex]:
                for edge in self.edges[vertex]:
                    if edge.pair == (vertex, w):
                        if edge.type == "TREE":
                            if path_start == 0:
                                path_start = vertex
                                path_counter += 1
                                self.paths.append(Path(edge))  # +
                            w.firstPath = path_counter
                            self.paths[-1].path_append(edge)  # +
                            print(vertex.name, w.name, "tree")  # +
                            pathfinder(w)
                            # delete stack entries and blocks corresponding to vertices no smaller than v;
                            for block in Block:
                                x = block[0]
                                y = block[1]
                                if ((x in Stack and x >= vertex.index) or x == 0) \
                                        and ((y in Stack and y >= vertex.index) or y == 0):
                                    Block.remove(block)
                            for block in Block:
                                x = block[0]
                                y = block[1]
                                if x in Stack and x >= vertex.index:
                                    block[0] = 0
                                    block[1] = y
                                if y in Stack and y >= vertex.index:
                                    block[0] = x
                                    block[1] = 0
                            while Next[0] != 0 and Stack[Next[0]] >= vertex.index:
                                Next[0] = Next[Next[0]]
                            while Next[1] != 0 and Stack[Next[1]] >= vertex.index:
                                Next[1] = Next[Next[1]]
                            if w.firstPath != vertex.firstPath:
                                #  all of segment with first edge (v, w) has been embedded. New blocks must be
                                #  moved from right to left
                                left_ = 0
                                for block in Block:
                                    x = block[0]
                                    y = block[1]
                                    if (x in Stack and x > self.paths[w.firstPath].path_vertices[-1].index) or (
                                            y in Stack and y > self.paths[w.firstPath].path_vertices[-1].index) \
                                            and Stack[Next[0]] != 0:
                                        if x in Stack and x > self.paths[w.firstPath].path_vertices[-1].index:
                                            if y in Stack and y > self.paths[w.firstPath].path_vertices[-1].index:
                                                return False  # nonplanar
                                            left_ = x
                                        else:  # y in Stack and y.index > w.firstPath.path_vertices[-1]
                                            save = Next[left_ + 1]
                                            Next[left_ + 1] = Next[y + 1]
                                            Next[y + 1] = save
                                            left_ = y
                                        Block.remove(block)
                                # block on B must be combined with new blocks just deleted;
                                if Block:
                                    block = Block[-1]
                                    x = block[0]
                                    y = block[1]
                                    Block.remove(block)
                                    if x != 0:
                                        Block.append(block)
                                    elif left_ != 0 or y != 0:
                                        Block.append([left_, y])
                                    # delete end-of-stack marker on right stack;
                                    Next[0] = Next[Next[0]]
                        # v -- -> w. Current path is complete. Path is normal if f(PATH(s)) < w;
                        elif edge.type == "BACK":
                            if path_start == 0 or path_start.index == 0:
                                path_counter += 1
                                path_start = vertex
                                self.paths.append(Path(edge))
                            self.paths[path_counter - 1].path_append(edge)
                            print(vertex.name, w.name, "back")  # +
                            # switch blocks of entries from left
                            # to right so that p may be embedded on left;
                            left_ = 0
                            right_ = -1
                            while (Next[left_ + 1] != 0 and Stack[Next[left_ + 1]] > w.index) or \
                                    (Next[right_ + 1] != 0 and Stack[Next[right_ + 1]] > w.index):
                                for block in Block:
                                    x = block[0]
                                    y = block[1]
                                    if x != 0 and y != 0:
                                        if Stack[Next[left_ + 1]] > w.index:
                                            if Stack[Next[left_ + 1]] > w.index:
                                                return False  # nonplanar
                                            save = Next[right_ + 1]
                                            Next[right_ + 1] = Next[left_ + 1]
                                            Next[left_ + 1] = save
                                            save = Next[x + 1]
                                            Next[x + 1] = Next[y + 1]
                                            Next[y + 1] = save
                                            left_ = y
                                            right_ = x
                                        else:  # STACK(NEXT(R')) > w;
                                            left_ = x
                                            right_ = y
                                    elif x != 0:  # STACK (NEXT(L')) > w;
                                        save = Next[x + 1]
                                        Next[x + 1] = Next[right_ + 1]
                                        Next[right_ + 1] = Next[left_ + 1]
                                        Next[left_ + 1] = save
                                        right_ = x
                                    elif y != 0:
                                        right_ = y
                                    Block.remove(block)
                            # add P to left stack if p is normal;
                            if self.paths[path_start.firstPath - 1].path_vertices[-1].index < w.index:
                                if left_ == 0:
                                    left_ = free

                                Stack[free] = self.paths[path_start.firstPath - 1].path_vertices[
                                    -1].index  # ?
                                Next[free + 1] = Next[1]
                                Next[1] = free
                                free += 1
                            # add new block corresponding to combined old blocks. New block may be empty
                            # if segment containing current path is not a single frond;
                            if right_ == -1:
                                right_ = 0
                            if left_ != 0 or right_ != 0 or vertex.index != path_start.index:
                                Block.append([left_, right_])
                            # if segment containing current path
                            # is not a single frond, add an end-of-stack
                            # marker to right stack;
                            if vertex.index != path_start.index:
                                Stack[free] = 0
                                Next[free + 1] = Next[0]
                                Next[0] = free
                                free += 1
                            path_start.index = 0

        # Initialization
        Stack = [0 for i in range(self.get_number_of_edges())]
        print(len(Stack))
        Next = [0 for i in range(self.get_number_of_edges() + 1)]
        Block = []
        Next.append(0)  # Next[0] = Next[1] = 0
        Next.append(0)
        free = 1
        Stack.append(0)
        self.start_vertex.firstPath = 1
        path_start = 0
        path_counter = 0
        pathfinder(self.start_vertex)
        return True


def main() -> None:
    # Initializing an example graph
    vertices = []
    for i in range(100):
        vertices.append(Vertex(str(i)))
    graph = Graph()
    for i in vertices:
        graph.add_vertex(i)
    for i in range(0, len(vertices) - 1):
        for j in range(1, len(vertices)):
            if j - i == 1:
                graph.add_edge(vertices[i], vertices[j])
                print("Added edge between", vertices[i].name,
                      "and", vertices[j].name)
    for j in range(1, len(vertices)):
        if 2 <= j <= len(vertices) - 1:
            graph.add_edge(vertices[0], vertices[j])
            print("Added edge between", vertices[0].name,
                  "and", vertices[j].name)

    # Logs to the console
    print("Adjacency list of original graph:")
    graph.print_adjacency_list_ns()
    print(f"This graph has {graph.get_number_of_edges()} edges, so each edge has both directions.")
    graph.dfs(vertices[0])
    print("Adjacency list of modified graph after first DFS:")
    graph.print_adjacency_list_ns()
    print(f"This oriented graph now has {graph.get_number_of_edges()} edges.")
    print("These are LowPoint1 and LowPoint2 of each vertex:")
    for v in graph.vertices.keys():
        print(f"{v.name}: [{v.lowPoint1}, {v.lowPoint2}]")
    print("Before pathfinding we need bucket sort. Now adjacency list is:")
    graph.bucket_sort()
    graph.print_adjacency_list_s()
    graph.vertices.clear()  # was replaced by adj_list
    print(graph.embed())


if __name__ == '__main__':
    main()
    end_time = datetime.now()  # marking the time of the end
    print('Duration: {0:.3f} milliseconds'.format(
        end_time.microsecond / 1000 - start_time.microsecond / 1000))
