import collections
from typing import List, Dict, Set  # module for working with annotations
from datetime import datetime  # module for working with time
start_time = datetime.now()  # marking the time of beginning


class Vertex:
    def __init__(self, name):
        self.name = name  # Unique Name of vertex in graph.
        self.index = 0  # Index of a vertex assigned during the DFS.
        self.neighbours = list()  # List of neighbours on graph.
        self.parent = None  # Ancestor of vertex in graph.

    def add_neighbour(self, v) -> None:
        if v not in self.neighbours:
            self.neighbours.append(v)


class Edge:
    def __init__(self, v, u):
        self.index = 0  # Index of an edge assigned during the DFS.
        self.type = None  # "TREE" or "BACK", None if not visited in DFS
        self.pair = (v, u)  # A pair of vertices defining an edge
        self.low = None  # low(e)
        self.fringe = set()  # Fringe(e) : set(Edge)
        self.status = ""  # Edge status: {"BLOCK", "THIN", "THICK"}
        self.color = 0  # colors = {-1, 1}, 0 is no color
        self.isProcessed = False  # is edge processed is Graph.embed()
        self.DS = []  # DS(e) : List[Edge]


def interlaced(edge1: Edge, edge2: Edge) -> List[Edge]:  # Interlaced(e1, e2)
    res = list()
    for f in edge1.fringe:
        if f.low.index > edge2.low.index:
            res.append(f)
    return res


class Graph:
    vertices = {}  # Dict{key = vertex : value = set of adjacent vertexes}
    edges = {}  # Dict{key = vertex : value = list of adjacent edges}
    DFCount = 0  # Counter for DFS
    start_vertex: Vertex  # The vertex from which DFS starts
    previous_edge = None  # The variable required to compute Fringe(e)
    ttt = 0  # int counter for indexing edges

    def add_vertex(self, v) -> bool:
        if isinstance(v, Vertex) and v.name not in self.vertices:
            self.vertices[v] = set()
            return True
        else:
            return False

    def add_edge(self, u, v) -> bool:
        if u in self.vertices and v in self.vertices and u != v:  #
            for key, value in self.vertices.items():
                if key == u:
                    u.add_neighbour(v)
                    self.vertices[u].add(v)
                    if self.edges.get(u) is None:
                        self.edges[u] = list()
                    self.edges[u].append(Edge(u, v))
                    # self.edges[u].add(Edge(v, u))
                if key == v:
                    v.add_neighbour(u)
                    self.vertices[v].add(u)
                    if self.edges.get(v) is None:
                        self.edges[v] = list()
                    self.edges[v].append(Edge(v, u))
                    # self.edges[v].add(Edge(u, v))

            return True
        else:
            return False

    def print_adjacency_list_vertex(self):
        """
        This method prints adjacency lists
        of vertices in graph (self.vertices).
        """
        for key in list(self.vertices.keys()):
            print(f"{key.name}: [", end=' ')
            for neigh in self.vertices[key]:
                print(neigh.name, end=' ')
            print(']')

    def print_adjacency_list_edges(self):
        """
        This method prints adjacency lists
        of edges (key: Vertex) in graph (self.edges).
        """
        for vertex, edges in self.edges.items():
            print(vertex.name + ': [', end=' ')
            for edge in edges:
                print(edge.index, end=' ')
            print(']')

    def get_number_of_edges(self) -> int:
        i = 0
        for arr in self.edges.values():
            i += len(arr)
        return i

    def dfs(self, vertex: Vertex) -> None:
        self.start_vertex = vertex
        self._dfs(vertex)

    def _dfs(self, vertex: Vertex) -> None:
        self.DFCount += 1
        vertex.index = self.DFCount
        temp_counter = -1
        for v in vertex.neighbours:
            if v.index == 0:
                self.ttt += 1
                print(vertex.name, v.name, "tree")
                v.parent = vertex
                self.vertices[v].remove(vertex)  # make edge is oriented
                for e in self.edges[vertex]:  # mark edge as a tree arc and make oriented
                    if e.pair == (vertex, v):
                        e.type = "TREE"
                        e.index = self.ttt  # vertex.index  # e.index = self.DFCount
                for e_b in self.edges[v]:
                    if e_b.pair == (v, vertex):
                        self.edges[v].remove(e_b)
                        break
                self._dfs(v)

                # calculating low(e)
                for edge in self.edges[vertex]:
                    if edge.pair == (vertex, v):
                        for e in self.edges[v]:
                            if edge.low is None or e.low.index <= edge.low.index:
                                edge.low = e.low
            # to avoid exploring an edge in both directions
            elif v.index < vertex.index and v != vertex.parent:
                self.ttt += 1
                temp_counter += 1
                print(vertex.name, v.name, "back")
                self.vertices[v].remove(vertex)  # make edge is oriented
                for e in self.edges[vertex]:  # mark edge as a back edge and make oriented
                    if e.pair == (vertex, v):
                        e.type = "BACK"
                        e.index = self.ttt  # vertex.index + temp_counter
                        e.low = v  # calculating low(e)
                for e_b in self.edges[v]:
                    if e_b.pair == (v, vertex):
                        self.edges[v].remove(e_b)
                        break

            # calculating Fringe(e)
            for edge in self.edges[vertex]:
                if edge.pair == (vertex, v):
                    # copy suitable edges from previous recursion step
                    if self.previous_edge is not None:
                        for f in self.previous_edge.fringe:
                            if f.low.index < vertex.index:
                                edge.fringe.add(f)
                    edges = self.edges[v] + [edge]  # the edge itself can also be
                    for e in edges:
                        if e.index > edge.index and e.type == "BACK" and e.low.index < vertex.index:
                            edge.fringe.add(e)
                    self.previous_edge = edge

    def compute_edge_status(self):
        """
        This method assign status of each edge:
        {"BLOCK", "THIN", "THICK"}.
        """
        def is_low_equal(fringe: List[Edge]) -> bool:
            """
            This is an auxiliary function for checking
            the equality of low (e) of each vertex from Fringe(e).
            """
            isFirst = True
            comparable = None
            for e in fringe:
                if isFirst:
                    comparable = e.low
                    isFirst = False
                if e.low != comparable:
                    return False
            return True

        for edges in self.edges.values():
            for edge in edges:
                if edge.fringe == {}:
                    edge.status = "BLOCK"
                elif is_low_equal(edge.fringe):
                    edge.status = "THIN"
                else:
                    edge.status = "THICK"

    def bucket_sort(self) -> None:  # The second step of algorithm - sorting
        for edges in self.edges.values():
            edges.sort(key=lambda edge: edge.low.index)

    def coloring_constraint(self, edge1: Edge, edge2: Edge) -> bool:
        """
        This method checks F-coloring constraints.
        """
        interlaced_e1_e2 = interlaced(edge1, edge2)
        interlaced_e2_e1 = interlaced(edge2, edge1)
        colored = []
        for e_interlaced in interlaced_e1_e2:
            edge = None
            for edges_list in self.edges.values():
                for ed in edges_list:
                    if ed == e_interlaced:
                        edge = ed
                        break
            if edge is not None:
                if edge.color == 0:
                    edge.color = -1
                elif edge.color == -1:
                    edge.color = 1
                elif edge.color == 1:
                    edge.color = -1
                colored.append(edge)
        for i in range(len(colored) - 1):
            if colored[i].color != colored[i+1].color:
                return False
        colored.clear()

        for e_interlaced in interlaced_e2_e1:
            edge = None
            for edges_list in self.edges.values():
                for ed in edges_list:
                    if ed == e_interlaced:
                        edge = ed
                        break
            if edge is not None:
                if edge.color == 0:
                    edge.color = -1
                elif edge.color == -1:
                    edge.color = 1
                elif edge.color == 1:
                    edge.color = -1
                colored.append(edge)
        for i in range(len(colored) - 1):
            if colored[i].color != colored[i+1].color:
                return False
        colored.clear()

        return True

    def embed(self) -> bool:
        """
        This method is the third and main step of the algorithm.
        It checks the planarity of the graph.
        If function returns True - graph is planar.
        If method returns False - non planar.
        """
        def isAllProcessed() -> bool:
            """
            This function will be used to check the
            processing of all edges coming from the vertices, except for the root
            """
            for v in self.edges.keys():
                if self.edges[v] != self.edges[self.start_vertex]:
                    for edge in self.edges[v]:
                        if not edge.isProcessed:
                            return False
            return True

        for v in self.edges.keys():
            for edge in self.edges[v]:  # We say that all the cotree edges have been
                if edge.type == "BACK":  # processed and that the tree edges are still unprocessed.
                    edge.DS.append(edge)  # DS(e) is empty if e is a tree edge and includes e (with no bicoloration
                    edge.isProcessed = True   # constraints) if e is a cotree edge

        while not isAllProcessed():
            for v in self.edges.keys():
                if self.edges[v] != self.edges[self.start_vertex]:
                    parent = v.parent
                    for ed in self.edges[parent]:
                        if ed.pair == (parent, v):
                            e = ed
                    temp_counter = 0
                    prev_edge: Edge
                    for e_i in self.edges[v]:
                        for e_ds in e_i.DS:
                            temp_counter += 1
                            prev_edge = e_ds
                            if temp_counter == 1:
                                e.DS.append(e_ds)
                            elif not self.coloring_constraint(e_ds, prev_edge):
                                return False  # nonplanar
                        temp_counter = 0
                    for edge in e.DS:
                        if edge.type == "BACK" and edge.pair[0] == e.pair[0]:
                            e.DS.remove(edge)
                    e.isProcessed = True
        return True


def main() -> None:
    # Initializing an example graph
    vertices = []
    for i in range(80):
        vertices.append(Vertex(str(i)))
    graph = Graph()
    for i in vertices:
        graph.add_vertex(i)
    for i in range(0, len(vertices) - 1):
        for j in range(1, len(vertices)):
            if j - i == 1:
                graph.add_edge(vertices[i], vertices[j])
                print("Added edge between", vertices[i].name, "and", vertices[j].name)
    for j in range(1, len(vertices)):
        if 2 <= j <= len(vertices) - 1:
            graph.add_edge(vertices[0], vertices[j])
            print("Added edge between", vertices[0].name, "and", vertices[j].name)

    # Logs to the console
    print("Adjacency list of original graph:")
    graph.print_adjacency_list_vertex()
    print(f"This graph has {graph.get_number_of_edges()} edges, so each edge has both directions.")
    graph.edges = collections.OrderedDict(sorted(graph.edges.items(), key=lambda pair: pair[0].name))
    graph.print_adjacency_list_edges()
    graph.dfs(vertices[0])
    print("Adjacency list of modified graph after first DFS:")
    graph.print_adjacency_list_vertex()
    print(f"This oriented graph now has {graph.get_number_of_edges()} edges.")
    graph.compute_edge_status()
    graph.print_adjacency_list_edges()
    graph.bucket_sort()
    print("Before embedding we need bucket sort. Now adjacency list is:")
    graph.print_adjacency_list_edges()
    print(graph.embed())


if __name__ == '__main__':
    main()
    end_time = datetime.now()  # marking the time of the end
    print('Duration: {0:.3f} milliseconds'.format(
        end_time.microsecond / 1000 - start_time.microsecond / 1000))
