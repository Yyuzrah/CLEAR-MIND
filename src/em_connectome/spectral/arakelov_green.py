"""Notebook-equivalent Arakelov–Green/effective-resistance spectrum."""

from __future__ import annotations

import networkx as nx
import numpy as np
from numpy.typing import NDArray
from scipy.linalg import eigh, pinvh

from em_connectome.data.records import NeuronRecord
from em_connectome.data.swc import SWCTree

from .base import SpectrumExtractor, SpectrumParameters, register_spectrum_method


class ArakelovGreenSpectrum(SpectrumExtractor):
    """Extract the descending absolute eigenvalues of the Green distance.

    This is a direct migration of ``RawExactSpectralExtractor`` in the champion
    MLP reference notebook. The name "Arakelov–Green" is retained from the
    experiment; numerically, the implementation constructs the effective-
    resistance/Green-distance matrix from a weighted skeleton graph.
    """

    method = "arakelov_green"
    implementation_version = "notebook-old64-eps-tau-affinity1e9-v1"

    def __init__(self, parameters: SpectrumParameters | None = None):
        self.parameters = parameters or SpectrumParameters()
        if self.parameters.method != self.method:
            raise ValueError(f"{type(self).__name__} cannot implement {self.parameters.method!r}")

    def extract(self, record: NeuronRecord) -> NDArray[np.float64]:
        return self.extract_tree(record.morphology)

    def extract_tree(self, tree: SWCTree) -> NDArray[np.float64]:
        graph, coordinates, root = self._build_skeleton(tree)
        graph = graph.copy()

        leaves = [node for node, degree in graph.degree() if degree == 1 and node != root]
        for leaf in leaves:
            distance = float(np.linalg.norm(coordinates[leaf] - coordinates[root]))
            if distance < self.parameters.epsilon and not graph.has_edge(leaf, root):
                graph.add_edge(leaf, root, weight=max(distance, 1e-5))

        nodes = list(graph.nodes())
        if len(nodes) < 2:
            return np.zeros(self.parameters.top_k, dtype=np.float64)

        bridges = {frozenset(edge) for edge in nx.bridges(graph)}
        index = {node: position for position, node in enumerate(nodes)}
        affinity = np.zeros((len(nodes), len(nodes)), dtype=np.float64)
        for left, right, data in graph.edges(data=True):
            i, j = index[left], index[right]
            length = max(float(data["weight"]), 1e-5)
            is_noise = (
                frozenset((left, right)) in bridges and length < self.parameters.noise_threshold
            )
            affinity[i, j] = affinity[j, i] = 1e9 if is_noise else 1.0 / length

        laplacian = np.diag(affinity.sum(axis=1)) - affinity
        laplacian_pinv = pinvh(laplacian)
        diagonal = np.diag(laplacian_pinv)
        green_distance = diagonal[:, None] + diagonal[None, :] - 2.0 * laplacian_pinv
        eigenvalues = np.sort(np.abs(eigh(green_distance, eigvals_only=True)))[::-1]

        result = np.zeros(self.parameters.top_k, dtype=np.float64)
        count = min(self.parameters.top_k, len(eigenvalues))
        result[:count] = eigenvalues[:count]
        return result

    @staticmethod
    def _build_skeleton(
        tree: SWCTree,
    ) -> tuple[nx.Graph, dict[int, NDArray[np.float64]], int]:
        graph = nx.Graph()
        coordinates: dict[int, NDArray[np.float64]] = {}
        pending_edges: list[tuple[int, int]] = []
        root: int | None = None

        for index, node_value in enumerate(tree.node_ids):
            node = int(node_value)
            parent = int(tree.parent_ids[index])
            coordinates[node] = np.asarray(tree.coordinates[index], dtype=np.float64)
            graph.add_node(node)
            if parent == -1:
                # The notebook overwrote root for each -1 row, so the last
                # explicit root wins when malformed files contain more than one.
                root = node
            else:
                pending_edges.append((node, parent))

        if root is None:
            root = next(iter(graph.nodes()))
        for node, parent in pending_edges:
            if node in coordinates and parent in coordinates:
                length = max(float(np.linalg.norm(coordinates[node] - coordinates[parent])), 1e-5)
                graph.add_edge(node, parent, weight=length)

        largest_component = max(nx.connected_components(graph), key=len)
        graph = graph.subgraph(largest_component).copy()
        if root not in graph:
            root = next(iter(graph.nodes()))

        degree_two = [node for node in graph if graph.degree(node) == 2 and node != root]
        for node in degree_two:
            if node not in graph or graph.degree(node) != 2:
                continue
            left, right = list(graph.neighbors(node))
            length = graph[left][node]["weight"] + graph[node][right]["weight"]
            if graph.has_edge(left, right):
                graph[left][right]["weight"] = min(graph[left][right]["weight"], length)
            else:
                graph.add_edge(left, right, weight=length)
            graph.remove_node(node)
        return graph, coordinates, root


register_spectrum_method(
    ArakelovGreenSpectrum.method,
    lambda parameters: ArakelovGreenSpectrum(parameters),
)
