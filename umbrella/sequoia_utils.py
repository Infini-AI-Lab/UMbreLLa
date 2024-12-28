import torch
import json
from collections import deque
import matplotlib.pyplot as plt
import networkx as nx
DEFAULT_ACC = [0.65, 0.2, 0.1, 0.05]

def save_pt_to_json(pt_file, json_file):
    """
    Convert a .pt file to a JSON file.

    Parameters:
        pt_file (str): Path to the input .pt file.
        json_file (str): Path to the output JSON file.
    """
    # 加载 .pt 文件
    data = torch.load(pt_file)

    # 将 tensor 转换为可序列化的列表或数字
    def tensor_to_serializable(obj):
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        elif isinstance(obj, list):
            return [tensor_to_serializable(o) for o in obj]
        elif isinstance(obj, dict):
            return {key: tensor_to_serializable(value) for key, value in obj.items()}
        else:
            return obj

    # 转换数据为 JSON 可序列化格式
    serializable_data = tensor_to_serializable(data)

    # 保存为 JSON 文件
    with open(json_file, "w") as f:
        json.dump(serializable_data, f, indent=4)

    print(f"Data saved to {json_file}")

def successor_list_to_mask(successor_list):
    """
    Generate an n x n mask matrix for a given successor list.

    Parameters:
    successor_list (list of list of int): The successor list of a directed tree.

    Returns:
    list of list of int: The mask matrix where the i-th row contains 1s for all predecessors of i (including i itself), and 0s otherwise.
    """
    n = len(successor_list)  # Number of nodes
    mask = [[0] * n for _ in range(n)]

    # Perform a reverse topological sort to compute predecessors for each node
    predecessors = [set() for _ in range(n)]  # Predecessors of each node
    reverse_graph = [[] for _ in range(n)]

    # Build the reverse graph
    for i, successors in enumerate(successor_list):
        for succ in successors:
            reverse_graph[succ].append(i)

    # Use BFS/DFS to find all predecessors for each node
    for node in range(n):
        visited = set()
        queue = deque([node])

        while queue:
            current = queue.popleft()
            if current not in visited:
                visited.add(current)
                predecessors[node].add(current)
                queue.extend(reverse_graph[current])

    # Fill the mask matrix
    for i in range(n):
        for pred in predecessors[i]:
            mask[i][pred] = 1

    return mask



def generate_sequoia_tree(width: int, depth: int, acc: list[float] = None, json_file=None):
    
    if acc is None:
        assert width <= 4, "Using default acceptance rate vector, require width<=4"
        acc = DEFAULT_ACC
    
    acc : torch.Tensor = torch.Tensor(acc)
    acc = torch.log(acc)
    num_beams = len(acc)
    size = width * depth + 1
    root = [[0]]
    score = [[0]]
    Successor = [[]]
    branches = [[0]]
    tree_depth = [0]
    for i in range(depth):
        root.append([j for j in range(i * width + 1, (i+1) * width + 1)])
        branches.append([0 for _ in range(width)])
        tree_depth.extend([i+1 for _ in range(width)])
        Successor.extend([[] for _ in range(width)])
        current_score = torch.Tensor(score[i]).repeat_interleave(num_beams)
        candidates_score = acc.repeat(1 if i == 0 else width) + current_score
        
        selected_candidate_score, selected_candidate_indices = candidates_score.topk(k=width)
        score.append(selected_candidate_score)
        offset = 0 if i == 0 else (i - 1) * width + 1
        selected_candidate_parents = selected_candidate_indices // num_beams + offset
        for child, parent in enumerate(sorted(selected_candidate_parents.tolist())):
            Successor[parent].append(child + i * width + 1)
            branches[i][parent - offset] += 1
    
    mask = successor_list_to_mask(Successor)

    result = {
        "roots": root,
        "branches": branches,
        "Successors": Successor,
        "mask": mask,
        "depth": tree_depth,
        "size": size
    }

    # Save to JSON file
    json_file = "./trees/sequoia_tree-{}x{}.json".format(width, depth) if json_file is None else json_file
    with open(json_file, "w") as f:
        json.dump(result, f, indent=4)

    return result
            
generate_sequoia_tree(width=5, depth=6, acc=[0.65, 0.2, 0.06, 0.03, 0.02])