import torch
import json
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
