import torch
import torch.utils.data


class TensorSensingDataset(torch.utils.data.Dataset):
    """
    Supports tensor completion and rank 1 tensor sensing tasks. Inputs are stored as matrices of shape (N, d), where N is the order of the tensor and
    d is the dimension of the modes. When each column is a one-hot encoded vectors then the corresponding task is tensor completion. For arbitrary
    encodings we get rank 1 tensor sensing.
    """

    def __init__(self, inputs: torch.Tensor, targets: torch.Tensor, additional_metadata: dict = None):
        self.inputs = inputs.detach()
        self.targets = targets.detach()

        if self.inputs.shape[0] != len(self.targets):
            raise ValueError(f"Mismatch in number of inputs and targets. "
                             f"Recieved {self.inputs.shape[0]} inputs and {len(self.targets)} targets")

        self.target_tensor_modes_dim = self.inputs.shape[2]
        self.additional_metadata = additional_metadata if additional_metadata is not None else {}
        self.target_tensor_order = self.inputs.shape[1]

    def __getitem__(self, index: int):
        return self.inputs[index], self.targets[index]

    def __len__(self) -> int:
        return len(self.targets)

    def to_device(self, device):
        self.inputs = self.inputs.to(device)
        self.targets = self.targets.to(device)

    def save(self, path: str):
        state_dict = {
            "inputs": self.inputs.cpu().detach(),
            "targets": self.targets.cpu().detach(),
            "additional_metadata": self.additional_metadata
        }
        torch.save(state_dict, path)

    @staticmethod
    def load(path: str, device=torch.device("cpu")):
        state_dict = torch.load(path, map_location=device)

        additional_metadata = state_dict["additional_metadata"] if "additional_metadata" in state_dict else {}
        return TensorSensingDataset(inputs=state_dict["inputs"],
                                    targets=state_dict["targets"],
                                    additional_metadata=additional_metadata)
