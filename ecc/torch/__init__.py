import torch

device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LABEL_TOTAL_ACCURACY: str = "Ø"
LABEL_QA1: str = "QA1"
LABEL_QA2: str = "QA2"
LABEL_QA3: str = "QA3"
