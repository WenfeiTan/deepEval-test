# data/dataset.py
from deepeval.dataset import EvaluationDataset, Golden

dataset = EvaluationDataset(
    goldens=[
        Golden(input="帮我查一下今天上海的天气"),
        Golden(input="明天北京的天气怎么样？要不要带伞？"),
    ]
)
