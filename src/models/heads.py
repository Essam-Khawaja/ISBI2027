from torch import nn


class PredictionHeads(nn.Module):
    def __init__(self, input_dim, task_specs):
        super().__init__()

        self.heads = nn.ModuleDict({
            task["name"]: nn.Linear(input_dim, task["output_dim"])
            for task in task_specs
        })

    def forward(self, embedding):
        # embedding: [batch, embedding_dim]
        return {
            task_name: head(embedding)
            for task_name, head in self.heads.items()
        }

