import torch
from torch import nn

from src.models.encoders import Conv3DEncoder, MLPEncoder
from src.models.heads import PredictionHeads


class ClinicalOnlyModel(nn.Module):
    def __init__(self, tabular_dim, config, task_specs):
        super().__init__()

        model_config = config.get("model", {})
        embedding_dim = int(model_config.get("tabular_embedding_dim", 32))

        self.tabular_encoder = MLPEncoder(
            input_dim=tabular_dim,
            hidden_dims=model_config.get("tabular_hidden_dims", [64, 32]),
            embedding_dim=embedding_dim,
            dropout=float(model_config.get("dropout", 0.2)),
        )
        self.heads = PredictionHeads(embedding_dim, task_specs)

    def forward(self, image=None, tabular=None):
        if tabular is None:
            raise ValueError("ClinicalOnlyModel requires tabular input.")

        z_tabular = self.tabular_encoder(tabular)
        # z_tabular: [batch, tabular_embedding_dim]

        return self.heads(z_tabular)


class ImagingOnlyModel(nn.Module):
    def __init__(self, image_channels, config, task_specs):
        super().__init__()

        model_config = config.get("model", {})
        embedding_dim = int(model_config.get("image_embedding_dim", 64))

        self.image_encoder = Conv3DEncoder(
            in_channels=image_channels,
            channels=model_config.get("image_channels", [8, 16, 32]),
            embedding_dim=embedding_dim,
            dropout=float(model_config.get("dropout", 0.2)),
        )
        self.heads = PredictionHeads(embedding_dim, task_specs)

    def forward(self, image=None, tabular=None):
        if image is None:
            raise ValueError("ImagingOnlyModel requires image input.")

        z_image = self.image_encoder(image)
        # z_image: [batch, image_embedding_dim]

        return self.heads(z_image)


class FusionPetCtTabularModel(nn.Module):
    def __init__(self, image_channels, tabular_dim, config, task_specs):
        super().__init__()

        model_config = config.get("model", {})
        image_embedding_dim = int(model_config.get("image_embedding_dim", 64))
        tabular_embedding_dim = int(model_config.get("tabular_embedding_dim", 32))
        fusion_hidden_dims = model_config.get("fusion_hidden_dims", [64])
        dropout = float(model_config.get("dropout", 0.2))

        self.image_encoder = Conv3DEncoder(
            in_channels=image_channels,
            channels=model_config.get("image_channels", [8, 16, 32]),
            embedding_dim=image_embedding_dim,
            dropout=dropout,
        )
        self.tabular_encoder = MLPEncoder(
            input_dim=tabular_dim,
            hidden_dims=model_config.get("tabular_hidden_dims", [64, 32]),
            embedding_dim=tabular_embedding_dim,
            dropout=dropout,
        )

        fusion_layers = []
        previous_dim = image_embedding_dim + tabular_embedding_dim

        for hidden_dim in fusion_hidden_dims:
            fusion_layers.append(nn.Linear(previous_dim, hidden_dim))
            fusion_layers.append(nn.ReLU())
            fusion_layers.append(nn.Dropout(dropout))
            previous_dim = hidden_dim

        self.fusion_mlp = nn.Sequential(*fusion_layers)
        self.heads = PredictionHeads(previous_dim, task_specs)

    def forward(self, image=None, tabular=None):
        if image is None or tabular is None:
            raise ValueError("FusionPetCtTabularModel requires image and tabular inputs.")

        z_image = self.image_encoder(image)
        # z_image: [batch, image_embedding_dim]

        z_tabular = self.tabular_encoder(tabular)
        # z_tabular: [batch, tabular_embedding_dim]

        fused = torch.cat([z_image, z_tabular], dim=1)
        # fused: [batch, image_embedding_dim + tabular_embedding_dim]

        fused = self.fusion_mlp(fused)
        # fused after MLP: [batch, final_fusion_dim]

        return self.heads(fused)


def build_model(config, tabular_dim, task_specs):
    model_name = config.get("model", {}).get("name", "clinical_only")
    image_channels = len(config.get("image", {}).get("modalities", ["ct", "pet"]))

    if model_name == "clinical_only":
        if tabular_dim <= 0:
            raise ValueError("clinical_only requires at least one tabular feature.")

        return ClinicalOnlyModel(
            tabular_dim=tabular_dim,
            config=config,
            task_specs=task_specs,
        )

    if model_name == "imaging_petct":
        return ImagingOnlyModel(
            image_channels=image_channels,
            config=config,
            task_specs=task_specs,
        )

    if model_name == "fusion_petct_tabular":
        if tabular_dim <= 0:
            raise ValueError("fusion_petct_tabular requires tabular features.")

        return FusionPetCtTabularModel(
            image_channels=image_channels,
            tabular_dim=tabular_dim,
            config=config,
            task_specs=task_specs,
        )

    raise ValueError(f"Unknown model.name: {model_name}")

