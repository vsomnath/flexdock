import torch
import torch.nn as nn


class AtomEncoder(nn.Module):
    def __init__(self, emb_dim, feature_dims, sigma_embed_dim, lm_embedding_type=None):
        # first element of feature_dims tuple is a list with the lenght of each categorical feature and the second is the number of scalar features
        super(AtomEncoder, self).__init__()
        self.atom_embedding_list = torch.nn.ModuleList()
        self.num_categorical_features = len(feature_dims[0])
        if lm_embedding_type is not None:
            if lm_embedding_type == "esm":
                lm_embedding_dim = 1280
            else:
                raise ValueError(
                    "LM Embedding type was not correctly determined. LM embedding type: ",
                    lm_embedding_type,
                )
        else:
            lm_embedding_dim = 0
        self.additional_features_dim = (
            feature_dims[1] + sigma_embed_dim + lm_embedding_dim
        )
        for i, dim in enumerate(feature_dims[0]):
            emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.atom_embedding_list.append(emb)

        if self.additional_features_dim > 0:
            self.additional_features_embedder = torch.nn.Linear(
                self.additional_features_dim + emb_dim, emb_dim
            )

    def forward(self, x):
        x_embedding = 0
        assert (
            x.shape[1] == self.num_categorical_features + self.additional_features_dim
        )
        for i in range(self.num_categorical_features):
            x_embedding += self.atom_embedding_list[i](x[:, i].long())

        if self.additional_features_dim > 0:
            x_embedding = self.additional_features_embedder(
                torch.cat([x_embedding, x[:, self.num_categorical_features :]], axis=1)
            )
        return x_embedding


class GaussianSmearing(nn.Module):
    # used to embed the edge distances
    def __init__(self, start=0.0, stop=5.0, num_gaussians=50):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.register_buffer("offset", offset)

    def forward(self, dist):
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))
