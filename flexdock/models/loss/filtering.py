import torch.nn as nn
import torch.nn.functional as F


class FilteringLoss(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

    def forward(self, outputs, batch, apply_mean: bool = True):
        loss_dict = {}

        predictions = outputs["filtering_pred"]

        if self.args.rmsd_prediction:
            labels = batch.rmsd
            filtering_loss = F.mse_loss(predictions, labels)
        else:
            if isinstance(self.args.rmsd_classification_cutoff, list):
                labels = batch.y_binned
                filtering_loss = F.cross_entropy(predictions, labels)
            else:
                labels = batch.y
                filtering_loss = F.binary_cross_entropy_with_logits(predictions, labels)

        if apply_mean:
            filtering_loss = filtering_loss.mean()

        if self.args.atom_lig_confidence:
            atom_predictions = outputs["filtering_atom_pred"]
            atom_labels = batch.y_aa
            atom_filtering_loss = F.binary_cross_entropy_with_logits(
                atom_predictions, atom_labels
            )
            if apply_mean:
                atom_filtering_loss = atom_filtering_loss.mean()

        loss = self.args.filtering_weight * filtering_loss
        if self.args.atom_lig_confidence:
            loss += self.args.filtering_weight_atom * atom_filtering_loss

        loss_dict["filtering_loss"] = filtering_loss.detach().clone()
        if self.args.atom_lig_confidence:
            loss_dict["atom_filtering_loss"] = atom_filtering_loss.detach().clone()
        loss_dict["loss"] = loss.detach().clone()

        return loss, loss_dict
