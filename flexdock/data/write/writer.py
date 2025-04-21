from pathlib import Path
import pickle
from lightning.pytorch.callbacks import BasePredictionWriter


class FlexDockWriter(BasePredictionWriter):
    def __init__(self, args, output_dir, write_interval="batch"):
        super().__init__(write_interval)
        self.args = args
        if isinstance(output_dir, str):
            output_dir = Path(output_dir)
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True, parents=True)

    def write_on_batch_end(
        self,
        trainer,
        pl_module,
        prediction,
        batch_indices,
        batch,
        batch_idx,
        dataloader_idx,
    ):
        assert batch.num_graphs == 1, "We only allow batch_size=1 currently"

        if not prediction:
            return

        if isinstance(prediction, dict) and len(prediction) == 0:
            return

        complex_id = batch["name"][0]
        out_dir = self.output_dir / complex_id
        out_dir.mkdir(exist_ok=True, parents=True)

        if "docking" in prediction:
            docking_outputs = prediction["docking"]
            with (out_dir / "docking_predictions.pkl").open("wb") as f:
                pickle.dump(docking_outputs, f)

        if "relaxation" in prediction:
            relax_outputs = prediction["relaxation"]
            with (out_dir / "relax_predictions.pkl").open("wb") as f:
                pickle.dump(relax_outputs, f)
