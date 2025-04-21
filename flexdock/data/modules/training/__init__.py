from flexdock.data.modules.training import docking


def setup_training_datamodule(data_cfg, transform_cfg, device: str = "cpu"):
    if data_cfg.task == "docking":
        docking_data_cfg = docking.DockingDataConfig.from_dict(data_cfg)

        return docking.DockingDataModule(
            data_cfg=docking_data_cfg, transform_cfg=transform_cfg
        )

    # elif args.task == "filtering":
    #     return filtering.FilteringDataModule(args=args, device=device)

    # else:
    #     return relaxation.RelaxationDataModule(args=args, device=device)
