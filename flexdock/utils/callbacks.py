from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor


def setup_docking_callbacks(args, run_dir):
    best_model_checkpoint = ModelCheckpoint(
        dirpath=run_dir,
        filename="best_model",
        monitor="val_loss",
        mode="min",
        every_n_epochs=1,
        save_on_train_epoch_end=True,
        save_top_k=1,
    )
    # By default this saves as ".ckpt"
    best_model_checkpoint.FILE_EXTENSION = ".pt"
    callbacks = [best_model_checkpoint]

    last_model_checkpoint = ModelCheckpoint(
        dirpath=run_dir,
        filename="last_model",
        monitor=None,
        every_n_epochs=1,
        save_on_train_epoch_end=True,
        save_top_k=1,
    )
    last_model_checkpoint.FILE_EXTENSION = ".pt"
    callbacks.append(last_model_checkpoint)

    if args.val_inference_freq is not None:
        for metric in args.inference_earlystop_metric.split(","):
            best_inf_checkpoint = ModelCheckpoint(
                dirpath=run_dir,
                filename=f"best_inference_epoch_model_{metric}",
                monitor=metric,
                mode=args.inference_earlystop_goal,
                every_n_epochs=args.val_inference_freq,
                save_on_train_epoch_end=True,
                save_top_k=1,
            )
            best_inf_checkpoint.FILE_EXTENSION = ".pt"
            callbacks.append(best_inf_checkpoint)

        if args.flexible_sidechains:
            best_sc_checkpoint = ModelCheckpoint(
                dirpath=run_dir,
                filename="best_inference_epoch_model_aa",
                monitor="valinf_aa_rmsds_lt1",
                mode="max",
                every_n_epochs=args.val_inference_freq,
                save_on_train_epoch_end=True,
                save_top_k=1,
            )
            best_sc_checkpoint.FILE_EXTENSION = ".pt"
            callbacks.append(best_sc_checkpoint)

        if args.flexible_backbone:
            best_bb_checkpoint = ModelCheckpoint(
                dirpath=run_dir,
                filename="best_inference_epoch_model_bb",
                monitor="valinf_bb_rmsds_lt1",
                mode="max",
                every_n_epochs=args.val_inference_freq,
                save_on_train_epoch_end=True,
                save_top_k=1,
            )
            best_bb_checkpoint.FILE_EXTENSION = ".pt"
            callbacks.append(best_bb_checkpoint)

    if args.wandb:
        lr_monitor = LearningRateMonitor(logging_interval="epoch")
        callbacks.append(lr_monitor)
    return callbacks


def setup_filtering_callbacks(args, run_dir):
    best_loss_checkpoint = ModelCheckpoint(
        dirpath=run_dir,
        filename="best_loss",
        monitor="val_filtering_loss",
        mode="min",
        every_n_epochs=1,
        save_on_train_epoch_end=True,
        save_top_k=1,
    )
    best_loss_checkpoint.FILE_EXTENSION = ".pt"

    best_model_checkpoint = ModelCheckpoint(
        dirpath=run_dir,
        filename="best_model",
        monitor="val_accuracy",
        mode="max",
        every_n_epochs=1,
        save_on_train_epoch_end=True,
        save_top_k=1,
    )
    # By default this saves as ".ckpt"
    best_model_checkpoint.FILE_EXTENSION = ".pt"
    callbacks = [best_loss_checkpoint, best_model_checkpoint]

    if args.atom_lig_confidence:
        best_atom_loss_checkpoint = ModelCheckpoint(
            dirpath=run_dir,
            filename="best_atom_loss",
            monitor="val_atom_filtering_loss",
            mode="min",
            every_n_epochs=1,
            save_on_train_epoch_end=True,
            save_top_k=1,
        )
        best_atom_loss_checkpoint.FILE_EXTENSION = ".pt"

        best_atom_model_checkpoint = ModelCheckpoint(
            dirpath=run_dir,
            filename="best_atom_model",
            monitor="val_atom_accuracy",
            mode="max",
            every_n_epochs=1,
            save_on_train_epoch_end=True,
            save_top_k=1,
        )
        best_atom_model_checkpoint.FILE_EXTENSION = ".pt"
        callbacks.extend([best_atom_loss_checkpoint, best_atom_model_checkpoint])

    if args.wandb:
        lr_monitor = LearningRateMonitor(logging_interval="epoch")
        callbacks.append(lr_monitor)
    return callbacks


def setup_relaxation_callbacks(args, run_dir):
    callbacks = []
    best_model_checkpoint = ModelCheckpoint(
        dirpath=run_dir,
        filename="best_model",
        monitor=args.main_metric,
        mode=args.main_metric_goal,
        every_n_epochs=1,
        save_on_train_epoch_end=True,
        save_top_k=1,
    )
    best_model_checkpoint.FILE_EXTENSION = ".pt"
    callbacks.append(best_model_checkpoint)

    last_model_checkpoint = ModelCheckpoint(
        dirpath=run_dir,
        filename="last_model",
        monitor=None,
        every_n_epochs=1,
        save_on_train_epoch_end=True,
        save_top_k=1,
    )
    last_model_checkpoint.FILE_EXTENSION = ".pt"
    callbacks.append(last_model_checkpoint)

    if args.val_inference_freq is not None:
        best_inf_checkpoint = ModelCheckpoint(
            dirpath=run_dir,
            filename="best_inference_epoch_model",
            monitor=args.inference_earlystop_metric,
            mode=args.inference_earlystop_goal,
            every_n_epochs=args.val_inference_freq,
            save_on_train_epoch_end=True,
            save_top_k=1,
        )
        best_inf_checkpoint.FILE_EXTENSION = ".pt"
        callbacks.append(best_inf_checkpoint)

    if args.wandb:
        lr_monitor = LearningRateMonitor(logging_interval="epoch")
        callbacks.append(lr_monitor)
    return callbacks


def setup_callbacks(args, run_dir, task: str = "docking"):
    if task == "docking":
        return setup_docking_callbacks(args=args, run_dir=run_dir)

    elif task == "relaxation":
        return setup_relaxation_callbacks(args=args, run_dir=run_dir)

    else:
        assert task == "filtering"
        return setup_filtering_callbacks(args=args, run_dir=run_dir)
