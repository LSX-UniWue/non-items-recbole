import argparse
from pathlib import Path

import torch.distributed as dist
from logging import getLogger

from recbole.data import (
    create_dataset,
    data_preparation,
)
from recbole.utils import (
    init_logger,
    get_model,
    get_trainer,
    init_seed,
    set_color,
    get_environment,
)

def run_recbole(
    model_file,
    device='gpu',
    write_predictions=None,):
    r"""A fast running api, which includes the complete process of
    training and testing a model on a specified dataset

    Args:
        model (str, optional): Model name. Defaults to ``None``.
        dataset (str, optional): Dataset name. Defaults to ``None``.
        config_file_list (list, optional): Config files used to modify experiment parameters. Defaults to ``None``.
        config_dict (dict, optional): Parameters dictionary used to modify experiment parameters. Defaults to ``None``.
        saved (bool, optional): Whether to save the model. Defaults to ``True``.
        queue (torch.multiprocessing.Queue, optional): The queue used to pass the result to the main process. Defaults to ``None``.
    """
    import torch

    model_file = Path(model_file)
    checkpoint = torch.load(model_file, map_location=torch.device(device))
    config = checkpoint["config"]

    init_seed(config["seed"], config["reproducibility"])
    init_logger(config)
    logger = getLogger()
    logger.info(config)

    dataset = create_dataset(config)
    logger.info(dataset)
    train_data, valid_data, test_data = data_preparation(config, dataset)

    init_seed(config["seed"], config["reproducibility"])
    model = get_model(config["model"])(config, train_data._dataset).to(config["device"])
    model.load_state_dict(checkpoint["state_dict"])
    model.load_other_parameter(checkpoint.get("other_parameter"))
    logger.info(model)
    # trainer loading and initialization
    trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)

    # model evaluation
    test_result = trainer.evaluate(
        test_data, load_best_model=False, show_progress=config["show_progress"], write_predictions=write_predictions
    )

    logger.info(test_result)
    environment_tb = get_environment(config)
    logger.info(
        "The running environment of this training is as follows:\n"
        + environment_tb.draw()
    )

    logger.info(set_color("test result", "yellow") + f": {test_result}")

    result = {
        "test_result": test_result,
    }

    if not config["single_spec"]:
        dist.destroy_process_group()

    return result  # for the single process

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_file", "-m", type=str, default=None, help="saved model")
    parser.add_argument("--device", type=str, default='cuda', help="device")
    parser.add_argument("--write_predictions", default=None, help="path to pred file")
    args, _ = parser.parse_known_args()

    res = run_recbole(args.model_file, args.device, args.write_predictions)
