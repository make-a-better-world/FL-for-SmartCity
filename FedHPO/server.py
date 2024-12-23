import argparse
import os
import csv
from datetime import date, datetime
from tabnanny import verbose
from mak.data.mnist import MnistData
from mak.model.models import MNISTCNN, SimpleCNN, SimpleDNN, KerasExpCNN
from mak.custom_server import ServerSaveData
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import tensorflow as tf
import flwr as fl
from mak.data.fashion_mnist import FashionMnistData
import yaml
from mak.utils import generate_config_server,gen_dir_outfile_server, set_seed

from mak.custom_strategy.fedprox import FedProx



def get_eval_fn(model,dataset):
    """Return an evaluation function for server-side evaluation."""
    # Load data and model here to avoid the overhead of doing it in `evaluate` itself
    # load last 5k samples for testing
    if dataset =='mnist':
        (x_val, y_val) = MnistData(num_clients=10).load_test_data()
    else:
        (x_val, y_val) = FashionMnistData(num_clients=10).load_test_data()
    print("Validation x shape : {}".format(x_val.shape))

    # The `evaluate` function will be called after every round
    def evaluate(
        weights: fl.common.Weights,
    ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
        model.set_weights(weights)  # Update model with the latest parameters
        loss, accuracy = model.evaluate(x_val, y_val,verbose=1)
        model.save('./saved_model')
        print("Accuracy {} ".format(accuracy))
        return loss, {"accuracy": accuracy}

    return evaluate

def fit_config(server_round: int):
    """Return training configuration dict for each round.

    passes the current round number to the client
    """
    config = {
        "round": server_round,
    }
    return config

def main() -> None:
    set_seed(13)
    input_shape = (28, 28, 1)
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument("--config",type=str,default = "config.yaml")
    args = parser.parse_args()
    server_config = generate_config_server(args)
    out_file_path = gen_dir_outfile_server(config=server_config)

    dataset = server_config['dataset']
    if dataset == 'mnist':
        model = MNISTCNN(input_shape=input_shape, num_classes=10)._model
    else:
        model = KerasExpCNN(input_shape=input_shape, num_classes=10)._model
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.categorical_crossentropy,
        metrics=["accuracy"],
        run_eagerly=True,
    )

    # Create strategy
    if server_config['strategy'] == "fedyogi":
        strategy = fl.server.strategy.FedYogi(
            fraction_fit=0.3,
            fraction_eval=0.2,
            min_fit_clients=server_config['min_fit_clients'],
            min_eval_clients=1,
            min_available_clients=server_config['min_avalaible_clients'],
            eval_fn=get_eval_fn(model,dataset),
            on_fit_config_fn=fit_config,
            initial_parameters=fl.common.weights_to_parameters(
                model.get_weights()),
            eta=1e-2,
            eta_l=0.0316,
            beta_1=0.9,
            beta_2=0.99,
            tau=1e-3,
        )
    elif server_config['strategy'] == "fedadagrad":
        strategy = fl.server.strategy.FedAdagrad(
            fraction_fit=0.3,
            fraction_eval=0.2,
            min_fit_clients=server_config['min_fit_clients'],
            min_eval_clients=1,
            min_available_clients=server_config['min_avalaible_clients'],
            eval_fn=get_eval_fn(model,dataset),
            on_fit_config_fn=fit_config,
            initial_parameters=fl.common.weights_to_parameters(
                model.get_weights()),
            eta=1e-2,
            eta_l=0.0316,
            tau=1e-3,
        )
    elif server_config['strategy'] == "fedavgm":
        strategy = fl.server.strategy.FedAvgM(
            fraction_fit=0.3,
            fraction_eval=0.2,
            min_fit_clients=server_config['min_fit_clients'],
            min_eval_clients=1,
            min_available_clients=server_config['min_avalaible_clients'],
            eval_fn=get_eval_fn(model,dataset),
            on_fit_config_fn=fit_config,
            initial_parameters=fl.common.weights_to_parameters(
                model.get_weights()),
            server_learning_rate=1.0,
            server_momentum=0.2,
        )
    elif server_config['strategy'] == "fedprox": #from flwr 1.XX
        strategy = FedProx(
            fraction_fit=0.3,
            fraction_eval=0.2,
            min_fit_clients=server_config['min_fit_clients'],
            min_eval_clients=1,
            min_available_clients=server_config['min_avalaible_clients'],
            eval_fn=get_eval_fn(model,dataset),
            on_fit_config_fn=fit_config,
            proximal_mu = 0.5,
            initial_parameters=fl.common.weights_to_parameters(
                model.get_weights()),
        )
    else:
        strategy = fl.server.strategy.FedAvg(
            fraction_fit=0.3,
            fraction_eval=0.2,
            min_fit_clients=server_config['min_fit_clients'],
            min_eval_clients=1,
            min_available_clients=server_config['min_avalaible_clients'],
            eval_fn=get_eval_fn(model,dataset),
            on_fit_config_fn=fit_config,
            initial_parameters=fl.common.weights_to_parameters(
                model.get_weights()),
        )

    print(f"Using Strategy : {strategy.__class__}")

  # Start Flower server for four rounds of federated learning
    server = ServerSaveData(
        strategy=strategy, client_manager=fl.server.client_manager.SimpleClientManager(),out_file_path=out_file_path,target_acc=server_config['target_acc'])
    fl.server.start_server(
        server=server,
        server_address="[::]:8080",
        config={"num_rounds": server_config['max_rounds']},
        strategy=strategy
    )


if __name__ == "__main__":
    main()
