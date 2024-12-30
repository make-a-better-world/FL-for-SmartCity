##################################################################################
#  SmartCityStrategy Class
#  Server aggregation 을 위한 class
#  2024. 11. eby
##################################################################################
from typing import Callable, Union, Optional,List, Tuple, Dict

from flwr.common import EvaluateIns,EvaluateRes,FitIns, FitRes, MetricsAggregationFn, NDArrays, Parameters, Scalar, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg
import flwr as fl
from timeit import default_timer as timer
import numpy as np
import io


class ScityStrategy(fl.server.strategy.Strategy):
    
    def __init__(
        self,
        sCity_conf
        ) -> None:
        super().__init__()
        self.sCity_conf = sCity_conf
        self.learningConfig = sCity_conf.get_learningParameters()
        self.flRound = int(sCity_conf.get_flRound()) 
        print(">>>> learning config ==> ", self.learningConfig)
     #   self.reafConfigFile(configFile)
     #   self.model = model
     

    def __repr__(self) -> str:
        return "ScityStrategy"

    #@override
    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        """Initialize global model parameters."""
    ###  net = Net()
    #    net = self.model
    #    ndarrays = get_parameters(net)
    #    return fl.common.ndarrays_to_parameters(ndarrays)
    

    def fit_config(self, server_round: int):
        """Return training configuration dict for each round."""
        print("inside fit_config(). server round: ", server_round, ", batch_size : ", self.sCity_conf.get_batchSize())
        config = {
            "batch_size": self.learningConfig['batch_size'], # str key, int value
            "dropout": self.learningConfig['dropout'],  # str key, bool value
            "learning_rate": self.learningConfig['learningrate'],  # str key, float value
            "optimizer": self.learningConfig['optimizer'],  # str key, float value
            "current_round": server_round,
            "local_epochs": 1 if server_round < 2 else 2, ##To-Do : check
        }
        return config
 
     
    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""

        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        
        clients = client_manager.sample(
            num_clients = sample_size, 
            min_num_clients = min_num_clients
        )
        
        print(">>eby>> sample_size, min_num_clients :" ,sample_size, min_num_clients)

        # Create custom configs
        n_clients = len(clients)
        '''
        fit_configurations = []
        for idx, client in enumerate(clients):
            if idx < half_clients:
                fit_configurations.append((client, FitIns(parameters, standard_config)))
            else:
                fit_configurations.append(
                    (client, FitIns(parameters, higher_lr_config))
                )
        '''
        fit_configurations = []
#       fit_configurations = fit_config
        #standard_config = self.config 
        standard_config = self.fit_config(server_round) 
        for idx, client in enumerate(clients):
            fit_configurations.append((client, FitIns(parameters, standard_config)))
       
        return fit_configurations


    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average for every FL round."""
        
        if len(results) == 0:
            return None, {}

        weights_results = []
        total_weight = 0.0  # 전체 가중치의 합
        alpha = 0.7  # Loss와 Accuracy의 조합 비율

        for _, fit_res in results:
            try:
                # 클라이언트의 파라미터와 샘플 개수 가져오기
                client_ndarrays = parameters_to_ndarrays(fit_res.parameters)
                num_examples = fit_res.num_examples
                
                # Loss와 Accuracy 기반 가중치 계산 (num_examples를 곱해 데이터 크기 고려)
                loss = fit_res.metrics["loss"]
                accuracy = fit_res.metrics["accuracy"]
                combined_weight = num_examples * (alpha * (1 / (loss + 1e-8)) + (1 - alpha) * accuracy)
                weights_results.append((client_ndarrays, combined_weight))
                total_weight += combined_weight
            except Exception as e:
                print(f"Error processing client parameters: {e}")
                continue

        # 가중치를 정규화하고 파라미터 집계
        normalized_weights = [(params, weight / total_weight) for params, weight in weights_results]
        aggregated_parameters = ndarrays_to_parameters(aggregate(normalized_weights))

        # Metrics 집계
        aggregated_metrics = {
            "accuracy": np.mean([fit_res.metrics["accuracy"] for _, fit_res in results]),
            "loss": np.mean([fit_res.metrics["loss"] for _, fit_res in results]),
        }

        # 디버깅용 출력
        if aggregated_parameters is not None:
            aggregated_ndarrays: List[np.ndarray] = fl.common.parameters_to_ndarrays(aggregated_parameters)
            print(f"Saving round {server_round} aggregated_ndarrays...")
            np.savez(f"round-{server_round}-weights.npz", *aggregated_ndarrays)
            for idx, param in enumerate(aggregated_ndarrays):
                print(f"Parameter {idx} shape: {param.shape}")
                if param.ndim > 0:
                    print(f"Aggregated Parameter {idx}: {param[:10]}")
                else:
                    print(f"Aggregated Parameter {idx}: {param}")

        print("Metrics aggregated:", aggregated_metrics)
        return aggregated_parameters, aggregated_metrics




    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""
        if float(self.sCity_conf.get_fractionEvaluate()) == 0.0:
            return []
        config = {}
        evaluate_ins = EvaluateIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_evaluation_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Return client/config pairs
        return [(client, evaluate_ins) for client in clients]


    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation losses using weighted average."""
        
        if not results:
            return None, {}

        loss_aggregated = weighted_loss_avg(
            [
                (evaluate_res.num_examples, evaluate_res.loss)
                for _, evaluate_res in results
            ]
        )
        metrics_aggregated = {}
        return loss_aggregated, metrics_aggregated

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate global model parameters using an evaluation function."""

        # Let's assume we won't perform the global model evaluation on the server side.
        return None

    def num_fit_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Return sample size and required number of clients."""
        num_clients = int(num_available_clients * float(self.sCity_conf.get_fractionFit()))
        print("num_available_clients :", num_available_clients)
        #  return max(num_clients, min_fit_clients), min_available_clients
        return max(num_clients, int(self.sCity_conf.get_minClients())),  int(self.sCity_conf.get_minClients())

    def num_evaluation_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Use a fraction of available clients for evaluation."""
        num_clients = int(num_available_clients * float(self.sCity_conf.get_fractionEvaluate()))
        #  return max(num_clients, min_evaluate_clients), min_available_clients
        return max(num_clients,  int(self.sCity_conf.get_minClients())),  int(self.sCity_conf.get_minClients()) 
     
