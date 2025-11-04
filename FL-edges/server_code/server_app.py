import argparse
from server_code.server_app import main
from config_loader import load_config
import flwr as fl

def main():
    parser = argparse.ArgumentParser(description="Flower Server")
    parser.add_argument("--config", type=str, default="config/server_config.yml",
                        help="Path to server config YAML")
    args = parser.parse_args()

    config = load_config(args.config)
    server_conf = config.get("server", {})
    strategy_conf = server_conf.get("strategy", {})

    # 예시: FedAvg 전략 설정
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=strategy_conf.get("fraction_fit", 0.5),
        min_available_clients=strategy_conf.get("min_available_clients", 1),
        evaluate_every=strategy_conf.get("evaluate_every", 1),
    )

    fl.server.start_server(
        server_address=server_conf.get("address", "0.0.0.0:8080"),
        config=fl.server.ServerConfig(num_rounds=strategy_conf.get("rounds", 10)),
        strategy=strategy,
    )

if __name__ == "__main__":
    main()
