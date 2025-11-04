# client_code/client_app.py

from ..config_loader import load_config
import argparse
import torch
import flwr as fl

class Client(fl.client.NumPyClient):
    def __init__(self, model, train_loader, val_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config

    def get_parameters(self):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def fit(self, parameters, config):
        # 모델에 파라미터 적용
        params_dict = dict(zip(self.model.state_dict().keys(), parameters))
        self.model.load_state_dict({k: torch.tensor(v) for k, v in params_dict.items()})

        # 로컬 학습 설정
        epochs = self.config.get("local_epochs", 1)
        batch_size = self.config.get("batch_size", 32)
        lr = self.config.get("learning_rate", 0.001)

        optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
        criterion = torch.nn.CrossEntropyLoss()

        self.model.train()
        for epoch in range(epochs):
            for inputs, targets in self.train_loader:
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # 학습 후 파라미터 반환
        return self.get_parameters(), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        # 모델 파라미터 적용
        params_dict = dict(zip(self.model.state_dict().keys(), parameters))
        self.model.load_state_dict({k: torch.tensor(v) for k, v in params_dict.items()})

        self.model.eval()
        total_loss = 0.0
        total_examples = 0
        criterion = torch.nn.CrossEntropyLoss()

        with torch.no_grad():
            for inputs, targets in self.val_loader:
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                total_loss += loss.item() * targets.size(0)
                total_examples += targets.size(0)

        avg_loss = total_loss / total_examples if total_examples > 0 else 0.0
        return float(avg_loss), total_examples, {}

def main():
    parser = argparse.ArgumentParser(description="Flower Client")
    parser.add_argument(
        "--config",
        type=str,
        default="client_code/config/client_config.yml",
        help="Path to client config YAML file",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    client_conf = config.get("client", {})
    training_conf = config.get("training", {})

    # 예시 모델 및 데이터로더 초기화 — 실제 구현으로 교체하세요
    model = torch.nn.Linear(10, 2)  # 예시: 입력크기10, 클래스2
    train_loader = []  # 실제 DataLoader로 변경 필요
    val_loader = []    # 실제 DataLoader로 변경 필요

    client = Client(model, train_loader, val_loader, training_conf)

    fl.client.start_numpy_client(
        server_address=client_conf.get("server_address"),
        client=client
    )

if __name__ == "__main__":
    main()
