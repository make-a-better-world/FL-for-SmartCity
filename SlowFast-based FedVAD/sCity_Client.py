import flwr as fl
import torch
from SlowFast.slowfast.config.defaults import get_cfg
from SlowFast.slowfast.utils.parser import parse_args, load_config
from SlowFast.slowfast.utils.meters import TestMeter, TrainMeter, ValMeter
from SlowFast.slowfast.models import build_model
from SlowFast.slowfast.utils.checkpoint import load_train_checkpoint
from SlowFast.slowfast.datasets import loader
import SlowFast.slowfast.utils.distributed as du
import SlowFast.slowfast.utils.misc as misc
import SlowFast.slowfast.models.optimizer as optim
import SlowFast.slowfast.utils.logging as logging
#from SlowFast.slowfast.utils.misc import launch_job
import SlowFast.slowfast.utils.misc as misc
from SlowFast.tools.train_net import train
from SlowFast.tools.test_net import  test
##from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
import numpy as np
import io


logger = logging.get_logger(__name__)

class sCityClient(fl.client.NumPyClient):
    def __init__(self,args):
        print("config files: {}".format(args.cfg_files))    
        self.init_method = args.init_method          
        self.cfg = load_config(args, args.cfg_files[0])
     
        self.model = build_model(self.cfg)
         
    def get_parameters(self, config=None):
        """Return the current local model parameters."""
        return [val.cpu().numpy() for val in self.model.state_dict().values()]  #단순히 모델 학습 파라미터 뿐 아니라 여러 모든 상태 값을 포함. 예를 들어, 배치 정규화 계층의 러닝 평균과 분산값 같은 학습되지 않는 값들도 포함할 수 있음

    def set_parameters(self, parameters):
        """Set the local model parameters."""
        # 디버깅: 전달받은 parameters 값을 출력
        print("Received parameters: \n")
        ##print(parameters)
 
        # Match the incoming parameters to the model's state_dict
        params_dict = zip(self.model.state_dict().keys(), parameters)
        
        # Convert parameters to tensors and create a new state_dict
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        
        # 디버깅: 일부 파라미터 출력 (예: 첫 번째 파라미터의 첫 5개 값)
        print("Before applying new parameters:", list(self.model.parameters())[0][:5])
         # Load the state_dict into the model
        self.model.load_state_dict(state_dict, strict=True) 
        # 디버깅: 적용 후 일부 파라미터 출력
        print("After applying new parameters:", list(self.model.parameters())[0][:5])
                                 
        
    def fit(self, parameters, config):
       # TO-Do :  파라미터 값 확인

       # Debug.. server로부터 받은 학습 환경 출력 
        print(config["batch_size"])     # Prints `32`
        print(config["current_round"])  # Prints `1`/`2`/`...`
        print(config["local_epochs"])   # Prints `2`
        print(config["optimizer"])      # Prints `1`/`2`/`...`
        print(config["learning_rate"])  # Prints `2`


        print(">>>> set parameters")  # 
       
        # 서버로부터 받은 파라미터를 모델에 설정
        self.set_parameters(parameters)
        print(">>>> input config...")  # 
        print(config)  ##server config

        # 학습 및 검증 실행
        train_meter, val_meter, cntDataset = train(self.cfg, self.model, True)  # newRound 를 true로..
        
        # 검증 실행 (train 함수에서 검증 로직 추가 또는 별도의 검증 함수 사용)
        
        # 학습 중의 정확도와 손실

### 24.11.27       accuracy, avg_loss, min_top1_err = train_meter.get_metrics()
        metrics = train_meter.get_metrics()
        accuracy = metrics["accuracy"]
        avg_loss = metrics["loss"]
        min_top1_err = train_meter.min_top1_err
        precision = metrics["precision"]
        recall = metrics["recall"]
        f1_score = metrics["f1_score"]

    #   검증 중의 정확도와 손실
        val_accuracy = float(100 - val_meter.min_top1_err)  # 검증 정확도
    #    val_loss = float(val_meter.avg_loss)  # 검증 손실
        val_avgLoss = val_meter.get_avg_loss()   
        
        print(f">>>>   cntDataset : {int(cntDataset)}, accuracy: {accuracy}, loss: {avg_loss}, valacc: {val_accuracy}, valloss: {val_avgLoss}")
     
        return self.get_parameters(), int(cntDataset), {
            "accuracy": accuracy,    # 학습 정확도
            "loss": avg_loss,        # 학습 손실
            "valacc": val_accuracy,  # 검증 정확도
            "valloss": val_avgLoss,  # 검증 손실
            "precision" : precision,
            "recall"  : recall,
            "f1_score" : f1_score
    } 
        
    def evaluate(self, parameters, config):  ##test data로 validation
        self.set_parameters(parameters)
        self.model.eval()
        test_meter, cntTestDaset = test(self.cfg)
        print(">>>> evaluation result :", test_meter.stats["loss"], (cntTestDaset), {"accuracy": test_meter.stats["top1_acc"]})
        return float(test_meter.stats["loss"]), int(cntTestDaset), {"accuracy": test_meter.stats["top1_acc"]}
        

def main():
    # Load configuration and setup
    args = parse_args()
    client = sCityClient(args)
    fl.client.start_numpy_client(server_address="129.254.88.249:8080", client=client)

if __name__ == "__main__":
    main()