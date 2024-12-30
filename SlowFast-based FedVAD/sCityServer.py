##################################################################################
#  sCityServer 프로그램  : Federated Learning의 서버
#   - 클라이언트와의 연합학습을 위한 서버 (port: 8080)
#  
##################################################################################
import flwr as fl
from  ScityStrategy import ScityStrategy
from sCityConfig import ScityConfig
import threading

iniFile = 'sCityServer.ini'
flaa = ''


def getConfig(configFile):
    """get FL Server IP:port from the config file"""
    global flaa
    sCity_conf = ScityConfig(configFile)
    flServer = sCity_conf.get_flServer()    # 연합학습  server
    ##sPort = sCity_conf.get_socketPort()     # 인증서버 포트
    flPort = sCity_conf.get_flPort()        # 연합학습 포트
    flaa = flServer + ':'+ flPort    
    return sCity_conf

def main():
#def runFLServer():
    sCity_conf = getConfig(iniFile)  
    """FL server start """
    strategy = ScityStrategy(sCity_conf)

    ##  get FL info
    server_round = strategy.flRound
    ##   config = strategy.fit_config(server_round)  #server_round


    hist = fl.server.start_server(server_address=flaa,
                config=fl.server.ServerConfig(num_rounds=server_round),
                #  force_final_distributed_eval = True, # hist.metrics_distributed 를 강제로 계산하기 위함  
                strategy=strategy,
                )

    print(">>>> FL result ==> ", hist)     
    # History 결과 출력
    print(">>>> hist.losses_distributed : ",hist.losses_distributed)
    print(">>>> hist.metrics_distributed : ",hist.metrics_distributed)

    print(">>>> Finishing FL..")


if __name__ == "__main__":
    main()