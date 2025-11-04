#!binbash
# deploy_clients.sh
# 여러 클라이언트 장비에 도커 이미지 빌드 및 실행 자동화

# 서버 주소 (클라이언트가 접속할 FL-server 주소)
SERVER_ADDR=your.server.ip8080

# 클라이언트 장비 목록 (IP 또는 hostname)
CLIENTS=(jetson-nano1 jetson-agx1 jetson-nx1)

# 클라이언트 이미지 태그 및 파일 위치
IMAGE_TAG=flwr-client1.23.0
DOCKERFILE_PATH=.client_codeDockerfile.client
CONTEXT_PATH=.client_code

# SSH 키 및 유저 설정
SSH_USER=ubuntu
SSH_KEY_PATH=~.sshid_rsa

for CLIENT in ${CLIENTS[@]}; do
  echo ---- Deploying to ${CLIENT} ----

  # 1) SSH 접속해서 기존 컨테이너 정지삭제
  ssh -i ${SSH_KEY_PATH} ${SSH_USER}@${CLIENT} 
    docker ps ­-q --filter name=flwr_client  xargs ­-r docker stop && docker container prune -f

  # 2) 이미지 빌드 (현지 또는 리모트)
  # 여기선 로컬에서 이미지 빌드하고 레지스트리에 푸시하거나
  # 또는 SSH 내부에서 빌드할 수 있어요.
  echo Building image locally...
  docker build -t ${IMAGE_TAG} -f ${DOCKERFILE_PATH} ${CONTEXT_PATH}
  
  # 3) 이미지 푸시 (프라이빗 레지 사용 시) 또는 직접 전송
  # 예 docker push your-registry${IMAGE_TAG}

  # 4) 클라이언트에서 이미지 풀 또는 직접 실행
  ssh -i ${SSH_KEY_PATH} ${SSH_USER}@${CLIENT} 
    docker pull your-registry${IMAGE_TAG}  true && 
     docker run -d --rm --gpus all --network host 
       -e SERVER_ADDRESS=${SERVER_ADDR} 
       -e CLIENT_ID=${CLIENT} 
       ${IMAGE_TAG}
  
  echo ---- ${CLIENT} deployment complete ----
done

echo All clients deployed.
