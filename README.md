# FL-for-SmartCity Project

## Overview

The FL-for-SmartCity project aims to develop federated learning-based solutions tailored for Smart Cities. As privacy preservation becomes increasingly important in Smart City initiatives, federated learning has emerged as a promising approach to safeguard sensitive data.
This project encompasses several tasks that reflect the research and development outcomes of federated learning applications in Smart City contexts. 
The FL-for-SmartCity project is currently in the development phase.

## Tasks
1. FedHPO  
   FedHPO focuses on Hyperparameter Optimization (HPO) within the Flower Framework. This task aims to leverage Flower for hyperparameter optimization (HPO) without modifying the core Flower framework code.

2. SlowFast-based FedVAD   
   VAD stands for Video Anomaly Detection. The purpose of this system is to identify abnormal events and ensure public safety. The video sources are CCTV cameras installed in parks. However, transmitting these videos to a centralized server for VAD poses potential risks related to privacy and security. To address these concerns, we adopted a Federated Learning for VAD and adjusted the open-source SlowFast model to suit the needs of this project.

3. FL-edges
   This project sets up a Docker-based federated learning (FL) environment for a server (Ubuntu) and edge clients (e.g., Jetson Nano, AGX Xavier, Xavier NX).  
   It is designed to facilitate development and deployment of FL workflows such as those using Flower with clients performing local training and a centralized server aggregating updates.

   ## Structure

   - `server_code/` : Server‐side application code and `Dockerfile.server`
   - `client_code/` : Client‐side application code and `Dockerfile.client`
   - `deploy_allClients/` : Script to deploy the client module to all edge devices at once `deploy_clients.sh`
   - `docker-compose.yml` : Compose file to spin up server and optionally one or more clients as containers
   - `.env` : Environment variables (e.g., SERVER_ADDRESS, VERSION)  


   ## Quick Start
   1) Clone the repository:
      ```bash
      git clone https://github.com/make-a-better-world/FL-for-SmartCity.git
      cd FL-for-SmartCity/FL-edges
   2) Build and run the server image:
      cd server_code
      docker build -t flwr-server:latest -f Dockerfile.server .
      docker run -d -p 8080:8080 flwr-server:latest
   3) Build and run a client image (example):
      cd ../client_code
      docker build -t flwr-client:latest -f Dockerfile.client .
      docker run -d --network host -e SERVER_ADDRESS=YOUR_SERVER_IP:8080 -e CLIENT_ID=client-edge-1 flwr-client:latest
   4) Alternatively use Docker Compose from project root:
      docker compose up --build -d






