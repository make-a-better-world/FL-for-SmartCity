import yaml
import os
from datetime import date, datetime
import csv
import random
import numpy as np
import tensorflow as tf

def generate_config_server(args):
    yaml_file = args.config
    with open(file=yaml_file) as file:
        try:
            config = yaml.safe_load(file)   
            server_config = {}

            server_config['max_rounds'] = config['server']['max_rounds']
            server_config['min_fit_clients'] = config['server']['min_fit_clients']
            server_config['min_avalaible_clients'] = config['server']['min_avalaible_clients']
            server_config['strategy'] = config['server']['strategy']
            server_config['data_type'] = config['common']['data_type']
            server_config['dataset'] = config['common']['dataset']
            server_config['target_acc'] = config['common']['target_acc']

            return server_config
        except yaml.YAMLError as exc:
            print(exc)


def generate_config_client(args):
    yaml_file = args.config
    with open(file=yaml_file) as file:
        try:
            config = yaml.safe_load(file)   
            client_config = {}
            client_config['partition'] = args.partition
            client_config['client_id'] = args.client_id
            client_config['epochs'] = config['client']['epochs']
            client_config['batch_size'] = config['client']['batch_size']
            # client_config['hpo'] = config['common']['hpo']
            client_config['data_type'] = config['common']['data_type']
            client_config['dataset'] = config['common']['dataset']
            client_config['strategy'] = config['server']['strategy']
            client_config['server_address'] = config['server']['address']

            return client_config
        except yaml.YAMLError as exc:
            print(exc)

def gen_dir_outfile_server(config):
    # generates the basic directory structure for out data and the header for file
    # Initialize variables
    today = date.today()
    data_dist_type = config['data_type']
    base_dir = "out"
    current_time = datetime.now().strftime("%H-%M-%S")

    # Define directory paths
    base_path = os.path.join(base_dir, str(today), config['dataset'], config['strategy'], data_dist_type)

    # Create the necessary directories using os.makedirs, which handles intermediate directories
    os.makedirs(base_path, exist_ok=True)

    # Create final directory based on the number of existing subdirectories
    final_dir_path = os.path.join(base_path, str(len(os.listdir(base_path))))
    os.makedirs(final_dir_path, exist_ok=True)

    # Define output file path
    out_file_path = os.path.join(final_dir_path, f"{current_time}_server.csv")

    # Print the base path (if needed for debugging)
    print(out_file_path)

    # create empty server history file
    if not os.path.exists(out_file_path):
        with open(out_file_path, 'w', encoding='UTF8') as f:
            # create the csv writer
            header = ["round", "accuracy", "loss", "time"]
            writer = csv.writer(f)
            writer.writerow(header)
            f.close()
    return out_file_path


def gen_out_file_client(config):
    # Generates the basic directory structure for out data and the header for the file
    today = date.today()
    data_dist_type = config['data_type']
    base_dir = "out"
    
    # Define the complete directory path structure
    base_path = os.path.join(base_dir, str(today), config['dataset'], config['strategy'], data_dist_type)
    
    # Create necessary directories in one step, if they don't exist
    os.makedirs(base_path, exist_ok=True)
    
    # List directories and determine the last updated directory
    dirs = sorted(os.listdir(base_path))
    last_updated_dir = dirs[-1] if dirs else '0'
    
    # Final directory path
    final_dir_path = os.path.join(base_path, str(last_updated_dir))
    os.makedirs(final_dir_path, exist_ok=True)
    
    # Get the current time for the filename
    current_time = datetime.now().strftime("%H-%M-%S")
    
    # Define the output file path
    file_path = os.path.join(final_dir_path, f"{current_time}_client_{config['client_id']}.csv")
    
    return file_path



def set_seed(seed: int = 13) -> None:
  random.seed(seed)
  np.random.seed(seed)
  tf.random.set_seed(seed)
  tf.experimental.numpy.random.seed(seed)
  # When running on the CuDNN backend, two further options must be set
  os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
  os.environ['TF_DETERMINISTIC_OPS'] = '1'
  # Set a fixed value for the hash seed
  os.environ["PYTHONHASHSEED"] = str(seed)
  print(f"Random seed set as {seed}")