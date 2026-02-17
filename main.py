import yaml


from train import process_train, process_pre_filter, process_test

def load_config(yaml_file):
    with open(yaml_file, 'r', encoding='utf-8') as f:
        return yaml.load(f, Loader=yaml.FullLoader)

if __name__ == '__main__':
    config = load_config('config.yaml')
    if config["stage"]["pre_filter"]:
        process_pre_filter(config)
    if config["stage"]["train"]:
        process_train(config)
    if config["stage"]["test"]:
        process_test(config)

