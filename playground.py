def init_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    config = process_config(args.config)

    return config

print(config.load_checkpoint)