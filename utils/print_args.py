def print_args(args):
    """Print experiment arguments in a formatted way."""
    if hasattr(args, '__dict__'):
        # argparse Namespace
        d = vars(args)
    elif hasattr(args, 'keys'):
        # dict-like (OmegaConf DictConfig)
        d = dict(args)
    else:
        print(args)
        return

    max_key_len = max(len(str(k)) for k in d.keys()) if d else 0
    for k, v in sorted(d.items()):
        print(f'  {k:<{max_key_len}} : {v}')
