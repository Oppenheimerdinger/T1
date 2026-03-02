from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom
from torch.utils.data import DataLoader

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'custom': Dataset_Custom,
    'Weather': Dataset_Custom,
    'ECL': Dataset_Custom,
    'Exchange': Dataset_Custom,
    'ILI': Dataset_Custom,
}


def data_provider(args, flag):
    data_source = getattr(args, 'data_source', 'tslib')

    if data_source == 'tslib':
        return _tslib_provider(args, flag)
    elif data_source == 'benchpots':
        return _benchpots_provider(args, flag)
    elif data_source == 'csdi':
        return _csdi_provider(args, flag)
    else:
        raise ValueError(f"Unknown data_source: {data_source}")


def _tslib_provider(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    shuffle_flag = flag == 'train'
    batch_size = args.batch_size
    freq = args.freq

    data_set = Data(
        args=args,
        root_path=args.root_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        data_path=args.data_path,
        target=args.target,
        timeenc=timeenc,
        freq=freq,
    )
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=False,
    )
    return data_set, data_loader


def _benchpots_provider(args, flag):
    from data_provider.data_loader_pypots import BenchPOTSWrapper

    subset_map = {'train': 'train', 'val': 'val', 'test': 'test'}
    data_set = BenchPOTSWrapper(
        dataset_name=args.data.lower(),
        subset=subset_map[flag],
        root_path=args.root_path,
        mit_rate=args.mask_rate,
    )

    shuffle_flag = flag == 'train'
    data_loader = DataLoader(
        data_set,
        batch_size=args.batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=False,
    )
    return data_set, data_loader


def _csdi_provider(args, flag):
    from data_provider.data_loader_pypots import PM25Wrapper

    subset_map = {'train': 'train', 'val': 'val', 'test': 'test'}
    data_set = PM25Wrapper(
        subset=subset_map[flag],
        root_path=args.root_path,
        eval_length=args.seq_len,
        target_dim=args.enc_in,
    )

    shuffle_flag = flag == 'train'
    data_loader = DataLoader(
        data_set,
        batch_size=args.batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=False,
    )
    return data_set, data_loader
