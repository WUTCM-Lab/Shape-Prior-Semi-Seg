from .tn3k import tn3kDataSet
from .BUSI import BUSIDataSet

def build_dataset(args):
    if args.manner == 'test':
        if args.dataset == 'tn3k':
            test_data = tn3kDataSet(args.root, args.expID, mode='test')
        elif args.dataset == 'BUSI':
            test_data = BUSIDataSet(args.root, args.tn3k, mode='test')
        return test_data
    else:
        if args.dataset == 'tn3k':
            train_data = tn3kDataSet(args.root, args.expID, mode='train', ratio=args.ratio, sign='label')
            valid_data = tn3kDataSet(args.root, args.expID, mode='valid')
            test_data = tn3kDataSet(args.root, args.expID, mode='test')
            train_u_data = None
            if args.manner == 'semi' or args.manner == 'self':
                train_u_data = tn3kDataSet(args.root, args.expID, mode='train', ratio=args.ratio, sign='unlabel')
        elif args.dataset == 'BUSI':
            train_data = BUSIDataSet(args.root, args.expID, mode='train', ratio=args.ratio, sign='label')
            valid_data = BUSIDataSet(args.root, args.expID, mode='valid')
            train_u_data = None
            if args.manner == 'semi':
                train_u_data = BUSIDataSet(args.root, args.expID, mode='train', ratio=args.ratio, sign='unlabel')
        return train_data, train_u_data, valid_data


