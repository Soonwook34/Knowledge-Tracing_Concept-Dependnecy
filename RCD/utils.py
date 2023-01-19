import argparse
from build_graph import build_graph


class CommonArgParser(argparse.ArgumentParser):
    def __init__(self):
        super(CommonArgParser, self).__init__()
        self.add_argument('--exer_n', type=int, default=598,
                          help='The number for exercise.')
        self.add_argument('--knowledge_n', type=int, default=148,
                          help='The number for knowledge concept.')
        self.add_argument('--student_n', type=int, default=540,
                          help='The number for student.')
        self.add_argument('--gpu', type=int, default=0,
                          help='The id of gpu, e.g. 0.')
        self.add_argument('--epoch_n', type=int, default=50,
                          help='The epoch number of training')
        self.add_argument('--lr', type=float, default=0.0001,
                          help='Learning rate')
        self.add_argument('--dim', type=int, default=128,
                          help='Embedding dimension')
        self.add_argument('--log', type=str, default="log",
                          help='Save file name')
        self.add_argument('--dir', type=str, default="../data/poly/",
                          help='Directory Path')
        # KT / OT 모드
        self.add_argument('--mode', type=int, default="0",
                          help='Training mode (0: KT, 1: KT+OT (use lamb), 2: OT')
        self.add_argument('--lamb', type=float, default=0.2,
                          help='Ratio between OT loss and KT loss (only in mode 1)')
        self.add_argument('--option_n', type=int, default="4",
                          help='Number of options in a exercise')
        # Edge Embedding
        self.add_argument('--edge_emb', action='store_true',
                          help='Use edge embedding')
        self.add_argument('--edge_type', type=int, default="0",
                          help='Edge embedding type / 0: src+tar, 1: src+tar+option, 2: option embedding')
        # Early Stopping
        self.add_argument('--early_stopping', type=float, default=0.0,
                          help='Early Stopping min difference')
        self.add_argument('--patience_max', type=int, default=5,
                          help='Early Stopping Patience Max Count')



def construct_local_map(args):
    local_map = {
        'directed_g': build_graph('direct', args.knowledge_n, args.dir),
        'undirected_g': build_graph('undirect', args.knowledge_n, args.dir),
        'k_from_e': build_graph('k_from_e', args.knowledge_n + args.exer_n, args.dir),
        'e_from_k': build_graph('e_from_k', args.knowledge_n + args.exer_n, args.dir),
        'u_from_e': build_graph('u_from_e', args.student_n + args.exer_n, args.dir),
        'e_from_u': build_graph('e_from_u', args.student_n + args.exer_n, args.dir),
    }
    return local_map
