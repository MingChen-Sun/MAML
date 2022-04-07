import argparse
def get_args():
    parser = argparse.ArgumentParser(description='SMC')
    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("--input-size", type=int, default=1433)
    parser.add_argument("--output-size", type=int, default=7)
    parser.add_argument("--hidden-size", type=list, default=[64,128])
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--lc", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--task_num", type=int, default=3)
    parser.add_argument("--sample_num_spt", type=int, default=5)
    parser.add_argument("--sample_num_qry", type=int, default=2)    
    parser.add_argument("--batch_size", type=int, default=3)
    args = parser.parse_args()
    return args