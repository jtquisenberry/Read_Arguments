import argparse

def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    '''
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-cased,bert-large-cased")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    '''

    # Other parameters
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev/test set.")
    parser.add_argument("--eval_on",
                        default="dev",
                        type=str,
                        help="Evaluation set, dev: Development, test: Test")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=64,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3,
                        type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    # training stratergy arguments
    parser.add_argument("--multi_gpu",
                        action='store_true',
                        help="Set this flag to enable multi-gpu training using MirroredStrategy."
                             "Single gpu training")
    # JQ
    parser.add_argument("--device_type", default='CPU', type=str,
                        help="Specify whether this application will be run on a CPU or GPUs")
    parser.add_argument("--gpus",default='0',type=str,
                        help="Comma separated list of gpus devices."
                              "For Single gpu pass the gpu id.Default '0' GPU"
                              "For Multi gpu,if gpus not specified all the available gpus will be used")

    args = parser.parse_args()
    return args

class Config():
    def __init__(self, args):
        for key in args:
            setattr(self, key, args[key])




if __name__ == '__main__':
    args = main()
    print(args.__dict__)
    c = Config(args.__dict__)
    print(c.__dict__)
    d = 1