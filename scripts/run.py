"""The main running script

All of the experiments carried out in this project can be run using this script
The script takes tw arguments, the model type as a positional argument and
--only_predict which is an optional argument that prevents the model from training if given
"""

from pathlib import Path
import sys
path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))

from argparse import ArgumentParser
from models import Runner
from models.MLP.mlp_runner import MlpRunner
from models.SVM.svm_runner import SvmRunner
from models.BiLSTM.bilstm_runner import BiLSTMRunner
from models.BERT.bert_runner import BertRunner
from models.BERT.bilstmbert_runner import BiLSTMBertRunner
from models.BERT.statsbert_runner import StatsBertRunner


def main(args):
    model: Runner = Runner()
    if args.model == 'mlp':
        print("running mlp")
        model = MlpRunner()
    elif args.model == 'svm':
        print("running svm")
        model = SvmRunner()
    elif args.model == 'bilstm':
        print("running biLSTM")
        model = BiLSTMRunner()
    elif args.model == 'bert':
        print("running bert")
        model = BertRunner()
    elif args.model == 'bilstm_bert':
        print("running biLSTM_bert")
        model = BiLSTMBertRunner()
    elif args.model == 'stats_bert':
        print("running stats_bert")
        model = StatsBertRunner()
    if not args.only_predict:
        model.train()
    model.evaluate()


if __name__ == "__main__":
    parser = ArgumentParser(description='Train and evaluate different models on the sentiment classification task in an end to end fashion.')
    parser.add_argument('model', choices=['mlp','svm', 'bilstm', 'bert', 'bilstm_bert', 'stats_bert'], default='bilstm', help='specify the model to be run')
    parser.add_argument('--only_predict', action='store_true', help='if given, will not train the model, if there is no checkpoint given in the configuration files, runs prediction with uninitialized model')
    args = parser.parse_args()
    main(args)
