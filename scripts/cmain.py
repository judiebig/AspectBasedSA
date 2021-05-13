import argparse
import importlib


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='laptop', help='dataset: laptop/restaurant/mooc')
parser.add_argument('--lr', default=0.01, help='learning rate')
parser.add_argument('--batch_size', default=100, help='batch size')
parser.add_argument('--epochs', default=20, help='epoch')
parser.add_argument('--emb_dim', default=50, help='embedding dim')
parser.add_argument('--vocab_size', default=3690, help='laptop:3690/restaurant:4379/self:?')
opt = parser.parse_args()


def main():
    model_module = importlib.import_module('model.text_cnn')
    model = getattr(model_module, "TextCNN")(opt)
    if opt.dataset in ['laptop', 'restaurant']:
        trainer_module = importlib.import_module('trainer.semeval_trainer')
        trainer = getattr(trainer_module, "SemEvalTrainer")(model, opt)
    elif opt.dataset == 'mooc':
        trainer_module = importlib.import_module('trainer.mooc_trainer')
        trainer = getattr(trainer_module, "MoocTrainer")(model, opt)
    # load data
    trainer.load_data(opt.dataset)
    trainer.train()


if __name__ == '__main__':
    main()