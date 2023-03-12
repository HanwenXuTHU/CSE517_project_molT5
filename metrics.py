import nltk
nltk.download('punkt')
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
import rouge

def calculate_bleu_score(pred_list, label_list, metrics=['bleu-2', 'bleu-4']):
    _pred_list, _label_list = [], []
    for pred, label in zip(pred_list, label_list):
        len_min = min(len(pred), len(label))
        _pred_list.append(pred[:len_min])
        _label_list.append(label[:len_min])
    pred_list, label_list = _pred_list, _label_list
    
    # bleu_scores = dict(**{metric: 0 for metric in metrics})
    bleu_scores = dict(**{metric: [] for metric in metrics})
    Ns = [int(metric.split('-')[-1]) for metric in metrics]
    Ws = [tuple([1. / N for _ in range(N)]) for N in Ns]
    # print(pred_list[0])
    # print(label_list[0])
    for pred, label in zip(pred_list, label_list):
        for k, metric in enumerate(metrics):
            # bleu_scores[metric] += corpus_bleu(label, pred, weights=Ws[k])
            # bleu_scores[metric].append(corpus_bleu(label, pred, weights=Ws[k]))
            bleu_scores[metric].append(sentence_bleu([label], pred, weights=Ws[k]))
    # n = len(pred_list)
    # for metric in metrics:
    #     bleu_scores[metric] /= n
    return bleu_scores

def calculate_rouge_score(pred_list, label_list, metrics=['rouge-l']):
    # rouge_scores = dict(**{metric: 0 for metric in metrics})
    rouge_scores = dict(**{metric: [] for metric in metrics})
    # evaluator = rouge.Rouge(metrics=metrics, max_n=2, limit_length=True, length_limit=100, length_limit_type='words', apply_avg=True, apply_best=False, alpha=0.5, weight_factor=1.2, stemming=True)
    evaluator = rouge.Rouge()
    # print(evaluator.metrics)
    # print(evaluator.stats)
    for pred, label in zip(pred_list, label_list):
        scores = evaluator.get_scores(pred, label)[0]
        for metric in metrics:
            # rouge_scores[metric] += scores[metric]['f']
            rouge_scores[metric].append(scores[metric]['f'])
    # n = len(pred_list)
    # for metric in metrics:
    #     rouge_scores[metric] /= n
    return rouge_scores
