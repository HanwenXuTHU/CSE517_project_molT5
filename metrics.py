from nltk.translate.bleu_score import sentence_bleu
import rouge
import nltk
nltk.download('punkt')

def calculate_bleu_score(pred_list, label_list):
    bleu_score = 0
    for pred, label in zip(pred_list, label_list):
        bleu_score += sentence_bleu([label], pred)
    return bleu_score / len(pred_list)

def calculate_rouge_score(pred_list, label_list, metrics=['rouge-n']):
    rouge_scores = dict(**{metric: 0 for metric in metrics})
    evaluator = rouge.Rouge(metrics=metrics, max_n=2, limit_length=True, length_limit=100, length_limit_type='words', apply_avg=True, apply_best=False, alpha=0.5, weight_factor=1.2, stemming=True)
    for pred, label in zip(pred_list, label_list):
        for metric in metrics:
            rouge_scores[metric] += evaluator.get_scores(pred, label)[metric]['f']
    n = len(pred_list)
    for metric in metrics:
        rouge_scores[metric] /= n
    return rouge_scores