from nltk.translate.bleu_score import sentence_bleu
import rouge

def calculate_bleu_score(pred_list, label_list):
    bleu_score = 0
    for pred, label in zip(pred_list, label_list):
        bleu_score += sentence_bleu([label], pred)
    return bleu_score / len(pred_list)

def calculate_rouge_score(pred_list, label_list):
    rouge_score = 0
    for pred, label in zip(pred_list, label_list):
        rouge_score += rouge.get_scores(pred, label)[0]['rouge-l']['f']
    return rouge_score / len(pred_list)