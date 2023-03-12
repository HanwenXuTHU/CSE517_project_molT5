from torch import nn
from transformers import T5Tokenizer, T5ForConditionalGeneration

class MolT5_smiles2text(nn.Module):
    def __init__(self, option='t5-base', max_len=512):
        super(MolT5_smiles2text, self).__init__()
        self.max_len = max_len
        self.tokenizer = T5Tokenizer.from_pretrained('laituan245/molt5-base')
        self.text_tokenizer = T5Tokenizer.from_pretrained('t5-base')
        self.model = T5ForConditionalGeneration.from_pretrained(option)
        # set the model learnable
        for param in self.model.parameters():
            param.requires_grad = True
    
    def forward(self, smiles_list, description_list=None):
        input_ids = self.tokenizer.batch_encode_plus(smiles_list, 
                                                    return_tensors='pt', 
                                                    padding=True, 
                                                    truncation=True, 
                                                    max_length=self.max_len)['input_ids']
        input_ids = input_ids.to(self.model.device)
        label_ids = self.text_tokenizer.batch_encode_plus(description_list,
                                                        return_tensors='pt',    
                                                        padding=True,
                                                        truncation=True,
                                                        max_length=self.max_len)['input_ids']
        label_ids = label_ids.to(self.model.device)
        outputs = self.model(input_ids=input_ids,
                            labels=label_ids,
                            return_dict=True)
        return outputs.loss, outputs.logits

    def generate(self, smiles_list):
        input_ids = self.tokenizer.batch_encode_plus(smiles_list, 
                                                    return_tensors='pt', 
                                                    padding=True, 
                                                    truncation=True, 
                                                    max_length=self.max_len)['input_ids']
        outputs = self.model.generate(input_ids=input_ids)

        return self.tokenizer.batch_decode(outputs, 
                                        num_beams=2,
                                        repetition_penalty=2.5,
                                        length_penalty=1.0,
                                        early_stopping=True,
                                        skip_special_tokens=True)

class MolT5_text2smiles(nn.Module):
    def __init__(self, option='t5-base', max_len=512):
        super(MolT5_text2smiles, self).__init__()
        self.max_len = max_len
        self.tokenizer = T5Tokenizer.from_pretrained('laituan245/molt5-base')
        self.text_tokenizer = T5Tokenizer.from_pretrained('t5-base')
        self.model = T5ForConditionalGeneration.from_pretrained(option)
        # set the model learnable
        for param in self.model.parameters():
            param.requires_grad = True
    
    def forward(self, description_list, smiles_list=None):
        input_ids = self.text_tokenizer.batch_encode_plus(description_list, 
                                                        return_tensors='pt', 
                                                        padding=True, 
                                                        truncation=True, 
                                                        max_length=self.max_len)['input_ids']
        input_ids = input_ids.to(self.model.device)
        label_ids = self.tokenizer.batch_encode_plus(smiles_list,
                                                    return_tensors='pt',    
                                                    padding=True,
                                                    truncation=True,
                                                    max_length=self.max_len)['input_ids']
        label_ids = label_ids.to(self.model.device)
        outputs = self.model(input_ids=input_ids,
                            labels=label_ids,
                            return_dict=True)
        return outputs.loss, outputs.logits

    def generate(self, description_list):
        input_ids = self.text_tokenizer.batch_encode_plus(description_list, 
                                                        return_tensors='pt', 
                                                        padding=True, 
                                                        truncation=True, 
                                                        max_length=self.max_len)['input_ids']
        outputs = self.model.generate(input_ids=input_ids)

        return self.tokenizer.batch_decode(outputs, 
                                        num_beams=2,
                                        repetition_penalty=2.5,
                                        ength_penalty=1.0,
                                        early_stopping=True,
                                        skip_special_tokens=True)