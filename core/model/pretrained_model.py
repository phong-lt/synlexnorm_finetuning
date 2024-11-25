from transformers import AutoModelForSeq2SeqLM
from torch import nn

class LexBARTModel(nn.Module):
    def __init__(self, 
                pretrained_name
                ):
        super().__init__()

        self.model = AutoModelForSeq2SeqLM.from_pretrained(pretrained_name)

    def forward(self, 
                input_ids, 
                label_ids, 
                src_attention_mask, 
                label_attention_mask):

        encoder_outputs = self.model.model.encoder(
                attention_mask=src_attention_mask,
                inputs_embeds=self.model.model.shared(input_ids),
            ).last_hidden_state

        decoder_outputs = self.model.model.decoder(
            encoder_hidden_states = encoder_outputs,
            inputs_embeds = self.model.model.shared(label_ids),
            attention_mask = label_attention_mask
        ).last_hidden_state


        return self.model.lm_head(decoder_outputs)
    
    def generate(self, 
                input_ids, 
                max_length):
        return self.model.generate(input_ids = input_ids,
                                    max_length = max_length)

class LexT5Model(nn.Module):
    def __init__(self, 
                pretrained_name
                ):
        super().__init__()

        self.model = AutoModelForSeq2SeqLM.from_pretrained(pretrained_name)

    def forward(self, 
                input_ids, 
                label_ids, 
                src_attention_mask, 
                label_attention_mask):

        encoder_outputs = self.model.encoder(
                attention_mask=src_attention_mask,
                inputs_embeds=self.model.shared(input_ids),
            ).last_hidden_state

        decoder_outputs = self.model.decoder(
            encoder_hidden_states = encoder_outputs,
            inputs_embeds = self.model.shared(label_ids),
            attention_mask = label_attention_mask
        ).last_hidden_state


        return self.model.lm_head(decoder_outputs)
    
    def generate(self, 
                input_ids, 
                max_length):
        return self.model.generate(input_ids = input_ids, 
                                    max_length = max_length)