from transformers import BertPreTrainedModel, BertModel
import torch
import torch.nn as nn
import torch.nn.functional as F

class QaBERT(BertPreTrainedModel):
    def __init__(self, config):
        super(QaBERT, self).__init__(config)
        self._device = config.device
        self.label_count = config.num_labels

        self.bert = BertModel(config)
        self.dropout_layer = nn.Dropout(0.1)
        self.pooler_usage = config.use_pooler
        if self.pooler_usage:
            self.output_layer = nn.Linear(config.hidden_size, config.num_labels)
        else:
            self.output_layer_cat = nn.Linear(config.hidden_size * 4, config.num_labels)

        self.class_weights = config.weight_class

        self.init_weights()


    def calculate(self, input_ids, attention_mask=None, token_type_ids=None):
        bert_outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooler_output = bert_outputs[1]

        if self.pooler_usage:
            final_output = self.dropout_layer(pooler_output)
        else:
            hidden_states = bert_outputs[2][1:]
            hidden_states = torch.stack(hidden_states, 1)

            last_four_hidden_states = hidden_states[:, 8:, :, :]
            first_four_last_hidden_states = last_four_hidden_states[:, :, 0, :]
            first_four_last_hidden_states = first_four_last_hidden_states.contiguous().view(
                first_four_last_hidden_states.shape[0],
                first_four_last_hidden_states.shape[1] *
                first_four_last_hidden_states.shape[2])
            final_output = self.dropout_layer(first_four_last_hidden_states)
        return final_output

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        with torch.no_grad():
            final_output = self.calculate(input_ids, attention_mask, token_type_ids)
            if self.pooler_usage:
                logits = self.output_layer(final_output)
            else:
                logits = self.output_layer_cat(final_output)
            return logits

    def compute_loss(self, input_ids, attention_mask, token_type_ids, label):
        target = label

        final_output = self.calculate(input_ids, attention_mask, token_type_ids)
        if self.pooler_usage:
            logits = self.output_layer(final_output)
        else:
            logits = self.output_layer_cat(final_output)

        class_weights = torch.FloatTensor(self.class_weights).to(self._device)
        loss = F.cross_entropy(logits, target, weight=class_weights)

        predicted_value = torch.max(logits, 1)[1]
        predicted_list = predicted_value.cpu().numpy().tolist()
        target_list = target.cpu().numpy().tolist()

        return loss, predicted_list, target_list


if __name__ == '__main__':
    from transformers.configuration_bert import BertConfig

    config = BertConfig.from_pretrained("bert-base-multilingual-uncased",
                                   cache_dir="../resources/cache_model")
    config = config.to_dict()
    config.update({"weight_class": [1, 1]})
    config = BertConfig.from_dict(config)
    model = QaBERT.from_pretrained("bert-base-multilingual-uncased",
                                   cache_dir="../resources/cache_model", config=config)
