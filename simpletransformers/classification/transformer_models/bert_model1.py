import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss
from transformers.models.bert.modeling_bert import BertModel, BertPreTrainedModel
import torch.nn.functional as F

class BertForSequenceClassification(BertPreTrainedModel):
    r"""
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for computing the sequence classification/regression loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.
            If ``config.num_labels == 1`` a regression loss is computed (Mean-Square loss),
            If ``config.num_labels > 1`` a classification loss is computed (Cross-Entropy).
    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Classification (or regression if config.num_labels==1) loss.
        **logits**: ``torch.FloatTensor`` of shape ``(batch_size, config.num_labels)``
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    Examples::
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, logits = outputs[:2]
    """  # noqa: ignore flake8"

    def __init__(self, config, weight=None):
        super(BertForSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels
        #print(config)
        self.bert = BertModel(config)
        #self.bert2 = BertModel(config)
        #self.bert2.load_state_dict(self.bert.state_dict())
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier1 = nn.Linear((config.hidden_size*2)+6, 512)
        self.classifier2 = nn.Linear(512, self.config.num_labels)
        self.weight = weight
        self.init_weights()
        #self.freeze()

    def forward(
        self,
        input_ids1=None,#
        attention_mask1=None,#
        token_type_ids1=None,#
        input_ids2=None,
        attention_mask2=None,
        token_type_ids2=None,
        ofs1=None,
        pro1=None,
        ofs2=None,
        pro2=None,
        ofs3=None,
        pro3=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,#
    ):

        outputs1 = self.bert(
            input_ids1,
            attention_mask=attention_mask1,
            token_type_ids=token_type_ids1,
            position_ids=position_ids,
            head_mask=head_mask,
        )
        outputs2 = self.bert(
            input_ids2,
            attention_mask=attention_mask2,
            token_type_ids=token_type_ids2,
            position_ids=position_ids,
            head_mask=head_mask,
        )

        # Complains if input_embeds is kept

        pooled_output1 = outputs1[1]
        pooled_output1 = self.dropout(pooled_output1)
        #print(f'shape of pooled text embedding: {pooled_output1.shape}')

        pooled_output2 = outputs2[1]
        pooled_output2 = self.dropout(pooled_output2)
        #print(f'shape of pooled main heading embedding: {pooled_output2.shape}')


        #print(f'shape of offset: {ofs.shape}')
        ofs1 = ofs1.unsqueeze(1)#.type(torch.float)
        pro1 = pro1.unsqueeze(1)
        ofs2 = ofs2.unsqueeze(1)
        pro2 = pro2.unsqueeze(1)
        ofs3 = ofs3.unsqueeze(1)
        pro3 = pro3.unsqueeze(1)
        pooled_output = torch.cat(
            [pooled_output1, pooled_output2, ofs1, pro1, ofs2, pro2, ofs3, pro3], dim=1)
        #print(f'shape of concat feauture: {pooled_output.shape}')

        logits = self.classifier2(F.relu(self.classifier1(pooled_output)))
        #print(f'shape of logits: {logits.shape}')
        outputs = (logits,) + outputs1[2:]  # add hidden states and attention if they are here
        # 这里暂且把outputs改成outputs1
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                if self.weight is not None:
                    weight = self.weight.to(labels.device)
                else:
                    weight = None
                loss_fct = CrossEntropyLoss(weight=weight)
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)
    
    def freeze(self):
        unfreeze_layers = ['layer.9', 'layer.10', 'layer.11',
                           'bert.pooler', 'classifier1.', 'classifier2.']
        for name, param in self.named_parameters():
            param.requires_grad = False
            for ele in unfreeze_layers:
                if ele in name:
                    param.requires_grad = True
                    break
