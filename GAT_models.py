from torch_geometric.nn import GATConv
import math
import torch
import torch.nn.functional as F


def glorot(tensor):
    stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
    if tensor is not None:
        tensor.data.uniform_(-stdv, stdv)


class Net(torch.nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()

        self.lin1 = torch.nn.Linear(args.input_size, args.hidden_size)
        self.convs = torch.nn.ModuleList()
        for i in range(args.stack_layer_num):
            self.convs.append(
                GATConv(args.hidden_size, args.hidden_size // args.heads, heads=args.heads, dropout=args.att_dropout))

        self.lin3 = torch.nn.Linear(args.hidden_size * 2, args.num_classes)
        # concatenate two object embeds together

        self.rnn = torch.nn.LSTM(args.hidden_size, args.hidden_size, 1)
        glorot(self.lin3.weight)
        glorot(self.lin1.weight)

    def forward(self, args, x, target_mask, edge_index, label_ids=None):
        if args.embed_dropout > 0:
            x = F.dropout(x, p=args.embed_dropout, training=self.training)
        x = self.lin1(x)

        for i in range(args.stack_layer_num):
            x = F.elu(self.convs[i](x, edge_index))
        #endfor

        batch_size = len(target_mask)

        concat_two_embed_list = []
        for i in range(batch_size):
            t_mask_i = target_mask[i]
            two_target_embed = x[t_mask_i]
            concat_two_embed = two_target_embed.flatten()
            concat_two_embed_list.append(concat_two_embed)
        # endfor

        concat_two_embed_list = torch.stack(concat_two_embed_list)

        logits = self.lin3(concat_two_embed_list)

        if label_ids is not None:
            loss_op = torch.nn.CrossEntropyLoss()
            loss = loss_op(logits.view(-1, args.num_classes), label_ids)
            return logits, loss
        else:
            return logits, None
        # endif