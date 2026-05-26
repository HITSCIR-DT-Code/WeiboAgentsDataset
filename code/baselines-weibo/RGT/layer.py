import torch
from torch_geometric.nn import TransformerConv


def masked_edge_index(edge_index, edge_mask):
    return edge_index[:, edge_mask]


class SemanticAttention(torch.nn.Module):
    def __init__(self, in_channel, num_head, hidden_size=128):
        super(SemanticAttention, self).__init__()

        self.num_head = num_head
        self.att_layers = torch.nn.ModuleList()
        for i in range(num_head):
            self.att_layers.append(
                torch.nn.Sequential(
                    torch.nn.Linear(in_channel, hidden_size),
                    torch.nn.Tanh(),
                    torch.nn.Linear(hidden_size, 1, bias=False),
                )
            )

    def forward(self, z):
        w = self.att_layers[0](z).mean(0)
        beta = torch.softmax(w, dim=0)
        beta = beta.expand((z.shape[0],) + beta.shape)
        output = (beta * z).sum(1)

        for i in range(1, self.num_head):
            w = self.att_layers[i](z).mean(0)
            beta = torch.softmax(w, dim=0)
            beta = beta.expand((z.shape[0],) + beta.shape)
            output += (beta * z).sum(1)

        return output / self.num_head


class RGTLayer(torch.nn.Module):
    def __init__(self, num_edge_type, in_channel, out_channel, trans_heads, semantic_head, dropout):
        super(RGTLayer, self).__init__()
        self.gate = torch.nn.Sequential(
            torch.nn.Linear(in_channel + out_channel, in_channel),
            torch.nn.Sigmoid(),
        )

        self.transformer_list = torch.nn.ModuleList()
        for i in range(int(num_edge_type)):
            self.transformer_list.append(
                TransformerConv(
                    in_channels=in_channel,
                    out_channels=out_channel,
                    heads=trans_heads,
                    dropout=dropout,
                    concat=False,
                )
            )

        self.num_edge_type = num_edge_type
        self.semantic_attention = SemanticAttention(in_channel=out_channel, num_head=semantic_head)

    def forward(self, features, edge_index, edge_type):
        """
        features  : [N, in_channel]
        edge_index: [2, E]
        edge_type : [E]  — 整型，值域 [0, num_edge_type)
        """
        edge_index_list = []
        for i in range(self.num_edge_type):
            tmp = masked_edge_index(edge_index, edge_type == i)
            edge_index_list.append(tmp)

        u = self.transformer_list[0](features, edge_index_list[0]).flatten(1)
        a = self.gate(torch.cat((u, features), dim=1))
        semantic_embeddings = (
            torch.mul(torch.tanh(u), a) + torch.mul(features, (1 - a))
        ).unsqueeze(1)  # [N, 1, out_channel]

        for i in range(1, len(edge_index_list)):
            u = self.transformer_list[i](features, edge_index_list[i]).flatten(1)
            a = self.gate(torch.cat((u, features), dim=1))
            output = torch.mul(torch.tanh(u), a) + torch.mul(features, (1 - a))
            semantic_embeddings = torch.cat(
                (semantic_embeddings, output.unsqueeze(1)), dim=1
            )  # [N, i+1, out_channel]

        # Bug fix: return 移到循环外，所有边类型处理完后再做语义注意力聚合
        return self.semantic_attention(semantic_embeddings)
