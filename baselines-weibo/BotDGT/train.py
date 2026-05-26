from datetime import datetime
from copy import deepcopy
from pathlib import Path
import torch
from pytorch_lightning import seed_everything
from config import get_train_args
from models.model import BotDyGNN
from utils.dataset import Dataset
from utils.loss import all_snapshots_loss
from utils.metrics import compute_metrics_one_snapshot, is_better, null_metrics

class Trainer:
    def __init__(self, args):
        self.args = args
        self.dataset = Dataset(
            args.dataset_name, args.interval, args.batch_size,
            args.seed, args.window_size, args.device,
        )

        ds = self.dataset
        self.des_tensor    = ds.des_tensor
        self.tweets_tensor = ds.tweets_tensor
        self.num_prop      = ds.num_prop
        self.category_prop = ds.category_prop
        self.labels        = ds.labels

        self.train_right   = ds.train_right
        self.train_n_id    = ds.train_n_id
        self.train_edge_index = ds.train_edge_index
        self.train_edge_type  = ds.train_edge_type
        self.train_exist_nodes = ds.train_exist_nodes
        self.train_clustering_coefficient    = ds.train_clustering_coefficient
        self.train_bidirectional_links_ratio = ds.train_bidirectional_links_ratio

        self.val_right    = ds.val_right
        self.val_n_id     = ds.val_n_id
        self.val_edge_index = ds.val_edge_index
        self.val_edge_type  = ds.val_edge_type
        self.val_exist_nodes = ds.val_exist_nodes
        self.val_clustering_coefficient    = ds.val_clustering_coefficient
        self.val_bidirectional_links_ratio = ds.val_bidirectional_links_ratio

        self.test_right   = ds.test_right
        self.test_n_id    = ds.test_n_id
        self.test_edge_index = ds.test_edge_index
        self.test_edge_type  = ds.test_edge_type
        self.test_exist_nodes = ds.test_exist_nodes
        self.test_clustering_coefficient    = ds.test_clustering_coefficient
        self.test_bidirectional_links_ratio = ds.test_bidirectional_links_ratio

        self.has_ood = len(ds.ood_right) > 0
        if self.has_ood:
            self.ood_right   = ds.ood_right
            self.ood_n_id    = ds.ood_n_id
            self.ood_edge_index = ds.ood_edge_index
            self.ood_edge_type  = ds.ood_edge_type
            self.ood_exist_nodes = ds.ood_exist_nodes
            self.ood_clustering_coefficient    = ds.ood_clustering_coefficient
            self.ood_bidirectional_links_ratio = ds.ood_bidirectional_links_ratio

        # 更新 window_size（Dataset 内部可能已截断）
        self.args.window_size = ds.window_size

        # 根据训练集标签分布，可选地对 loss 做类别加权
        if args.weighted_loss:
            train_labels_cpu = ds.labels[ds.train_idx.cpu()].cpu()
            counts = torch.bincount(train_labels_cpu, minlength=2).float()
            class_weight = counts.sum() / (counts.clamp_min(1.0) * 2)
            class_weight = class_weight.to(args.device)
            print(f"[weighted_loss] class weights: human={class_weight[0]:.4f}, bot={class_weight[1]:.4f}")
            self.criterion = torch.nn.CrossEntropyLoss(reduction='mean', weight=class_weight)
        else:
            self.criterion = torch.nn.CrossEntropyLoss(reduction='mean')

        self.model = BotDyGNN(self.args)
        self.model.to(self.args.device)

        params = [
            {'params': self.model.node_feature_embedding_layer.parameters(),
             'lr': self.args.structural_learning_rate},
            {'params': self.model.structural_layer.parameters(),
             'lr': self.args.structural_learning_rate},
            {'params': self.model.temporal_layer.parameters(),
             'lr': self.args.temporal_learning_rate},
        ]
        self.optimizer = torch.optim.AdamW(params, weight_decay=self.args.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=20, eta_min=0)

        self.pbar = range(self.args.epoch)
        self.best_val_metrics    = null_metrics()
        self.best_val_epoch      = -1
        self.best_checkpoint_path = Path('output') / self.args.dataset_name / 'checkpoints' / (
            f"best_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
        )
        self.best_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        self.best_state_dict     = None
        self.test_state_dict_list = []
        self.test_epoch_list     = []
        self.test_metrics        = null_metrics()
        self.test_state_dict     = None
        self.last_state_dict     = None

    # ----------------------------------------------------------
    # 单 batch 前向
    # ----------------------------------------------------------

    def forward_one_batch(self, batch_size, batch_n_id, batch_edge_index,
                          batch_exist_nodes, batch_clustering_coefficient,
                          batch_bidirectional_links_ratio,
                          override_labels=None):
        des_list   = [self.des_tensor[n_id].to(self.args.device)    for n_id in batch_n_id]
        tweet_list = [self.tweets_tensor[n_id].to(self.args.device) for n_id in batch_n_id]
        num_list   = [self.num_prop[n_id].to(self.args.device)      for n_id in batch_n_id]
        cat_list   = [self.category_prop[n_id].to(self.args.device) for n_id in batch_n_id]
        edge_list  = [e.to(self.args.device) for e in batch_edge_index]
        cc_list    = [c.to(self.args.device) for c in batch_clustering_coefficient]
        bi_list    = [b.to(self.args.device) for b in batch_bidirectional_links_ratio]

        exist_nodes_list = torch.stack(
            [en[:batch_size].to(self.args.device) for en in batch_exist_nodes], dim=0)

        if override_labels is not None:
            # OOD 评测：ground truth 全为 1（机器人）
            label_list = override_labels.unsqueeze(0).expand(
                len(batch_n_id), -1)           # [T, batch_size]
            label_list = label_list             # keep on device
        else:
            label_list = torch.stack(
                [self.labels[n_id][:batch_size].to(self.args.device) for n_id in batch_n_id],
                dim=0)

        output = self.model(
            des_list, tweet_list, num_list, cat_list,
            edge_list, cc_list, bi_list,
            exist_nodes_list, batch_size,
        )
        output = output.transpose(0, 1)  # [T, batch_size, 2]
        loss = all_snapshots_loss(
            self.criterion, output, label_list,
            exist_nodes_list, self.args.coefficient)
        return output, loss, label_list, exist_nodes_list

    # ----------------------------------------------------------
    # 单 epoch 前向（用于 val / test / ood）
    # ----------------------------------------------------------

    @torch.no_grad()
    def forward_one_epoch(self, right, n_id, edge_index,
                          exist_nodes, clustering_coefficient,
                          bidirectional_links_ratio,
                          override_all_labels=None):
        all_output = []
        all_label  = []
        all_exist  = []
        total_loss = 0.0

        for i, (batch_size, bn_id, bei, ben, bcc, bbi) in enumerate(zip(
                right, n_id, edge_index,
                exist_nodes, clustering_coefficient, bidirectional_links_ratio)):
            if override_all_labels is not None:
                ov_lbl = torch.ones(batch_size, dtype=torch.long, device=self.args.device)
            else:
                ov_lbl = None
            output, loss, label_list, exist_list = self.forward_one_batch(
                batch_size, bn_id, bei, ben, bcc, bbi,
                override_labels=ov_lbl,
            )
            total_loss += loss.item() / max(self.args.window_size, 1) / max(len(right), 1)
            all_output.append(output)
            all_label.append(label_list)
            all_exist.append(exist_list)

        all_output = torch.cat(all_output, dim=1)
        all_label  = torch.cat(all_label,  dim=1)
        all_exist  = torch.cat(all_exist,  dim=1)
        metrics = compute_metrics_one_snapshot(
            all_label[-1], all_output[-1], exist_nodes=all_exist[-1])
        metrics['loss'] = total_loss
        return metrics

    # ----------------------------------------------------------
    # 训练
    # ----------------------------------------------------------

    def train_per_epoch(self, current_epoch):
        self.model.train()
        all_output = []
        all_label  = []
        all_exist  = []
        total_loss = 0.0

        for batch_size, bn_id, bei, ben, bcc, bbi in zip(
                self.train_right, self.train_n_id, self.train_edge_index,
                self.train_exist_nodes,
                self.train_clustering_coefficient,
                self.train_bidirectional_links_ratio):
            self.optimizer.zero_grad()
            output, loss, label_list, exist_list = self.forward_one_batch(
                batch_size, bn_id, bei, ben, bcc, bbi)
            total_loss += loss.item() / max(self.args.window_size, 1) / max(len(self.train_right), 1)
            loss.backward()
            self.optimizer.step()
            all_output.append(output)
            all_label.append(label_list)
            all_exist.append(exist_list)

        all_output = torch.cat(all_output, dim=1)
        all_label  = torch.cat(all_label,  dim=1)
        all_exist  = torch.cat(all_exist,  dim=1)
        metrics = compute_metrics_one_snapshot(
            all_label[-1], all_output[-1], exist_nodes=all_exist[-1])
        plog = f'Epoch-{current_epoch} train loss: {total_loss:.6f}'
        for key in ('accuracy', 'precision', 'recall', 'f1'):
            plog += f'  {key}: {metrics[key]:.6f}'
        print(plog)
        metrics['loss'] = total_loss
        return metrics

    @torch.no_grad()
    def val_per_epoch(self, current_epoch):
        self.model.eval()
        metrics = self.forward_one_epoch(
            self.val_right, self.val_n_id, self.val_edge_index,
            self.val_exist_nodes,
            self.val_clustering_coefficient,
            self.val_bidirectional_links_ratio,
        )
        plog = f'Epoch-{current_epoch} val   loss: {metrics["loss"]:.6f}'
        for key in ('accuracy', 'precision', 'recall', 'f1'):
            plog += f'  {key}: {metrics[key]:.6f}'
        print(plog)
        return metrics

    @torch.no_grad()
    def _run_test(self, state_dict, label='test'):
        self.model.load_state_dict(state_dict)
        self.model.eval()
        metrics = self.forward_one_epoch(
            self.test_right, self.test_n_id, self.test_edge_index,
            self.test_exist_nodes,
            self.test_clustering_coefficient,
            self.test_bidirectional_links_ratio,
        )
        plog = f'[{label}] loss: {metrics["loss"]:.6f}'
        for key in ('accuracy', 'precision', 'recall', 'f1'):
            plog += f'  {key}: {metrics[key]:.6f}'
        print(plog)
        return metrics

    @torch.no_grad()
    def ood_eval(self, state_dict):
        """在 Agent 账户上进行分布外检测，ground truth 全部视为 1（机器人）"""
        if not self.has_ood:
            print('无 OOD 数据，跳过 OOD 评测')
            return None
        self.model.load_state_dict(state_dict)
        self.model.eval()
        # override_all_labels=True 触发 ground truth 全为 1 的逻辑
        metrics = self.forward_one_epoch(
            self.ood_right, self.ood_n_id, self.ood_edge_index,
            self.ood_exist_nodes,
            self.ood_clustering_coefficient,
            self.ood_bidirectional_links_ratio,
            override_all_labels=True,
        )
        plog = f'[OOD/Agent] loss: {metrics["loss"]:.6f}'
        for key in ('accuracy', 'precision', 'recall', 'f1'):
            plog += f'  {key}: {metrics[key]:.6f}'
        print(plog)
        return metrics

    # ----------------------------------------------------------
    # 主训练循环
    # ----------------------------------------------------------

    def train(self):
        no_improve_count = 0

        for current_epoch in self.pbar:
            self.train_per_epoch(current_epoch)
            self.scheduler.step()
            val_metrics = self.val_per_epoch(current_epoch)

            if is_better(val_metrics, self.best_val_metrics):
                self.best_val_metrics = val_metrics
                self.best_val_epoch = current_epoch
                self.best_state_dict = deepcopy(self.model.state_dict())
                torch.save(
                    {
                        'epoch': current_epoch,
                        'val_metrics': val_metrics,
                        'model_state_dict': self.best_state_dict,
                        'weighted_loss': self.args.weighted_loss,
                    },
                    self.best_checkpoint_path,
                )
                print(
                    f"Saved best checkpoint to {self.best_checkpoint_path} "
                    f"(epoch={current_epoch}, val_f1={val_metrics['f1']:.4f})"
                )
                self.test_epoch_list.append(current_epoch)
                self.test_state_dict_list.append(deepcopy(self.model.state_dict()))
                no_improve_count = 0
            else:
                no_improve_count += 1

            self.last_state_dict = deepcopy(self.model.state_dict())

            if self.args.early_stop and no_improve_count >= self.args.patience:
                print(f'Early stopping at epoch {current_epoch}')
                break

        # 直接加载验证集表现最好的 checkpoint 做测试
        if self.best_checkpoint_path.exists():
            checkpoint = torch.load(self.best_checkpoint_path, map_location=self.args.device)
            self.test_state_dict = checkpoint['model_state_dict']
            self.test_metrics = self._run_test(
                self.test_state_dict,
                label=f"best-epoch-{checkpoint.get('epoch', self.best_val_epoch)}",
            )
        else:
            self.test_state_dict = self.last_state_dict
            self.test_metrics = self._run_test(self.test_state_dict, label='last-epoch')

        # OOD 评测
        self.ood_eval(self.test_state_dict)

        print(f'\n最佳 checkpoint: {self.best_checkpoint_path}')


def main(args):
    seed_everything(args.seed)
    trainer = Trainer(args)
    trainer.train()


if __name__ == '__main__':
    args = get_train_args()
    print(args)
    main(args)
