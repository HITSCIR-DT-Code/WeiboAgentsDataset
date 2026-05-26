import os
import torch
from torch_geometric.loader import NeighborLoader


def load_graphs(dataset_name, interval):
    interval_dict = {
        'year': 12, 'month': 1, 'three_months': 3, 'six_months': 6,
        '9_months': 9, '15_months': 15, '18_months': 18,
        '21_months': 21, '24_months': 24,
    }
    assert interval in interval_dict, f"Unknown interval: {interval}"
    step = interval_dict[interval]

    graph_dir = os.path.join('./data', dataset_name, 'graph_data', 'graphs')
    files = sorted(os.listdir(graph_dir))
    # 按步长从末尾向前采样，再反转使时间升序
    file_name = []
    for index in range(-1, -len(files) - 1, -step):
        file_name.append(files[index])
    file_name.reverse()
    print(f"加载 {len(file_name)} 个快照: {file_name[0]} → {file_name[-1]}")

    graph_list = [
        torch.load(os.path.join(graph_dir, f), weights_only=False)
        for f in file_name
    ]
    return graph_list, file_name


def load_split_index(dataset_name):
    base = os.path.join('./data', dataset_name, 'processed_data')
    train_idx = torch.load(os.path.join(base, 'train_idx.pt'), weights_only=True)
    val_idx   = torch.load(os.path.join(base, 'val_idx.pt'),   weights_only=True)
    test_idx  = torch.load(os.path.join(base, 'test_idx.pt'),  weights_only=True)
    return train_idx, val_idx, test_idx


def load_ood_index(dataset_name):
    path = os.path.join('./data', dataset_name, 'processed_data', 'ood_idx.pt')
    if os.path.exists(path):
        return torch.load(path, weights_only=True)
    return torch.tensor([], dtype=torch.long)


def load_labels(dataset_name):
    path = os.path.join('./data', dataset_name, 'processed_data', 'label.pt')
    return torch.load(path, weights_only=True)


def load_features(dataset_name):
    base = os.path.join('./data', dataset_name, 'processed_data')
    des_tensor    = torch.load(os.path.join(base, 'des_tensor.pt'),            weights_only=True)
    tweets_tensor = torch.load(os.path.join(base, 'tweets_tensor.pt'),        weights_only=True)
    num_prop      = torch.load(os.path.join(base, 'num_properties_tensor.pt'), weights_only=True)
    category_prop = torch.load(os.path.join(base, 'cat_properties_tensor.pt'), weights_only=True)
    return des_tensor, tweets_tensor, num_prop, category_prop


class Dataset:
    def __init__(self, dataset_name, interval, batch_size, seed, window_size, device):
        super().__init__()
        self.dataset_name = dataset_name
        self.interval     = interval
        self.batch_size   = batch_size
        self.seed         = seed
        self.window_size  = window_size
        self.device       = device

        self.graphs, self.graphs_file_name_list = load_graphs(dataset_name, interval)

        self.train_idx, self.val_idx, self.test_idx = load_split_index(dataset_name)
        self.ood_idx   = load_ood_index(dataset_name)
        self.labels    = load_labels(dataset_name)
        self.des_tensor, self.tweets_tensor, self.num_prop, self.category_prop = load_features(dataset_name)

        self.graphs       = [graph.to(self.device) for graph in self.graphs]
        self.train_idx    = self.train_idx.to(self.device)
        self.val_idx      = self.val_idx.to(self.device)
        self.test_idx     = self.test_idx.to(self.device)
        self.ood_idx      = self.ood_idx.to(self.device)
        self.labels       = self.labels.to(self.device)
        self.des_tensor   = self.des_tensor.to(self.device)
        self.tweets_tensor = self.tweets_tensor.to(self.device)
        self.num_prop     = self.num_prop.to(self.device)
        self.category_prop = self.category_prop.to(self.device)

        self.train_right, self.train_n_id, self.train_edge_index, self.train_edge_type, \
            self.train_exist_nodes, self.train_clustering_coefficient, \
            self.train_bidirectional_links_ratio = self.get_final_data('train')

        self.val_right, self.val_n_id, self.val_edge_index, self.val_edge_type, \
            self.val_exist_nodes, self.val_clustering_coefficient, \
            self.val_bidirectional_links_ratio = self.get_final_data('val')

        self.test_right, self.test_n_id, self.test_edge_index, self.test_edge_type, \
            self.test_exist_nodes, self.test_clustering_coefficient, \
            self.test_bidirectional_links_ratio = self.get_final_data('test')

        if len(self.ood_idx) > 0:
            self.ood_right, self.ood_n_id, self.ood_edge_index, self.ood_edge_type, \
                self.ood_exist_nodes, self.ood_clustering_coefficient, \
                self.ood_bidirectional_links_ratio = self.get_final_data('ood')
        else:
            self.ood_right = []

        if len(self.graphs) > self.window_size and self.window_size != -1:
            print(f'window_size={self.window_size}，截取最近 {self.window_size} 个快照')
            self.get_window_data()
        else:
            print('使用全部快照')
            self.window_size = len(self.graphs)

    def get_window_data(self):
        attrs = [
            'train_n_id', 'train_edge_index', 'train_edge_type',
            'train_exist_nodes', 'train_clustering_coefficient', 'train_bidirectional_links_ratio',
            'val_n_id',   'val_edge_index',   'val_edge_type',
            'val_exist_nodes',   'val_clustering_coefficient',   'val_bidirectional_links_ratio',
            'test_n_id',  'test_edge_index',  'test_edge_type',
            'test_exist_nodes',  'test_clustering_coefficient',  'test_bidirectional_links_ratio',
            'ood_n_id',   'ood_edge_index',   'ood_edge_type',
            'ood_exist_nodes',   'ood_clustering_coefficient',   'ood_bidirectional_links_ratio',
        ]
        for attr in attrs:
            if hasattr(self, attr) and getattr(self, attr):
                setattr(self, attr, [_[-self.window_size:] for _ in getattr(self, attr)])

    def get_final_data(self, split_type):
        if split_type == 'train':
            input_nodes = self.train_idx
            shuffle = True
        elif split_type == 'val':
            input_nodes = self.val_idx
            shuffle = False
        elif split_type == 'test':
            input_nodes = self.test_idx
            shuffle = False
        elif split_type == 'ood':
            input_nodes = self.ood_idx
            shuffle = False
        else:
            raise ValueError(f'Unknown split_type: {split_type}')

        print(f'构建 {split_type} 数据集 ({self.dataset_name})...')

        dir_path = os.path.join(
            './data', self.dataset_name, 'final_data',
            self.interval,
            f'batch-size-{self.batch_size}',
            f'seed-{self.seed}',
            split_type
        )
        os.makedirs(dir_path, exist_ok=True)

        file_names = [
            'all_right', 'all_n_id', 'all_edge_index', 'all_edge_type',
            'all_exist_nodes', 'all_clustering_coefficient', 'all_bidirectional_links_ratio',
        ]
        data_dict = {
            name: {'path': os.path.join(dir_path, f'{name}.pt'), 'data': []}
            for name in file_names
        }

        if all(os.path.exists(data_dict[name]['path']) for name in data_dict):
            for name in data_dict:
                data_dict[name]['data'] = torch.load(
                    data_dict[name]['path'], weights_only=False)
        else:
            loader = _DataLoader(
                graphs=self.graphs,
                input_nodes=input_nodes,
                batch_size=self.batch_size,
                shuffle=shuffle,
                seed=self.seed,
                split_type=split_type,
            )
            total = len(input_nodes)
            for i in range(0, total, self.batch_size):
                right = min(self.batch_size, total - i)
                data_dict['all_right']['data'].append(right)
                subgraph_list = loader.iterate()
                data_dict['all_n_id']['data'].append(
                    [sg.n_id.to('cpu') for sg in subgraph_list])
                data_dict['all_edge_index']['data'].append(
                    [sg.edge_index.to('cpu') for sg in subgraph_list])
                data_dict['all_edge_type']['data'].append(
                    [sg.edge_type.to('cpu') for sg in subgraph_list])
                data_dict['all_exist_nodes']['data'].append(
                    [sg.exist_nodes.to('cpu') for sg in subgraph_list])
                data_dict['all_clustering_coefficient']['data'].append(
                    [sg.clustering_coefficient.to('cpu') for sg in subgraph_list])
                data_dict['all_bidirectional_links_ratio']['data'].append(
                    [sg.bidirectional_links_ratio.to('cpu') for sg in subgraph_list])
            for name in data_dict:
                torch.save(data_dict[name]['data'], data_dict[name]['path'])

        return (
            data_dict['all_right']['data'],
            data_dict['all_n_id']['data'],
            data_dict['all_edge_index']['data'],
            data_dict['all_edge_type']['data'],
            data_dict['all_exist_nodes']['data'],
            data_dict['all_clustering_coefficient']['data'],
            data_dict['all_bidirectional_links_ratio']['data'],
        )


class _DataLoader:
    def __init__(self, graphs, input_nodes, seed, batch_size, shuffle, split_type):
        num_neighbors = [2560] * 2 if split_type == 'train' else [-1] * 2
        self.loader_list = [
            NeighborLoader(
                graph,
                shuffle=shuffle,
                generator=torch.Generator().manual_seed(seed),
                batch_size=batch_size,
                input_nodes=input_nodes,
                num_neighbors=num_neighbors,
            )
            for graph in graphs
        ]
        self.iter_list = [iter(loader) for loader in self.loader_list]

    def iterate(self):
        return [next(it) for it in self.iter_list]
