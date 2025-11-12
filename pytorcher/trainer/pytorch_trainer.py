import torch
import abc
import mlflow

class PytorchTrainer:

    def __init__(self, metrics=[]):

        # Create device for training (GPU if available)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Using device: {self.device}')

        # create dataset / loader / model / optimizer / objective / metrics
        self.loader_train, self.loader_val = self.create_data_loader()
        self.model = self.create_model()
        self.signature = self.get_signature()
        self.optimizer = self.get_optimizer(learning_rate=self.learning_rate)
        self.objective = self.get_objective()
        self.metrics = self.get_metrics(metrics)

        #
        self.initial_epoch = 0

    @abc.abstractmethod
    def create_data_loader(self):
        """
        self.dataset = SomeDataset(params...)
        loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=self.shuffle)

        return loader
        """
        pass

    def get_signature(self):
        sample = self.dataset.__getitem__(0)[0]
        signature = (1, ) + tuple(sample.shape)
        return signature

    @abc.abstractmethod
    def create_model(self):
        """
        model = someModel(params...)
        model = model.to(self.device)
        return model
        """
        pass

    def get_optimizer(self, learning_rate=1e-3):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        return optimizer

    @abc.abstractmethod
    def get_objective(self):
        """
        objective = someLossFunction()
        return objective
        """
        pass

    def get_metrics(self, metrics):
        out = []
        for metric in metrics:
            metric_name, metric_config = metric
            metric_class = getattr(__import__('pytorcher.metrics', fromlist=[metric_name]), metric_name)
            out.append(metric_class(**metric_config))
        if 'loss' not in [m.name for m in out]:
            out.append(getattr(__import__('pytorcher.metrics', fromlist=[metric_name]), 'Mean')(**{'name': 'loss'}))
        return out

    def on_epoch_end(self, epoch):
        for metric in self.metrics:
            metric.reset_states()

    def update_metrics(self, loss, y_true, y_pred):
        for metric in self.metrics:
            if metric.name == 'loss':
                metric.update_state(None, loss)
            else:
                metric.update_state(y_true, y_pred)        

    def fit(self):

        for epoch in range(self.initial_epoch, self.n_epochs):
            self.model.train()
            epoch_loss = 0.0
            for batch_idx, (inputs, targets) in enumerate(self.loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.objective(outputs, targets)
                loss.backward()
                self.optimizer.step()

                self.update_metrics(loss.item(), targets, outputs)

            print(f'Epoch {epoch+1}, metrics: ')
            for metric in self.metrics:
                print(f'{metric.name}: {metric.result():.4f}')
                metric.reset_states()

            #
            self.on_epoch_end(epoch)