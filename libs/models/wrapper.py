import numpy as np
import torch
import torch.nn as nn
import os
from tqdm import tqdm
from torch.utils.data import DataLoader

# Relative imports based on your new structure
from ..datasets.loader import SupervisedDatasets
from ..models.WER import SL_model_flex
from ..models.DCNN import SimpleMultiStreamCNN


class DeepLearningClassifier:
    def __init__(self,
                 clf_name,
                 feature_modalities,
                 dl_general_cfg,
                 dl_model_cfg,
                 dl_train_cfg,
                 dl_test_cfg,
                 dl_opt_cfg,
                 input_dim,
                 n_classes):
        """
        Wrapper to make PyTorch models behave like Scikit-Learn models.

        Args:
            clf_name (str): 'WER' or 'DCNN'
            input_dim (int): Number of input modalities (features)
            n_classes (int): Number of output classes
        """
        self.clf_name = clf_name
        self.feature_modalities = feature_modalities
        self.dl_general_cfg = dl_general_cfg
        self.dl_model_cfg = dl_model_cfg
        self.dl_train_cfg = dl_train_cfg
        self.dl_test_cfg = dl_test_cfg
        self.dl_opt_cfg = dl_opt_cfg
        self.input_dim = input_dim
        self.n_classes = n_classes
        self.device = torch.device(dl_general_cfg['device'] if torch.cuda.is_available() else 'cpu')

        # Name extension logic for saving files (reproduced from original)
        self.name_ext = self._build_name_extension()

        # Initialize the Network Architecture
        if self.clf_name == 'WER':
            self.model = SL_model_flex(
                num_classes=n_classes,
                num_modalities=input_dim,
                tcn_nfilters=self.dl_model_cfg['tcn_nfilters'],
                tcn_kernel_size=self.dl_model_cfg['tcn_kernel_size'],
                tcn_dropout=self.dl_model_cfg['tcn_dropout'],
                trans_d_model=self.dl_model_cfg['trans_d_model'],
                trans_n_heads=self.dl_model_cfg['trans_n_heads'],
                trans_num_layers=self.dl_model_cfg['trans_num_layers'],
                trans_dim_feedforward=self.dl_model_cfg['trans_dim_feedforward'],
                shared_embed_dim=self.dl_model_cfg['shared_embed_dim'],
                trans_dropout=self.dl_model_cfg['trans_dropout'],
                trans_activation=self.dl_model_cfg['trans_activation'],
                trans_norm=self.dl_model_cfg['trans_norm'],
                trans_freeze=False,
                sl_embed_dim1=self.dl_model_cfg['sl_embed_dim1'],
                sl_activation=self.dl_model_cfg['sl_activation'],
                sl_dropout=self.dl_model_cfg['sl_dropout']
            )
        elif self.clf_name == 'DCNN':
            self.model = SimpleMultiStreamCNN(
                num_modalities=input_dim,
                conv_channels=self.dl_model_cfg['conv_channels'],
                kernel_sizes=self.dl_model_cfg['kernel_sizes'],
                pool_sizes=self.dl_model_cfg['pool_sizes'],
                mlp_hidden=self.dl_model_cfg['mlp_hidden'],
                num_classes=n_classes,
                dropout=self.dl_model_cfg['dropout'],
            )
        else:
            raise ValueError(f"Unknown Deep Learning Classifier: {self.clf_name}")

        self.model.to(self.device)

        # loss function
        self.criterion = nn.CrossEntropyLoss().to(self.device)

        # optimizer
        if self.dl_opt_cfg['opt_name'] == 'sgd':
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.dl_train_cfg['lr'],
                momentum=self.dl_opt_cfg['momentum'],
                weight_decay=self.dl_opt_cfg['weight_decay']
            )
        elif self.dl_opt_cfg['opt_name'] == 'adam':
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.dl_train_cfg['lr']
            )
        else:
            raise ValueError(f"Optimizer {self.dl_opt_cfg['opt_name']} is not supported")

        # Placeholders for training state
        self.best_epoch = None
        self.min_valid_loss = None
        self.scheduler = None

    def _build_name_extension(self):
        ext = "_".join(self.feature_modalities)
        if self.dl_general_cfg.get('name_extension', '') == '':
            ext += '/noext'
        else:
            ext += f"/{self.dl_general_cfg['name_extension']}"
        return ext

    def _init_scheduler(self, len_train_loader):
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.dl_train_cfg['lr'],
            epochs=self.dl_train_cfg['num_epochs'],
            steps_per_epoch=len_train_loader
        )

    def fit(self, X_train, y_train, X_valid=None, y_valid=None,
            cv_split_index=0, affect_subdomain=None, prediction_target=None):
        """
        Training loop with validation and model saving.

        Args:
            X_train: Training features
            y_train: Training labels
            X_valid: Validation features (optional)
            y_valid: Validation labels (optional)
            cv_split_index: Current CV fold (for file naming)
            affect_subdomain: Specific trait name (e.g., 'Extraversion')
            prediction_target: High level domain (e.g., 'personality', 'emotion')
        """
        # 1. Prepare DataLoaders
        train_ds = SupervisedDatasets(X_train, y_train)
        train_loader = DataLoader(
            train_ds,
            batch_size=self.dl_train_cfg['batch_size'],
            num_workers=self.dl_general_cfg['num_workers'],
            shuffle=True
        )

        valid_loader = None
        if X_valid is not None and y_valid is not None:
            valid_ds = SupervisedDatasets(X_valid, y_valid)
            valid_loader = DataLoader(
                valid_ds,
                batch_size=self.dl_test_cfg['batch_size'],
                num_workers=self.dl_general_cfg['num_workers'],
                shuffle=False
            )

        # 2. Initialize Scheduler (needs loader length)
        self._init_scheduler(len(train_loader))

        # 3. Training Loop
        self.min_valid_loss = None
        self.best_epoch = -1

        for epoch in range(self.dl_train_cfg['num_epochs']):
            # --- TRAIN ---
            self.model.train()
            running_loss = 0.0

            # Use tqdm for progress if verbose, else silent
            tbar = tqdm(train_loader, ncols=80, desc=f"Ep {epoch}")

            for i_batch, batch in enumerate(tbar):
                X = batch[0].to(self.device).float()
                y = batch[1].to(self.device).long()

                self.optimizer.zero_grad()

                # Forward pass: handle multi-stream input [B, M, T] -> list of [B, T]
                inputs = [X[:, m, :] for m in range(X.shape[1])]
                pred = self.model(*inputs)

                if y.dim() == 0 or len(y) == 1:
                    # Handle edge case of batch_size=1
                    loss = self.criterion(pred, y.view(-1))
                else:
                    loss = self.criterion(pred, y.squeeze())

                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

                running_loss += loss.item()
                tbar.set_postfix(loss=loss.item())

            # --- VALIDATE ---
            if valid_loader:
                current_val_loss = self._validate(valid_loader, epoch)

                # Checkpoint logic
                if self.min_valid_loss is None or current_val_loss < self.min_valid_loss:
                    self.min_valid_loss = current_val_loss
                    self.best_epoch = epoch
                    print(f"Update best model! Best epoch: {self.best_epoch}")

                # Save checkpoint every epoch (as per original logic)
                self._save_checkpoint(epoch, prediction_target, cv_split_index, affect_subdomain)

        # 4. Save best epoch index for reference
        self._save_best_epoch_info(prediction_target, cv_split_index, affect_subdomain)

        # 5. Load best model weights for immediate prediction use
        self._load_best_weights(prediction_target, cv_split_index, affect_subdomain)

    def _validate(self, loader, epoch):
        self.model.eval()
        valid_loss = []

        with torch.no_grad():
            for batch in loader:
                X = batch[0].to(self.device).float()
                y = batch[1].to(self.device).long()

                inputs = [X[:, m, :] for m in range(X.shape[1])]
                pred = self.model(*inputs)

                if y.dim() == 0 or len(y) == 1:
                    loss = self.criterion(pred, y.view(-1))
                else:
                    loss = self.criterion(pred, y.squeeze())

                valid_loss.append(loss.item())

        return np.mean(valid_loss)

    def predict(self, X_test):
        """
        Standard sklearn-like predict.
        Assumes self.model is already loaded with the best weights from fit().
        """
        # Prepare data
        # We create a dummy Y because SupervisedDatasets expects it
        dummy_y = np.zeros(len(X_test))
        test_ds = SupervisedDatasets(X_test, dummy_y)
        test_loader = DataLoader(
            test_ds,
            batch_size=self.dl_test_cfg['batch_size'],
            num_workers=self.dl_general_cfg['num_workers'],
            shuffle=False
        )

        self.model.eval()
        y_pred_all = []

        with torch.no_grad():
            for batch in tqdm(test_loader, ncols=80, desc="Predicting"):
                X = batch[0].to(self.device).float()

                inputs = [X[:, m, :] for m in range(X.shape[1])]
                pred = self.model(*inputs)

                # Get class index
                y_pred = torch.max(pred.data, 1)[1]
                y_pred_all.append(y_pred)

        # Concatenate and move to CPU/Numpy
        return torch.cat(y_pred_all, dim=0).cpu().numpy()

    # =========================================================================
    #  Internal I/O Helpers
    # =========================================================================

    def _get_save_dir(self, prediction_target):
        return f"{self.dl_general_cfg['pretrained_model_path']}/{self.clf_name}/{prediction_target}/{self.name_ext}/"

    def _save_checkpoint(self, epoch, prediction_target, cv_split_index, affect_subdomain):
        save_dir = self._get_save_dir(prediction_target)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        if affect_subdomain is None:
            fname = f"cv{cv_split_index}_Epoch{epoch}.pth"
        else:
            fname = f"cv{cv_split_index}_{affect_subdomain}_Epoch{epoch}.pth"

        torch.save(self.model.state_dict(), os.path.join(save_dir, fname))

    def _save_best_epoch_info(self, prediction_target, cv_split_index, affect_subdomain):
        best_epoch_dir = self.dl_general_cfg['log_path'] + f'/train/{self.clf_name}/{prediction_target}/{self.name_ext}'
        if not os.path.exists(best_epoch_dir):
            os.makedirs(best_epoch_dir)

        if affect_subdomain is None:
            fname = f"cv{cv_split_index}_best_epoch.npy"
        else:
            fname = f"cv{cv_split_index}_{affect_subdomain}_best_epoch.npy"

        np.save(os.path.join(best_epoch_dir, fname), np.asarray(self.best_epoch))

    def load_best_epoch_from_file(self, prediction_target, cv_split_index, affect_subdomain):
        """
        Retrieves the best epoch index from the saved .npy log file.
        This is essential for 'only_test' mode to know which model checkpoint to load.
        """
        # 1. Check if best_epoch is manually forced in config
        if self.dl_test_cfg['best_epoch'] is not None:
            self.best_epoch = self.dl_test_cfg['best_epoch']
            print(f"Using manually forced best epoch: {self.best_epoch}")
            return

        # 2. Construct path to the .npy file (Same logic as _save_best_epoch_info)
        best_epoch_dir = self.dl_general_cfg['log_path'] + f'/train/{self.clf_name}/{prediction_target}/{self.name_ext}'

        if affect_subdomain is None:
            fname = f"cv{cv_split_index}_best_epoch.npy"
        else:
            fname = f"cv{cv_split_index}_{affect_subdomain}_best_epoch.npy"

        path = os.path.join(best_epoch_dir, fname)

        # 3. Load the file
        if os.path.exists(path):
            # .item() is used because np.save wraps the integer in a 0-d array
            self.best_epoch = np.load(path, allow_pickle=True).item()
            print(f"Loaded best epoch index {self.best_epoch} from {path}")
        else:
            raise FileNotFoundError(f"Best epoch file not found at {path}")

    def _load_best_weights(self, prediction_target, cv_split_index, affect_subdomain):
        """
        Loads the actual model weights (.pth) using the self.best_epoch index.
        """
        if self.best_epoch is None:
            raise ValueError("Error: self.best_epoch is None. Cannot load weights.")

        # 1. Construct path to the .pth file (Same logic as _save_checkpoint)
        save_dir = self._get_save_dir(prediction_target)

        if affect_subdomain is None:
            fname = f"cv{cv_split_index}_Epoch{self.best_epoch}.pth"
        else:
            fname = f"cv{cv_split_index}_{affect_subdomain}_Epoch{self.best_epoch}.pth"

        path = os.path.join(save_dir, fname)

        # 2. Load the weights
        if os.path.exists(path):
            # map_location ensures it loads even if you move from GPU to CPU
            state_dict = torch.load(path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state_dict)
            print(f"Model weights loaded successfully from {path}")
        else:
            print(f"Warning: Model weight file not found at {path}")