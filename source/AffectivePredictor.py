import cupy as cp
import numpy as np
import torch
import torch.nn as nn
import xgboost as xgb
import os

from tqdm import tqdm
from torch.utils.data import DataLoader

from sklearn.model_selection import LeaveOneGroupOut, LeaveOneOut, GroupKFold
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, balanced_accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from dataloader.datasets import SupervisedDatasets
from models.nets import SL_model_flex
from models.DCNN import SimpleMultiStreamCNN
from sp_utils import fisher_idx, calculate_accuracy



import warnings
warnings.filterwarnings(
    "ignore",
    message=(
        r"(A single label was found in 'y_true' and 'y_pred'\."
        r"|y_pred contains classes not in y_true"
        r"|The groups parameter is ignored by LeaveOneOut)"
    ),
    category=UserWarning
)


def _scale_train_test(X, train_idx, test_idx, scaler_name='MinMax', feature_range=(-1, 1)):
    """Fit a Min-Max scaler on X[train_idx] and transform both splits."""
    if scaler_name == 'minmax':
        scaler = MinMaxScaler(feature_range=feature_range)
    elif scaler_name == 'standard':
        scaler = StandardScaler()
    elif scaler_name == 'no_scaling':
        return X[train_idx], X[test_idx]
    else:
        raise ValueError(f"Unknown scaler: {scaler_name}")
    X_train = scaler.fit_transform(X[train_idx])
    X_test  = scaler.transform(X[test_idx])
    return X_train, X_test


def _feature_selection(X_train, X_test, y_train, feature_selection_method, num_features, random_seed):
    if feature_selection_method =='fisher':
        idx_selected = fisher_idx(num_features, X_train, y_train)
        return X_train[:, idx_selected], X_test[:, idx_selected]
    elif feature_selection_method == 'mi':
        sel = SelectKBest(mutual_info_classif, k=num_features).fit(X_train, y_train)
        return sel.transform(X_train), sel.transform(X_test)
    elif feature_selection_method == 'l1':
        l1 = LinearSVC(C=0.01, penalty='l1', dual=False).fit(X_train, y_train)
        idx = np.flatnonzero(l1.coef_.sum(axis=0))
        idx = idx[:num_features] if num_features < len(idx) else idx
        return X_train[:, idx], X_test[:, idx]
    elif feature_selection_method == 'tree':
        et = ExtraTreesClassifier(n_estimators=300, random_state=random_seed).fit(X_train, y_train)
        idx = np.argsort(et.feature_importances_)[::-1][:num_features]
        return X_train[:, idx], X_test[:, idx]
    elif feature_selection_method == 'anova':
        sel = SelectKBest(f_classif, k=min(num_features, X_train.shape[1])).fit(X_train, y_train)
        return sel.transform(X_train), sel.transform(X_test)
    elif feature_selection_method == 'lrc':
        lr = LogisticRegression(C=1, penalty='l2', solver='liblinear', class_weight='balanced',
                                random_state=random_seed)
        lr.fit(X_train, y_train)
        coef_per_feat = np.abs(lr.coef_).max(axis=0)
        idx = np.argsort(coef_per_feat)[::-1][:num_features]  # leng
        return X_train[:, idx], X_test[:, idx]
    elif feature_selection_method == 'pca':                          # projection, not true FS
        svd = TruncatedSVD(n_components=num_features, random_state=random_seed).fit(X_train)
        return svd.transform(X_train), svd.transform(X_test)
    elif feature_selection_method == 'no_feature_selection':
        return X_train, X_test
    else:
        raise ValueError(f"Unknown feature selection method: {feature_selection_method}")


class AffectivePredictor:
    def __init__(self, X, groups, df_all_labels, participants, emotion_domains, personality_domains,
                 feature_selection_method, num_features, clf_name, affect_domains, random_seed, cfg, session_label,
                 dfs_all, chunk_len, feature_modalities, fisherface_labels, use_dl_method, splitter_name, cfg_dl=None):
        self.X = X
        self.groups = groups
        self.df_all_labels = df_all_labels
        self.participants = participants
        self.emotion_domains = emotion_domains
        self.personality_domains = personality_domains
        self.feature_selection_method = feature_selection_method
        self.num_features = num_features
        self.affect_domains = affect_domains
        self.random_seed = random_seed
        self.clf_name = clf_name
        self.cfg = cfg
        self.session_label = session_label
        self.dfs_all = dfs_all
        self.chunk_len = chunk_len
        self.feature_modalities = feature_modalities
        self.fisherface_labels = fisherface_labels
        self.feats_per_mod = 15
        self.use_dl_method = use_dl_method
        self.splitter_name = splitter_name

        if use_dl_method:
            self.cfg_dl = cfg_dl
            self.name_ext = ''
            for feature_modality in self.feature_modalities:
                self.name_ext += f"{feature_modality}_"
            self.name_ext = self.name_ext[:-1]

            if self.cfg_dl['name_extension'] == '':
                self.name_ext += '/noext'
            else:
                self.name_ext += f"/{self.cfg_dl['name_extension']}"


    def _init_classifier(self, n_classes):
        if self.clf_name == 'LogisticRegression':
            self.clf = LogisticRegression(C=1, solver='liblinear', random_state=self.random_seed)
        elif self.clf_name == 'GaussianNB':
            self.clf = GaussianNB()
        elif self.clf_name == 'LinearSVM':
            self.clf = LinearSVC(C=0.25, dual=False, random_state=self.random_seed)
        elif self.clf_name == 'ExtraTrees':
            self.clf = ExtraTreesClassifier(n_estimators=300, max_features='sqrt', random_state=self.random_seed)
        elif self.clf_name == 'HistGB':
            self.clf = HistGradientBoostingClassifier(learning_rate=0.05, max_depth=None, random_state=self.random_seed)
        elif self.clf_name == 'RbfSVM':
            self.clf = SVC(kernel='rbf', probability=True, C=1, gamma='scale', random_state=self.random_seed)
        elif self.clf_name == 'kNN':
            self.clf = KNeighborsClassifier(n_neighbors=7, weights='distance')
        elif self.clf_name == 'RF':
            self.clf = RandomForestClassifier(random_state=self.random_seed)
        elif self.clf_name == 'xGB':
            self.clf = xgb.XGBClassifier(device='cuda', objective="binary:logistic", tree_method="hist",
                                         random_state=self.random_seed)
        elif self.clf_name == 'WER':
            self.clf = SL_model_flex(num_classes=n_classes,
                                num_modalities=self.X.shape[1],
                                tcn_nfilters=self.cfg_dl['tcn_nfilters'],
                                tcn_kernel_size=self.cfg_dl['tcn_kernel_size'],
                                tcn_dropout=self.cfg_dl['tcn_dropout'],
                                trans_d_model=self.cfg_dl['trans_d_model'],
                                trans_n_heads=self.cfg_dl['trans_n_heads'],
                                trans_num_layers=self.cfg_dl['trans_num_layers'],
                                trans_dim_feedforward=self.cfg_dl['trans_dim_feedforward'],
                                shared_embed_dim=self.cfg_dl['shared_embed_dim'],
                                trans_dropout=self.cfg_dl['trans_dropout'],
                                trans_activation=self.cfg_dl['trans_activation'],
                                trans_norm=self.cfg_dl['trans_norm'],
                                trans_freeze=False,
                                sl_embed_dim1=self.cfg_dl['sl_embed_dim1'],
                                sl_activation=self.cfg_dl['sl_activation'],
                                sl_dropout=self.cfg_dl['sl_dropout'])
        elif self.clf_name == 'DCNN':
            self.clf = SimpleMultiStreamCNN(num_modalities=self.X.shape[1],
                                         conv_channels=[64, 128, 256],
                                         kernel_sizes=[5, 5, 5],
                                         pool_sizes=[2, 2, 2],
                                         mlp_hidden=128,
                                         num_classes=n_classes,
                                         dropout=0.5)
        else:
            raise ValueError(f"Unknown classifier: {self.clf_name}")
        if self.clf_name in ['DCNN', 'WER']:
            self.device = torch.device(self.cfg_dl['device'] if torch.cuda.is_available() else 'cpu')
            self.clf.to(self.device)
            if self.cfg_dl['optim'] == 'sgd':
                self.optimizer = torch.optim.SGD(self.clf.parameters(), self.cfg_dl['lr'],
                                            momentum=self.cfg_dl['momentum'],
                                            weight_decay=self.cfg_dl['weight_decay'])
            elif self.cfg_dl['optim'] == 'adam':
                self.optimizer = torch.optim.Adam(self.clf.parameters(), self.cfg_dl['lr'])
            else:
                raise ValueError('Optimizer %s is not supported' % self.cfg_dl['optim'])
            self.criterion = nn.CrossEntropyLoss().to(self.device)
            self.min_valid_loss = None
            self.best_epoch = None
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer, max_lr=self.cfg_dl['lr'], epochs=self.cfg_dl['num_epochs'],
                steps_per_epoch=len(self.data_loader_train))


    def _init_dataloader(self, X_train, y_train, X_valid, y_valid, X_test, y_test):
        train = SupervisedDatasets(X_train, y_train)
        valid = SupervisedDatasets(X_valid, y_valid)
        test = SupervisedDatasets(X_test, y_test)

        self.data_loader_train = DataLoader(train, batch_size=self.cfg_dl['batch_size'], num_workers=16, shuffle=True)
        self.data_loader_valid = DataLoader(valid, batch_size=self.cfg_dl['batch_size'], num_workers=16)
        self.data_loader_test = DataLoader(test, batch_size=self.cfg_dl['batch_size'], num_workers=16)


    def _get_train_valid_splits(self, train_idx, y_full):
        train_groups = np.unique(self.groups[train_idx])
        rng = np.random.RandomState(0)
        val_groups = rng.choice(train_groups, size=int(0.1*len(train_groups)), replace=False)

        grp_at_train = self.groups[train_idx]
        is_val = np.isin(grp_at_train, val_groups)
        is_train = ~is_val

        inner_train_idx = train_idx[is_train]
        val_idx = train_idx[is_val]

        X_train, y_train = self.X[inner_train_idx], y_full[inner_train_idx]
        X_valid, y_valid = self.X[val_idx], y_full[val_idx]

        return X_train, y_train, X_valid, y_valid\


    def _save_model(self, epoch, affect_name, cv_split_index, affect_subdomain):
        # Row feature modalities behind each other
        save_dir = f"{self.cfg_dl['pretrained_model_path']}/{self.clf_name}/{affect_name}/{self.name_ext}/"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if affect_subdomain is None:
            torch.save(self.clf.state_dict(), save_dir + f'/cv{cv_split_index}_Epoch{str(epoch)}.pth')
        else:
            torch.save(self.clf.state_dict(), save_dir + f'/cv{cv_split_index}_{affect_subdomain}_Epoch{str(epoch)}.pth')


    def train_SL(self, epoch, affect_name, cv_split_index, affect_subdomain=None):
        running_loss = 0.0
        train_loss = []

        self.clf.train()
        tbar = tqdm(self.data_loader_train, ncols=80)
        for i_batch, batch in enumerate(tbar):
            X = batch[0].to(self.device).float()  # (B, M, T)
            y = batch[1].to(self.device).long()  # (B,)

            self.optimizer.zero_grad()

            # forward; works for any 1 ≤ M ≤ 9
            inputs = [X[:, m, :] for m in range(X.shape[1])]
            pred = self.clf(*inputs)

            if np.size(np.array(y.cpu())) == 1:
                loss = self.criterion(pred, y)
            else:
                loss = self.criterion(pred, y.squeeze())

            if i_batch % 10 == 0 and i_batch > 0:  # print every 50 mini-batches
                print(f'[{epoch}, {i_batch + 1:5d}] loss: {running_loss / 10:.3f}')
                running_loss = 0.0
            train_loss.append(loss.item())

            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            running_loss += loss.item()
            tbar.set_postfix(loss=loss.item())

        self._save_model(epoch, affect_name, cv_split_index, affect_subdomain)


    def valid_SL(self, cv_split_index, affect_subdomain=None):
        valid_loss = []
        y_true_all = []
        y_pred_all = []
        self.clf.eval()
        with torch.no_grad():
            tbar = tqdm(self.data_loader_valid, ncols=80)
            for i_batch, batch in enumerate(tbar):
                X = batch[0].to(self.device).float()  # (B, M, T)
                y = batch[1].to(self.device).long()  # (B,)

                # forward; works for any 1 ≤ M ≤ 9
                inputs = [X[:, m, :] for m in range(X.shape[1])]
                pred = self.clf(*inputs)
                y_pred = torch.max(pred.data, 1)[1]

                if np.size(np.array(y.cpu())) == 1:
                    loss = self.criterion(pred, y)
                else:
                    loss = self.criterion(pred, y.squeeze())
                valid_loss.append(loss.item())
                y_pred_all.append(y_pred)
                y_true_all.append(y)
                tbar.set_postfix(loss=loss.item())

        y_pred_all = torch.cat(y_pred_all, dim=0).cpu().numpy()
        y_true_all = torch.cat(y_true_all, dim=0).cpu().numpy()

        if affect_subdomain is None:
            print(f'Validation loss for CV index {cv_split_index}: {np.mean(valid_loss):.3f}')
            print(f'Validation accuracy for CV index {cv_split_index}: {calculate_accuracy(y_pred_all, y_true_all):.3f}\n')
        else:
            print(f'Validation loss for CV index {cv_split_index} for subdomain '
                  f'{affect_subdomain}: {np.mean(valid_loss):.3f}')
            print(f'Validation accuracy for CV index {cv_split_index} for subdomain {affect_subdomain}: '
                  f'{calculate_accuracy(y_pred_all, y_true_all):.3f}\n')
        return np.mean(valid_loss)


    def test_SL(self, affect_name, cv_split_index, affect_subdomain=None):
        print('Using best epoch: ', self.best_epoch)
        if affect_subdomain is None:
            model_path = (f"{self.cfg_dl['pretrained_model_path']}/{self.clf_name}/{affect_name}/{self.name_ext}/"
                          f"cv{cv_split_index}_Epoch{self.best_epoch}.pth")
        else:
            model_path = (f"{self.cfg_dl['pretrained_model_path']}/{self.clf_name}/{affect_name}/{self.name_ext}/"
                          f"cv{cv_split_index}_{affect_subdomain}_Epoch{self.best_epoch}.pth")
        self.clf.load_state_dict(torch.load(model_path, weights_only=True))

        y_pred_all = []
        y_true_all = []
        self.clf.eval()
        with torch.no_grad():
            tbar = tqdm(self.data_loader_test, ncols=80)
            for i_batch, batch in enumerate(tbar):
                X = batch[0].to(self.device).float()  # (B, M, T)
                y = batch[1].to(self.device).long()  # (B,)

                # forward; works for any 1 ≤ M ≤ 9
                inputs = [X[:, m, :] for m in range(X.shape[1])]
                pred = self.clf(*inputs)
                y_pred = torch.max(pred.data, 1)[1]

                y_pred_all.append(y_pred)
                y_true_all.append(y)

        y_pred_all = torch.cat(y_pred_all, dim=0).cpu().numpy()
        y_true_all = torch.cat(y_true_all, dim=0).cpu().numpy()

        if affect_subdomain is None:
            print(f'Test accuracy for CV index {cv_split_index}: {calculate_accuracy(y_pred_all, y_true_all):.3f}\n')
        else:
            print(f'Test accuracy for CV index {cv_split_index}, subdomain {affect_subdomain}: '
                  f'{calculate_accuracy(y_pred_all, y_true_all):.3f}\n')
        return y_pred_all


    def train_test_dl(self, affect_name, cv_split_index, train_idx, y_full, X_test, y_test, n_classes,
                      affect_subdomain=None):
        X_train, y_train, X_valid, y_valid = self._get_train_valid_splits(train_idx, y_full)
        self._init_dataloader(X_train, y_train, X_valid, y_valid, X_test, y_test)
        self._init_classifier(n_classes)
        if self.cfg_dl['mode'] == 'train_and_test':
            for epoch in range(self.cfg_dl['num_epochs']):
                print(f"Epoch {epoch}/{self.cfg_dl['num_epochs']}")
                self.train_SL(epoch, affect_name, cv_split_index, affect_subdomain=affect_subdomain)
                valid_loss = self.valid_SL(cv_split_index, affect_subdomain=affect_subdomain)
                if self.min_valid_loss is None:
                    self.min_valid_loss = valid_loss
                    self.best_epoch = epoch
                    print("Update best model! Best epoch: {}".format(self.best_epoch))
                elif valid_loss < self.min_valid_loss:
                    self.min_valid_loss = valid_loss
                    self.best_epoch = epoch
                    print("Update best model! Best epoch: {}".format(self.best_epoch))

            # Save file with best epoch
            best_epoch_dir = self.cfg_dl['log_path'] + f'/train/{self.clf_name}/{affect_name}/{self.name_ext}'
            if not os.path.exists(best_epoch_dir):
                os.makedirs(best_epoch_dir)
            if affect_subdomain is None:
                best_epochs_path = best_epoch_dir + f'/cv{cv_split_index}_best_epoch.npy'
            else:
                best_epochs_path = best_epoch_dir + f'/cv{cv_split_index}_{affect_subdomain}_best_epoch.npy'
            np.save(best_epochs_path, np.asarray(self.best_epoch))

            y_pred = self.test_SL(affect_name, cv_split_index, affect_subdomain=affect_subdomain)
        elif self.cfg_dl['mode'] == 'only_test':
            if self.cfg_dl['best_epoch'] is not None:
                self.best_epoch = self.cfg_dl['best_epoch']
            else:
                best_epoch_dir = self.cfg_dl['log_path'] + f'/train/{self.clf_name}/{affect_name}/{self.name_ext}'
                if affect_subdomain is None:
                    best_epochs_path = best_epoch_dir + f'/cv{cv_split_index}_best_epoch.npy'
                else:
                    best_epochs_path = best_epoch_dir + f'/cv{cv_split_index}_{affect_subdomain}_best_epoch.npy'
                self.best_epoch = np.load(best_epochs_path, allow_pickle=True).item()
            y_pred = self.test_SL(affect_name, cv_split_index, affect_subdomain=affect_subdomain)
        else:
            raise ValueError(f"Unknown mode: {self.cfg_dl['mode']}")

        return y_pred


    # ------------------------------------------------------------------
    def predict_affect(self):
        """
        Binary affect prediction with LOSO‑CV.
        Prints F1, BAC and random baselines; returns a dict of macro scores.
        """
        # Define cross‑validation splitter
        if self.splitter_name == 'LOGO':
            splitter = LeaveOneGroupOut()
        elif self.splitter_name == 'GKF':
            splitter = GroupKFold(n_splits=5)
        else:
            raise ValueError(f"Unknown splitter: {self.splitter_name}")

        results_f1, results_bac = {}, {}
        rand_f1, rand_bac = {}, {}
        n_classes = 2

        for affect in self.affect_domains:
            # ---------- build labels ---------------------------------------
            y_cont = self.df_all_labels[affect].values.astype(float)
            thr = 2 if affect in ('Liking', 'Familiarity') else np.median(y_cont)
            y = (y_cont >= thr).astype(int)

            # random baselines for this affect
            p_pos = y.mean()
            rand_f1[affect] = p_pos  # see derivation
            rand_bac[affect] = 0.5  # BAC_rand always 0.5

            f1_folds, bac_folds = [], []
            tested_participants = []
            cv_split_index = 0

            # ---------- LOSO loop -----------------------------------------
            for train_idx, test_idx in splitter.split(self.X, y, self.groups):
                X_train, X_test = self.X[train_idx], self.X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                tested_participants.append(np.unique(self.groups[test_idx])[0])

                # Fisherface modality selection (unchanged)
                if 'fisherfaces' in self.feature_modalities:
                    cols_keep = [i for i in range(self.feats_per_mod * len(self.fisherface_labels),
                                                  self.X.shape[1])]
                    if affect not in self.fisherface_labels:
                        raise RuntimeError(f"{affect} not in fisherface labels")
                    aff_idx = np.where(np.asarray(self.fisherface_labels) == affect)[0][0]
                    fisher_cols = list(range(aff_idx * self.feats_per_mod,
                                             (aff_idx + 1) * self.feats_per_mod))
                    cols_keep = fisher_cols + cols_keep
                    X_train, X_test = X_train[:, cols_keep], X_test[:, cols_keep]

                # Feature selection
                X_train, X_test = _feature_selection(X_train, X_test, y_train, self.feature_selection_method,
                                                     self.num_features, self.random_seed)

                # skip degenerate training folds
                if len(np.unique(y_train)) < n_classes:
                    y_pred = np.full_like(y_test, fill_value=y_train[0])
                else:
                    if self.clf_name == "xGB":
                        X_train, X_test = cp.asarray(X_train), cp.asarray(X_test)
                        y_train = cp.asarray(y_train)

                    # Predict
                    if self.use_dl_method:
                        y_pred = self.train_test_dl('affect', cv_split_index, train_idx, y, X_test, y_test, n_classes,
                                                    affect_subdomain=affect)
                    else:
                        self._init_classifier(n_classes)
                        self.clf.fit(X_train, y_train)
                        y_pred = self.clf.predict(X_test)
                    # self._init_classifier(n_classes)
                    # self.clf.fit(Xtr, ytr)
                    # y_pred = self.clf.predict(Xte)

                cv_split_index += 1
                f1_folds.append(f1_score(y_test, y_pred, zero_division='warn'))
                bac_folds.append(balanced_accuracy_score(y_test, y_pred))

            results_f1[affect] = np.mean(f1_folds)
            results_bac[affect] = np.mean(bac_folds)
            # results_f1[affect] = np.std(f1_folds)  # if you want to report std instead of mean
            # results_bac[affect] = np.std(bac_folds)

        # ---------- macro scores (model & random) -------------------------
        macro_f1_model = np.mean(list(results_f1.values()))
        macro_bac_model = np.mean(list(results_bac.values()))
        macro_f1_rand = np.mean(list(rand_f1.values()))
        macro_bac_rand = np.mean(list(rand_bac.values()))  # always 0.5

        # ---------- print --------------------------------------------------
        print("\n=== Affect prediction (LOSO) ===")
        print("Affect          F1    (rand)    BAC   (rand)")
        print("------------------------------------------------")
        for aff in self.affect_domains:
            print(f"{aff:15s} {results_f1[aff]:.3f} ({rand_f1[aff]:.3f})   "
                  f"{results_bac[aff]:.3f} ({rand_bac[aff]:.3f})")

        print("------------------------------------------------")
        print(f"Macro‑F1  : {macro_f1_model :.3f}   (rand {macro_f1_rand :.3f})")
        print(f"Macro‑BAC : {macro_bac_model:.3f}   (rand {macro_bac_rand:.3f})")

        # Save results to .txt file
        if self.use_dl_method:
            save_log_dir = f"{self.cfg_dl['log_path']}/test/affect/{self.name_ext}"
            if not os.path.exists(save_log_dir):
                os.makedirs(save_log_dir)
            with open(f"{save_log_dir}/results.txt", 'w') as f:
                f.write("=== Affect prediction (LOSO) ===\n")
                f.write("Affect          F1    (rand)    BAC   (rand)\n")
                f.write("------------------------------------------------\n")
                for aff in self.affect_domains:
                    f.write(f"{aff:15s} {results_f1[aff]:.3f} ({rand_f1[aff]:.3f})   "
                            f"{results_bac[aff]:.3f} ({rand_bac[aff]:.3f})\n")
                f.write("------------------------------------------------\n")
                f.write(f"Macro‑F1  : {macro_f1_model :.3f}   (rand {macro_f1_rand :.3f})\n")
                f.write(f"Macro‑BAC : {macro_bac_model:.3f}   (rand {macro_bac_rand:.3f})\n")
                f.write(f"Participants: {', '.join(map(str, self.participants))}\n")

        return macro_f1_model, [v for k, v in results_f1.items()], [v for k, v in rand_f1.items()]


    # Hard 1‑of‑9 prediction– each clip is labelled with *argmax* emotion.
    # Predict dominant emotion for each task
    def predict_emotion(self, return_macro=True):
        # 1) Get labels
        y_list = []
        df_labels = self.df_all_labels[["participant"] + self.emotion_domains]
        for p in self.participants:
            y_list.append(df_labels[df_labels["participant"] == p].iloc[:, 1:].values.argmax(axis=1))
        y_full = np.concatenate(y_list)
        n_classes = 9

        # Random baselines
        p_vec = np.bincount(y_full, minlength=len(self.emotion_domains)) / y_full.size
        rand_per_class_f1 = p_vec.astype(float)  # F1rand_k = p_k
        rand_per_class_bac = p_vec.astype(float)  # recall = p_k
        rand_macro_f1 = rand_per_class_f1.mean()
        rand_macro_bac = rand_per_class_bac.mean()

        # Define cross‑validation splitter
        if self.splitter_name == 'LOGO':
            splitter = LeaveOneGroupOut()
        elif self.splitter_name == 'GKF':
            splitter = GroupKFold(n_splits=5)
        else:
            raise ValueError(f"Unknown splitter: {self.splitter_name}")

        fold_scores = []  # macro‑F1 or accuracy
        y_true_all = []  # for global per‑class F1 at the end
        y_pred_all = []
        tested_participants = []
        per_class_f1_folds = []
        cv_split_index = 0

        for train_idx, test_idx in splitter.split(self.X, y_full, self.groups):
            print(f"CV split {cv_split_index}...")
            tested_participants.append(np.unique(self.groups[test_idx])[0])
            X_train, X_test = self.X[train_idx], self.X[test_idx]
            if 'fisherfaces' in self.feature_modalities:
                cols_keep = [i for i in range(self.feats_per_mod * len(self.fisherface_labels), self.X.shape[1])]
                if 'strongest_emotion' not in self.fisherface_labels:
                    raise RuntimeError(f"Warning: strongest_emotion not in fisherface labels")
                affect_index = np.where(np.asarray(self.fisherface_labels) == 'strongest_emotion')[0][0]
                fisher_cols_keep = [i for i in
                                    range(affect_index * self.feats_per_mod, (affect_index + 1) * self.feats_per_mod)]
                cols_keep = fisher_cols_keep + cols_keep
                X_train, X_test = X_train[:, cols_keep], X_test[:, cols_keep]
            y_train, y_test = y_full[train_idx], y_full[test_idx]

            if len(np.unique(y_train)) != n_classes: continue

            # Feature selection
            X_train, X_test = _feature_selection(X_train, X_test, y_train, self.feature_selection_method,
                                                 self.num_features, self.random_seed)

            # Load on GPU if xgboost
            if self.clf_name == 'xGB':
                X_train, X_test = cp.asarray(X_train), cp.asarray(X_test)
                y_train = cp.asarray(y_train)

            # Predict
            if self.use_dl_method:
                y_pred = self.train_test_dl('emotion_single_label', cv_split_index, train_idx, y_full, X_test, y_test,
                                            n_classes)
            else:
                self._init_classifier(n_classes)
                self.clf.fit(X_train, y_train)
                y_pred = self.clf.predict(X_test)

            y_pred_all.append(y_pred)
            y_true_all.append(y_test)
            cv_split_index += 1

            if return_macro:
                fold_scores.append(f1_score(y_full[test_idx], y_pred, average="macro"))
            else:
                fold_scores.append((y_pred == y_full[test_idx]).mean())

            f1_fold = f1_score(y_test, y_pred, labels=list(range(n_classes)), average=None, zero_division=0)
            per_class_f1_folds.append(f1_fold)

        # Per‑class F1 (computed once on concatenated TRUE/PRED)
        y_true = np.concatenate(y_true_all)
        y_pred = np.concatenate(y_pred_all)
        n_cls = len(self.emotion_domains)
        per_class_f1 = f1_score(y_true, y_pred, labels = list(range(n_cls)), average = None, zero_division = 0)
        present_mask = np.array([(y_true == k).any() or (y_pred == k).any() for k in range(n_cls)])
        per_class_f1[~present_mask] = np.nan  # gap labels → NaN

        # Print participants with the lowest f1 scores
        k = min(3, len(fold_scores))
        worst_idx = np.argsort(fold_scores)[:k]
        for i_part in worst_idx:
            print(f"Worst participant for single emotion {tested_participants[i_part]}: {np.round(fold_scores[i_part], 2)}")
        print('')

        # 4) Per‑class BAC
        per_class_bac = []
        for k in range(len(self.emotion_domains)):
            mask = y_true == k
            recall_k = (y_pred[mask] == k).mean() if mask.any() else 0.0
            per_class_bac.append(recall_k)
        per_class_bac = np.array(per_class_bac)

        per_class_f1_folds = np.vstack(per_class_f1_folds)  # (n_folds, n_cls)
        per_class_f1_std = np.nanstd(per_class_f1_folds, axis=0, ddof=1)  # <-- this is the std you want

        # 5) Print
        print("\n=== Emotion single-label===")
        for k, emo in enumerate(self.emotion_domains):
            print(f"{emo:12s}: "
                  f"F1 = {per_class_f1[k]:.3f} (rand {rand_per_class_f1[k]:.3f})   "
                  f"F1 STD = {per_class_f1_std[k]:.3f}   "
                  f"BAC = {per_class_bac[k]:.3f} (rand {rand_per_class_bac[k]:.3f})")

        macro_f1 = float(np.nanmean(per_class_f1))
        macro_f1_std = float(np.nanstd(fold_scores, ddof=1)) if len(fold_scores) > 1 else float('nan')
        macro_bac = float(np.nanmean(per_class_bac))
        print(f"\n→ Global macro-F1  : {macro_f1 :.3f}   (rand {rand_macro_f1 :.3f})")
        print(f"→ Global macro-F1 STD : {macro_f1_std :.3f}")
        print(f"→ Global macro-BAC : {macro_bac:.3f}   (rand {rand_macro_bac:.3f})")

        return macro_f1, per_class_f1, rand_per_class_f1
        # return macro_f1, per_class_f1_std, rand_per_class_f1

    def predict_personality(self, df_personality_means, df_personality_T, features_all, use_T=False,
                            do_single_prediction=False):
        results_f1, results_bac = {}, {}
        rand_f1, rand_bac = {}, {}

        if do_single_prediction:
            X = np.vstack([self.X[self.groups == p].mean(axis=0) for p in self.participants])
            splitter = LeaveOneOut()
            X_groups = np.arange(len(self.participants))  # dummy
        else:
            X = self.X
            if self.splitter_name == 'LOGO':
                splitter = LeaveOneGroupOut()
            elif self.splitter_name == 'GKF':
                splitter = GroupKFold(n_splits=5)
            else:
                raise ValueError(f"Unknown splitter: {self.splitter_name}")
            X_groups = self.groups

        if use_T:
            df_labels = df_personality_T
        else:
            df_labels = df_personality_means

        thresholds = {d: df_labels[d].median() for d in self.personality_domains}

        # =============================================================== #
        for dom in self.personality_domains:
            # -------- build binary labels ------------------------------ #
            if do_single_prediction:
                y_cont = df_labels.loc[self.participants, dom].values
            else:
                part_scores = df_labels[dom]
                y_cont = np.concatenate([np.full(features_all[p].shape[0], part_scores.loc[p])
                                         for p in self.participants])

            y = (y_cont >= thresholds[dom]).astype(int)
            p_pos = y.mean()  # class prior
            rand_f1[dom] = p_pos  # E[F1] random
            rand_bac[dom] = 0.5  # E[BAC] random

            # -------- CV ------------------------------------------------ #
            f1s, bacs = [], []
            cv_split_index = 0
            n_classes = 2
            for train_idx, test_idx in splitter.split(X, y, X_groups):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                # ---- modality selection --------------------------------
                if 'fisherfaces' in self.feature_modalities:
                    cols_keep = [i for i in range(self.feats_per_mod * len(self.fisherface_labels), self.X.shape[1])]
                    if dom not in self.fisherface_labels:
                        raise RuntimeError(f"{dom} not in fisherface labels")
                    aff_idx = np.where(np.asarray(self.fisherface_labels) == dom)[0][0]
                    fisher_cols = list(range(aff_idx * self.feats_per_mod,
                                             (aff_idx + 1) * self.feats_per_mod))
                    cols_keep = fisher_cols + cols_keep
                    X_train, X_test = X_train[:, cols_keep], X_test[:, cols_keep]

                # Feature selection
                X_train, X_test = _feature_selection(X_train, X_test, y_train, self.feature_selection_method,
                                                     self.num_features, self.random_seed)

                if len(np.unique(y_train)) != n_classes: continue

                # ---- GPU transfer (xgboost) -----------------------------
                if self.clf_name == "xGB":
                    X_train, X_test = cp.asarray(X_train), cp.asarray(X_test)
                    y_train = cp.asarray(y_train)

                # Predict
                if self.use_dl_method:
                    y_pred = self.train_test_dl('personality', cv_split_index, train_idx, y, X_test,
                                                y_test, n_classes, affect_subdomain=dom)
                else:
                    self._init_classifier(n_classes)
                    self.clf.fit(X_train, y_train)
                    y_pred = self.clf.predict(X_test)

                cv_split_index += 1

                # ---- metrics -------------------------------------------
                if y_test.sum() == 0 and y_pred.sum() == 0:
                    f1s.append(1.0)
                else:
                    f1s.append(f1_score(y_test, y_pred, labels=[0, 1]))
                bacs.append(balanced_accuracy_score(y_test, y_pred))

            results_f1[dom] = np.mean(f1s)
            results_bac[dom] = np.mean(bacs)
            # results_f1[dom] = np.std(f1s)  # if you want to report std instead of mean
            # results_bac[dom] = np.std(bacs)

        mean_f1_overall = float(np.mean(list(results_f1.values())))
        mean_bac_overall = float(np.mean(list(results_bac.values())))
        mean_rand_f1_overall = float(np.mean(list(rand_f1.values())))
        mean_rand_bac_overall = float(np.mean(list(rand_bac.values())))

        # =========================  PRINT  ============================== #
        print("\n=== Personality prediction ===")
        print(f"Trait                    F1   (rand)    BAC  (rand)")
        print("------------------------------------------------------")
        for dom in self.personality_domains:
            print(f"{dom:23s}  {results_f1[dom]:.3f} ({rand_f1[dom]:.3f})   "
                  f"{results_bac[dom]:.3f} ({rand_bac[dom]:.3f})")

        print("------------------------------------------------------")
        print(f"Mean F1  over 5 traits : {mean_f1_overall :.3f} (rand {mean_rand_f1_overall :.3f})")
        print(f"Mean BAC over 5 traits : {mean_bac_overall:.3f} (rand {mean_rand_bac_overall:.3f})")

        return mean_f1_overall, [v for k, v in results_f1.items()], [v for k, v in rand_f1.items()]