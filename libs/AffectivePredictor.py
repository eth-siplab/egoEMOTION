import numpy as np
import cupy as cp

from sklearn.model_selection import LeaveOneGroupOut, LeaveOneOut, GroupKFold
from sklearn.metrics import f1_score, balanced_accuracy_score
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.decomposition import TruncatedSVD

from .models.factory import get_classifier
from .utils.signal import fisher_idx


# personality prediction warnings suppression
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


class AffectivePredictor:
    def __init__(self,X, cfg, groups, df_all_labels, fisherface_labels):
        self.X = X
        self.groups = groups
        self.df_all_labels = df_all_labels
        self.random_seed = cfg['random_seed']
        self.session = cfg['session']
        self.prediction_target = cfg['prediction_target']

        # Dataset info
        self.participants = cfg['dataset']['participants']
        self.affect_domains = cfg['dataset']['affect_domains']
        self.emotion_domains = cfg['dataset']['emotion_domains']
        self.personality_domains = cfg['dataset']['personality_domains']

        # Model config
        self.feature_selection_method = cfg['model']['feature_selection_method']
        self.num_features = cfg['model'].get('num_features', None)
        self.clf_name = cfg['model']['clf_name']
        self.is_dl_method = cfg['model']['is_dl_method']

        # DL-specific config
        self.dl_general_cfg = cfg.get('dl_general_cfg', None)
        self.dl_model_cfg = cfg.get('dl_model_cfg', None)
        self.dl_train_cfg = cfg.get('dl_train_cfg', None)
        self.dl_test_cfg = cfg.get('dl_test_cfg', None)
        self.dl_opt_cfg = cfg.get('dl_opt_cfg', None)

        # Metadata / Extra info needed for logging
        self.feature_modalities = cfg['features']['feature_modalities']
        self.splitter_name = cfg['splitter_name']
        self.fisherface_labels = fisherface_labels
        self.feats_per_mod = 15  # hard-coded constant depending on get_stats function in extraction.py

        # Construct name extension for logging
        self.name_ext = self._build_name_extension()

    def _build_name_extension(self):
        if not self.is_dl_method:
            return ""
        ext = "_".join(self.feature_modalities)
        if self.dl_general_cfg.get('name_extension', '') == '':
            ext += '/noext'
        else:
            ext += f"/{self.dl_general_cfg['name_extension']}"
        return ext

    # =========================================================================
    #  Internal Helper: Create Validation Split
    # =========================================================================
    def _get_train_valid_splits(self, X_full, y_full, groups_full, train_idx):
        """
        Splits the training set further into Train and Validation (10% of groups).
        Reproduces logic from original _get_train_valid_splits.
        """
        # Get the groups corresponding to the training indices
        grp_at_train = groups_full[train_idx]
        train_groups_unique = np.unique(grp_at_train)

        rng = np.random.RandomState(0)  # Fixed seed as in original

        # Select 10% of groups for validation
        n_val = max(1, int(0.1 * len(train_groups_unique)))
        val_groups = rng.choice(train_groups_unique, size=n_val, replace=False)

        # Create boolean masks
        is_val = np.isin(grp_at_train, val_groups)
        is_train = ~is_val

        # Get the actual indices relative to the FULL dataset
        inner_train_idx = train_idx[is_train]
        val_idx = train_idx[is_val]

        # Slice data
        X_train = X_full[inner_train_idx]
        y_train = y_full[inner_train_idx]

        X_valid = X_full[val_idx]
        y_valid = y_full[val_idx]

        return X_train, y_train, X_valid, y_valid

    # =========================================================================
    #  Shared Cross-Validation Logic
    # =========================================================================
    def _run_cv(self, X, y, groups, n_classes, domain_name, splitter=None, n_groups=None):
        """
        Runs the Cross-Validation loop.
        Returns a list of tuples: (y_true_fold, y_pred_fold, participant_id)
        """
        if splitter is None:
            if self.splitter_name == 'LOGO':
                splitter = LeaveOneGroupOut()
                n_splits = len(np.unique(groups))
            elif self.splitter_name == 'GKF':
                n_splits = 5
                splitter = GroupKFold(n_splits=n_splits)
            else:
                raise ValueError(f"Unknown splitter: {self.splitter_name}")
        else:
            n_splits = n_groups

        results = []

        for cv_split_index, (train_idx, test_idx) in enumerate(splitter.split(X, y, groups)):
            X_train_full = X[train_idx]  # This might contain validation data if DL
            y_train_full = y[train_idx]
            X_test = X[test_idx]
            y_test = y[test_idx]

            # Identify test participant(s) (for logging/analysis)
            unique_test_groups = np.unique(groups[test_idx])

            if self.splitter_name == 'LOGO':
                curr_participant = unique_test_groups[0]
            else:
                curr_participant = f"Fold_{cv_split_index}"

            # -----------------------------------------------------------------
            # 1. Validation Split (Deep Learning Only)
            # -----------------------------------------------------------------
            X_valid = None
            y_valid = None

            if self.is_dl_method:
                X_train, y_train, X_valid, y_valid = self._get_train_valid_splits(
                    X, y, groups, train_idx
                )
            else:
                # For classical ML, use all training data for training
                X_train = X_train_full
                y_train = y_train_full

            # -----------------------------------------------------------------
            # 2. Fisherface Modality Filtering
            # -----------------------------------------------------------------
            if 'fisherfaces' in self.feature_modalities:
                X_train, X_test = self._apply_fisherface_filter(X_train, X_test, domain_name)
                if X_valid is not None:
                    X_valid, _ = self._apply_fisherface_filter(X_valid, X_valid, domain_name)

            # -----------------------------------------------------------------
            # 3. General Feature Selection
            # -----------------------------------------------------------------
            # Note: Fit on Train, Transform Train/Valid/Test
            X_train, X_test, selector = self._feature_selection(X_train, X_test, y_train, return_selector=True)
            if X_valid is not None and selector is not None:
                # Transform validation set using the selector fitted on training set
                if self.feature_selection_method in ['mi', 'pca', 'anova']:
                    X_valid = selector.transform(X_valid)
                elif self.feature_selection_method in ['fisher', 'l1', 'tree', 'lrc']:
                    X_valid = X_valid[:, selector]

            # -----------------------------------------------------------------
            # 4. Handle Degenerate Folds
            # -----------------------------------------------------------------
            if len(np.unique(y_train)) < n_classes:
                y_pred = np.full_like(y_test, fill_value=y_train[0])
            else:
                # GPU Transfer for XGBoost
                if self.clf_name == "xGB":
                    X_train, X_test = cp.asarray(X_train), cp.asarray(X_test)
                    y_train = cp.asarray(y_train)

                # 5. Initialize Model
                model = get_classifier(
                    self.clf_name,
                    self.feature_modalities,
                    dl_general_cfg=self.dl_general_cfg,
                    dl_model_cfg=self.dl_model_cfg,
                    dl_train_cfg=self.dl_train_cfg,
                    dl_test_cfg=self.dl_test_cfg,
                    dl_opt_cfg=self.dl_opt_cfg,
                    input_shape=X_train.shape[1],
                    n_classes=n_classes
                )

                # 6. Train / Test Logic
                if self.is_dl_method:
                    mode = self.dl_general_cfg['mode']

                    if mode == 'train_and_test':
                        model.fit(
                            X_train, y_train,
                            X_valid=X_valid, y_valid=y_valid,
                            cv_split_index=cv_split_index,
                            affect_subdomain=domain_name,
                            prediction_target=self.prediction_target
                        )
                    elif mode == 'only_test':
                        model.load_best_epoch_from_file(self.prediction_target, cv_split_index, domain_name)
                        model._load_best_weights(self.prediction_target, cv_split_index, domain_name)
                else:
                    model.fit(X_train, y_train)

                y_pred = model.predict(X_test)

            results.append((y_test, y_pred, curr_participant))
            print(f'Finished cross-validation split {cv_split_index + 1}/{n_splits}')

        return results

    # =========================================================================
    #  Helper Methods
    # =========================================================================
    def _apply_fisherface_filter(self, X_train, X_test, domain_name):
        """Logic to only select the fisherface features relevant to the current affect domain."""
        # Calculate start index of non-fisher features
        fisher_offset = self.feats_per_mod * len(self.fisherface_labels)
        cols_keep = list(range(fisher_offset, self.X.shape[1]))  # all columns without fisherfaces columns

        # Find which fisher block belongs to this domain
        if domain_name not in self.fisherface_labels:
            # Fallback for strongest_emotion in predict_emotion
            if domain_name == 'emotion' and 'strongest_emotion' in self.fisherface_labels:
                target_label = 'strongest_emotion'
            else:
                raise RuntimeError(f"{domain_name} not in fisherface labels: {self.fisherface_labels}")
        else:
            target_label = domain_name

        aff_idx = np.where(np.asarray(self.fisherface_labels) == target_label)[0][0]
        fisher_cols = list(range(aff_idx * self.feats_per_mod, (aff_idx + 1) * self.feats_per_mod))

        final_cols = fisher_cols + cols_keep
        return X_train[:, final_cols], X_test[:, final_cols]

    def _feature_selection(self, X_train, X_test, y_train, return_selector=False):
        """Modified to return the selector so we can transform X_valid too."""
        return _feature_selection_routine(
            X_train, X_test, y_train,
            self.feature_selection_method,
            self.num_features,
            self.random_seed,
            return_selector=return_selector
        )

    # =========================================================================
    #  Public API 1: Affect Prediction
    # =========================================================================
    def predict_affect(self):
        print("\n=== Affect prediction (LOSO) ===")
        results_f1, results_bac = {}, {}
        rand_f1, rand_bac = {}, {}

        for affect in self.affect_domains:
            # 1. Label Generation
            y_cont = self.df_all_labels[affect].values.astype(float)
            thr = 2 if affect in ('Liking', 'Familiarity') else np.median(y_cont)
            y = (y_cont >= thr).astype(int)

            # 2. Random Baselines
            rand_f1[affect] = y.mean()
            rand_bac[affect] = 0.5

            # 3. Run CV Engine
            fold_results = self._run_cv(self.X, y, self.groups, n_classes=2, domain_name=affect)

            # 4. Calculate Metrics (Mean over folds)
            f1_folds = [f1_score(yt, yp, zero_division=0) for yt, yp, _ in fold_results]
            bac_folds = [balanced_accuracy_score(yt, yp) for yt, yp, _ in fold_results]

            results_f1[affect] = np.mean(f1_folds)
            results_bac[affect] = np.mean(bac_folds)

        # Print & Save
        # self._print_and_save_affect_results(results_f1, results_bac, rand_f1, rand_bac)
        macro_f1 = np.mean(list(results_f1.values()))
        macro_bac = np.mean(list(results_bac.values()))
        macro_f1_rand = np.mean(list(rand_f1.values()))
        macro_bac_rand = np.mean(list(rand_bac.values()))
        print("Affect          F1    (rand)    BAC   (rand)")
        print("-" * 48)
        for aff in self.affect_domains:
            print(f"{aff:15s} {results_f1[aff]:.3f} ({rand_f1[aff]:.3f})   {results_bac[aff]:.3f} ({rand_bac[aff]:.3f})")
        print("-" * 48)
        print(f"Macro‑F1  : {macro_f1 :.3f}   (rand {macro_f1_rand :.3f})")
        print(f"Macro‑BAC : {macro_bac:.3f}   (rand {macro_bac_rand:.3f})\n")

    # =========================================================================
    #  Public API 2: Emotion Prediction (Multiclass)
    # =========================================================================
    def predict_emotion(self):
        print("\n=== Emotion single-label ===")

        # 1. Label Generation (One-hot argmax)
        y_list = []
        df_labels = self.df_all_labels[["participant"] + self.emotion_domains]
        for p in self.participants:
            y_list.append(df_labels[df_labels["participant"] == p].iloc[:, 1:].values.argmax(axis=1))
        y_full = np.concatenate(y_list)
        n_classes = 9

        # 2. Run CV Engine (Pass 'emotion' as domain name)
        fold_results = self._run_cv(self.X, y_full, self.groups, n_classes=n_classes, domain_name='emotion')

        # 3. Process Results (Specific to Emotion: Worst Participant & Per-Class Stats)
        y_true_all = np.concatenate([res[0] for res in fold_results])
        y_pred_all = np.concatenate([res[1] for res in fold_results])

        # Calculate fold-wise macro F1 to find worst participants
        fold_scores = [f1_score(yt, yp, average="macro") for yt, yp, _ in fold_results]
        tested_parts = [res[2] for res in fold_results]

        # Print worst
        worst_idx = np.argsort(fold_scores)[:3]
        for i in worst_idx:
            print(f"Worst participant {tested_parts[i]}: {fold_scores[i]:.2f}")

        # Global Per-Class F1
        per_class_f1 = f1_score(y_true_all, y_pred_all, labels=list(range(n_classes)), average=None, zero_division=0)

        # Print Table
        for k, emo in enumerate(self.emotion_domains):
            print(f"{emo:12s}: F1 = {per_class_f1[k]:.3f}")

        macro_f1 = np.mean(per_class_f1)
        print(f"\n→ Global macro-F1  : {macro_f1:.3f}")

    # =========================================================================
    #  Public API 3: Personality Prediction
    # =========================================================================
    def predict_personality(self, df_personality_means, df_personality_T, features_all, use_T=False):
        print("\n=== Personality prediction ===")
        results_f1, results_bac = {}, {}
        rand_f1, rand_bac = {}, {}

        # Single prediction uses one sample per participant (mean features)
        do_single_prediction = False if self.is_dl_method else True  # DL methods use per-sample prediction
        if do_single_prediction:
            X_curr = np.vstack([self.X[self.groups == p].mean(axis=0) for p in self.participants])
            groups_curr = np.arange(len(self.participants))
            splitter = LeaveOneOut()
            n_groups = len(self.participants)
        # Otherwise, predict per sample
        else:
            X_curr = self.X
            groups_curr = self.groups
            splitter = None  # Use default
            n_groups = None

        df_labels = df_personality_T if use_T else df_personality_means
        thresholds = {d: df_labels[d].median() for d in self.personality_domains}

        for dom in self.personality_domains:
            # 2. Label Generation
            if do_single_prediction:
                y_cont = df_labels.loc[self.participants, dom].values
            else:
                part_scores = df_labels[dom]
                y_cont = np.concatenate(
                    [np.full(features_all[p].shape[0], part_scores.loc[p]) for p in self.participants])

            y = (y_cont >= thresholds[dom]).astype(int)
            rand_f1[dom] = y.mean()  # E[F1] random
            rand_bac[dom] = 0.5  # E[BAC] random

            # 3. Run CV Engine
            fold_results = self._run_cv(X_curr, y, groups_curr, n_classes=2, domain_name=dom, splitter=splitter,
                                        n_groups=n_groups)

            # 4. Metrics
            f1s = []
            bacs = []
            for yt, yp, _ in fold_results:
                if yt.sum() == 0 and yp.sum() == 0:
                    f1s.append(1.0)
                else:
                    f1s.append(f1_score(yt, yp, labels=[0, 1]))
                    bacs.append(balanced_accuracy_score(yt, yp))

            results_f1[dom] = np.mean(f1s)
            results_bac[dom] = np.mean(bacs)
            print(f"{dom:23s}  F1: {results_f1[dom]:.3f}")

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


# =============================================================================
# Static Helper for Feature Selection
# =============================================================================
def _feature_selection_routine(X_train, X_test, y_train, method, num_features, seed, return_selector=False):
    selector = None
    X_train_out, X_test_out = X_train, X_test

    if method == 'fisher':
        idx = fisher_idx(num_features, X_train, y_train)
        X_train_out = X_train[:, idx]
        X_test_out = X_test[:, idx]
        selector = idx
    elif method == 'mi':
        sel = SelectKBest(mutual_info_classif, k=num_features).fit(X_train, y_train)
        X_train_out = sel.transform(X_train)
        X_test_out = sel.transform(X_test)
        selector = sel
    elif method == 'pca':
        svd = TruncatedSVD(n_components=num_features, random_state=seed).fit(X_train)
        X_train_out = svd.transform(X_train)
        X_test_out = svd.transform(X_test)
        selector = svd
    elif method is None:
        pass
    else:
        # Fallback for methods not strictly implemented in this snippet
        raise ValueError(f"Unknown selection method: {method}")

    if return_selector:
        return X_train_out, X_test_out, selector
    else:
        return X_train_out, X_test_out