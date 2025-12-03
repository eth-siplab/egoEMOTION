# libs/models/factory.py
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost as xgb
from .wrapper import DeepLearningClassifier

def get_classifier(name,
                   feature_modalities=None,
                   dl_general_cfg=None,
                   dl_model_cfg=None,
                   dl_train_cfg=None,
                   dl_test_cfg=None,
                   dl_opt_cfg=None,
                   input_shape=None,
                   n_classes=2,
                   random_seed=0
                   ):
    if name == 'RF':
        return RandomForestClassifier(random_state=random_seed)
    elif name == 'RbfSVM':
        return SVC(kernel='rbf', probability=True, C=1, gamma='scale', random_state=random_seed)
    elif name == 'xGB':
        return xgb.XGBClassifier(device='cuda:0', objective="binary:logistic", tree_method="hist",
                                 random_state=random_seed)
    elif name in ['WER', 'DCNN']:
        return DeepLearningClassifier(name, feature_modalities, dl_general_cfg, dl_model_cfg, dl_train_cfg, dl_test_cfg,
                                      dl_opt_cfg, input_shape, n_classes)
    else:
        raise ValueError(f"Unknown model: {name}")