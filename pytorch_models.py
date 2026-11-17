import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import time
from pyclassifiers import ResNetClassifier
from sklearn import metrics
from sklearn.metrics import average_precision_score
import scipy
# Simple classifier for Graident Shaple
class SimplePytorchModel:
    def __init__(self, model, num_classes,epochs=1, training_batch_size=32,
                 optimizer='sgd', loss_fn=None, device=None,lr=0.01):
        self.num_classes = num_classes
        self.epochs = epochs
        self.training_batch_size = training_batch_size
        self.lr = lr

        # self.device = torch.device('cuda' if torch.cuda.is_available() and device == 'cuda' else 'cpu')
        self.device = 'cpu'
        self.model = model.to(self.device)
        self.model_state_dict = self.model.state_dict()
        self.loss_fn = loss_fn

        # Set the optimizer with Adam as default
        if optimizer == 'sgd':
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr)
        elif optimizer == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    def reinitialise_model(self):
        for m in self.model.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
            elif isinstance(m, torch.nn.BatchNorm1d):
                torch.nn.init.ones_(m.weight)
                torch.nn.init.zeros_(m.bias)

        self.model_state_dict = self.model.state_dict()

    def numpy_to_tensors(self, X, Y=None):
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        Y_tensor = None
        if Y is not None:
            if self.num_classes <= 2:
                Y_tensor = torch.tensor(Y, dtype=torch.float32).view(-1, 1).to(self.device)
            else:
                Y_tensor = torch.tensor(Y, dtype=torch.long).to(self.device)
        return X_tensor, Y_tensor

    def numpy_to_dataloader(self, X, Y):
        X_tensor, y_tensor = self.numpy_to_tensors(X, Y)
        dataset = TensorDataset(X_tensor, y_tensor)
        return DataLoader(dataset=dataset, batch_size=self.training_batch_size, shuffle=True)

    def fit(self, X_train, y_train):

        self.model.load_state_dict(self.model_state_dict)

        train_loader = self.numpy_to_dataloader(X_train, y_train)

        for epoch in range(self.epochs):
            self.model.train()
            train_loss = 0.0
            for batch in train_loader:
                X_batch, y_batch = batch
                self.optimizer.zero_grad()
                y_pred = self.model(X_batch)
                loss = self.loss_fn(y_pred, y_batch)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()

            # # Print epoch results
            # print(f"Epoch {epoch + 1}/{self.epochs}, "
            #       f"Train Loss: {train_loss/len(train_loader):.4f}")

    def predict(self, X):
        X_test_tensor, _ = self.numpy_to_tensors(X)
        self.model.eval()
        with torch.no_grad():
            y_pred = self.model(X_test_tensor)
            if self.num_classes > 2:
                predictions = torch.argmax(y_pred, dim=1).cpu().numpy()
            else:
                predictions = (torch.sigmoid(y_pred) >= 0.5).cpu().numpy().squeeze().astype(int)
        return predictions

    def predict_proba(self, X):
        X_test_tensor, _ = self.numpy_to_tensors(X)
        self.model.eval()
        with torch.no_grad():
            y_pred = self.model(X_test_tensor)
            if self.num_classes > 2:
                return torch.softmax(y_pred, dim=1).cpu().numpy()
            else:
                proba = torch.sigmoid(y_pred).cpu().numpy().squeeze()
                return np.column_stack((1 - proba, proba))


    def performance_score(self, X, y):
        X_tensor, y_tensor = self.numpy_to_tensors(X, y)
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)
            if self.num_classes > 2:
                y_pred = torch.argmax(outputs, dim=1) 
            else:
                y_pred = (torch.sigmoid(outputs) >= 0.5).float()
            return (y_pred == y_tensor).float().mean().item()  # Accuracy
        
class PytorchModel:
    def __init__(self, model, num_classes, epochs=100, training_batch_size=32,
                 validation_batch_size=32, loss_fn=None, device=None, patience=10, hyperparam_search=False,lr=1e-5,set_scheduler=False,warm_up=False):
        self.num_classes = num_classes

        self.epochs = epochs
        self.training_batch_size = training_batch_size
        self.validation_batch_size = validation_batch_size
        self.patience = patience
        self.loss_fn = loss_fn
        self.hyperparam_search = hyperparam_search
        self.device = 'cpu'
        self.model = model.to(self.device)
        self.warm_up = warm_up
        self.warmup_epochs = 5  # Number of warm-up epochs
        self.warmup_factor = 10  # How much smaller to start the LR
        self.lr = lr
        self.training_time = 0.0
        self.set_scheduler = set_scheduler
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        # Add tracking lists for losses
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.best_epoch = 0


        # Define hyperparameters dictionary
        self.hyperparams = {
            'lr': 0.00001,
            'weight_decay': 0.01,
            'scheduler_factor': 0.5,
            'scheduler_patience': 10,
            'grad_clip': 1.0
        }

        self.input_dim = model.input_dim

        if self.set_scheduler:
            if epochs <= 15:
                tmp_patience = 3  # Remove the comma
            else:
                tmp_patience = 10

            self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=tmp_patience)
        else:
            self.scheduler = None


    def _adjust_learning_rate(self, epoch):
        """Adjust learning rate during warm-up phase"""
        if self.warm_up and epoch < self.warmup_epochs:
            # Linear warm-up
            target_lr = self.hyperparams['lr']
            current_lr = target_lr / self.warmup_factor + \
                        (target_lr - target_lr / self.warmup_factor) * \
                        (epoch / self.warmup_epochs)

            for param_group in self.optimizer.param_groups:
                param_group['lr'] = current_lr

    def numpy_to_tensors(self, X, Y=None):
       X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
       Y_tensor = None
       if Y is not None:
           if self.num_classes <= 2:
               Y_tensor = torch.tensor(Y, dtype=torch.float32).view(-1, 1).to(self.device)
           else:
               Y_tensor = torch.tensor(Y, dtype=torch.long).to(self.device)
       return X_tensor, Y_tensor

    def numpy_to_dataloader(self, X, Y, batch_size, shuffle=True):
        X_tensor, y_tensor = self.numpy_to_tensors(X, Y)
        dataset = TensorDataset(X_tensor, y_tensor)
        return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)

    def _evaluate(self, X_val, y_val):
        """Evaluate model on validation set"""
        self.model.eval()
        val_loader = self.numpy_to_dataloader(X_val, y_val, self.validation_batch_size, shuffle=False)
        val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                X_batch, y_batch = batch
                y_pred = self.model(X_batch)
                loss = self.loss_fn(y_pred, y_batch)
                val_loss += loss.item()

        return val_loss / len(val_loader)

    def fit(self, X_train, y_train, X_val, y_val, verbose=True):

        train_loader = self.numpy_to_dataloader(X_train, y_train, self.training_batch_size)
        val_loader = self.numpy_to_dataloader(X_val, y_val, self.validation_batch_size, shuffle=False)

        best_val_loss = float('inf')
        epochs_without_improvement = 0
        best_model_state = None

        start_train_time = time.time()
        for epoch in range(self.epochs):
            if self.warm_up:
                self._adjust_learning_rate(epoch)
            # Training phase
            self.model.train()
            train_loss = 0.0
            for batch in train_loader:
                X_batch, y_batch = batch
                self.optimizer.zero_grad()
                y_pred = self.model(X_batch)
                loss = self.loss_fn(y_pred, y_batch)

                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()

            # Calculate average training loss
            train_loss /= len(train_loader)
            self.train_losses.append(train_loss)

            # Validation phase
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    X_batch, y_batch = batch
                    y_pred = self.model(X_batch)
                    loss = self.loss_fn(y_pred, y_batch)
                    val_loss += loss.item()

            val_loss /= len(val_loader)
            self.val_losses.append(val_loss)

            # Learning rate scheduling
            if self.scheduler is not None:
                self.scheduler.step(val_loss)
            min_delta = 1e-4  # You can adjust this value

            if val_loss < (best_val_loss - min_delta):  # Only count significant improvements
                best_val_loss = val_loss
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                epochs_without_improvement = 0
                # Create a deep copy of model state
                best_model_state = {
                    key: val.cpu().detach().clone()
                    for key, val in self.model.state_dict().items()
                }
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= self.patience:
                if verbose:
                    print(f"Early stopping triggered at epoch {epoch + 1}")
                    break

        # Load the best model state
        if best_model_state is not None:  # Changed from 'if best_model_state:'
            self.model.load_state_dict(best_model_state)
            if verbose:
                print(f"Best model found at epoch {self.best_epoch + 1} "
                      f"with validation loss: {self.best_val_loss:.4f}")

        self.training_time = time.time() - start_train_time

    def predict_with_proba(self, X):
        X_test_tensor, _ = self.numpy_to_tensors(X)
        self.model.eval()
        with torch.no_grad():
            y_pred = self.model(X_test_tensor)

            if self.num_classes > 2:
                probabilities = torch.softmax(y_pred, dim=1).cpu().numpy()
                predictions = torch.argmax(y_pred, dim=1).cpu().numpy()
            else:
                probabilities = torch.sigmoid(y_pred).cpu().numpy().squeeze()
                predictions = (probabilities >= 0.5).astype(int)
                probabilities = np.column_stack((1 - probabilities, probabilities))

        return predictions, probabilities

    def predict(self, X):
        predictions, _ = self.predict_with_proba(X)
        return predictions

    def predict_proba(self, X):
        _, probabilities = self.predict_with_proba(X)
        return probabilities

class ModelEvaluator:
    def __init__(self, config):
        self.config = config
        self.supported_models = ['IMSFNN','RESNET']
        self.best_hyperparams = None
        self.best_training_time = None

    @staticmethod
    def create_model(model_name, input_dim, output_dim, hidden_layer_list):
        if model_name == 'RESNET':
            return ResNetClassifier(input_dim=input_dim, hidden1_dim=hidden_layer_list[0], hidden2_dim=hidden_layer_list[1], hidden3_dim=hidden_layer_list[2], output_dim=output_dim)
        else:
            raise ValueError(f"Unsupported model name: {model_name}. Choose from ['IMSFNN','RESNET'].")

    @staticmethod
    def generate_model_lossfn_numclass(model_name, trnX, trnY, hidden_layer_list):
        unique_classes = np.unique(trnY)
        num_classes = len(unique_classes)

        if num_classes > 2:
            print("Setting up multi-class classification with CrossEntropyLoss")
            loss_function = nn.CrossEntropyLoss()
            output_dim = num_classes
        else:
            num_positives = np.sum(trnY)
            num_negatives = len(trnY) - num_positives
            if num_positives == 0:
                pos_weight = torch.tensor([1.0], device='cpu')  # default to balanced weight
            else:
                pos_weight = torch.tensor([num_negatives / num_positives], device='cpu')

            loss_function = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            output_dim = 1
        input_dim = trnX.shape[1]
        print(f"Input dimension: {trnX.shape}")

        model = ModelEvaluator.create_model(model_name, input_dim, output_dim, hidden_layer_list)

        return model, loss_function, num_classes

    def evaluate_model(self):
        processed_trnX = self.config['trnX']
        processed_trnY = self.config['trnY']
        processed_valX = self.config['valX']
        processed_valY = self.config['valY']
        processed_tstX = self.config['tstX']
        processed_tstY = self.config['tstY']

        model, loss_function, num_classes = self.generate_model_lossfn_numclass(
            model_name =self.config['model_name'],
            trnX = processed_trnX,
            trnY = processed_trnY,
            hidden_layer_list=self.config['hidden_layer_list'],
        )
        
        clf = PytorchModel(
            model,
            num_classes=num_classes,
            epochs=self.config['epochs'],
            device=self.config.get('device', 'cpu'),
            loss_fn=loss_function,
            hyperparam_search=self.config.get('hyperparam_search', False),
            set_scheduler= self.config.get('set_scheduler',False),
            warm_up= self.config.get('warm_up',False),
            lr = self.config.get('lr',1e-5),
            patience=self.config.get('patience', 10),
        )

        # Apply best hyperparameters if they exist
        # if self.best_hyperparams:
        #     clf.hyperparams.update(self.best_hyperparams)
        #     clf._init_optimizer_scheduler()
        #     clf.hyperparam_search = False  # Disable search since we're using best params

        # Only start timing after hyperparameter search

        clf.fit(processed_trnX, processed_trnY,
                processed_valX, processed_valY)


        predictions, probabilities = clf.predict_with_proba(processed_tstX)
        micro_f1 = metrics.f1_score(processed_tstY, predictions, average='micro') # Micro for imbalance class
        macro_f1 = metrics.f1_score(processed_tstY, predictions, average='macro')
        roc_auc = metrics.roc_auc_score(processed_tstY, predictions)
        pr_auc = average_precision_score(processed_tstY, predictions)
        precision = metrics.precision_score(processed_tstY, predictions, average='macro')
        recall = metrics.recall_score(processed_tstY, predictions, average='macro')

        per_class_f1 = metrics.f1_score(processed_tstY, predictions, average=None)
        per_class_precision = metrics.precision_score(processed_tstY, predictions, average=None)
        per_class_recall = metrics.recall_score(processed_tstY, predictions, average=None)

        unique_classes = np.unique(processed_tstY)

        # Create detailed metrics dictionary
        score_dict = {
            'roc_auc': roc_auc,
            'micro_f1': micro_f1,
            'macro_f1': macro_f1,
            'pr_auc': pr_auc,
            'precision': precision,
            'recall': recall
        }

        # Add per-class metrics
        for class_label, f1, prec, rec in zip(unique_classes, per_class_f1, per_class_precision, per_class_recall):
            class_prefix = f'class_{int(class_label)}_'
            score_dict.update({
                f'{class_prefix}f1': f1,
                f'{class_prefix}precision': prec,
                f'{class_prefix}recall': rec
            })

        if self.config['eval_metrics'] == 'f1':
            score = micro_f1
        elif self.config['eval_metrics'] == 'pr_auc':
            score = pr_auc
        elif self.config['eval_metrics'] == 'roc_auc':
            score = roc_auc
        elif self.config['eval_metrics'] == 'precision':
            score = precision
        elif self.config['eval_metrics'] == 'recall':
            score = recall
        elif self.config['eval_metrics'] == 'both': # all
            score = score_dict

        return clf, score, probabilities

    def evaluate_avg_metrics(self, total_count):
        # Initialize storage for metrics
        total_score_dict = {}
        total_proba_list = []
        total_training_time = []

        # Perform hyperparameter search only on first iteration if enabled
        hyperparam_search = self.config.get('hyperparam_search',False)
        if hyperparam_search==True:
            print("\nPerforming initial hyperparameter search...")
            # Create first model and get best hyperparameters
            clf, score, proba = self.evaluate_model()
            self.best_hyperparams = clf.hyperparams
            total_training_time.append(clf.training_time)
            print(f"Best hyperparameters found: {self.best_hyperparams}")


            # Store first iteration results
            if isinstance(score, dict):
                total_score_dict = {key: value for key, value in score.items()}
            else:
                total_score_dict = {'score': score}
            total_proba_list.append(proba)
            remaining_iterations = total_count - 1
            print(f"\nTrain iteration 1/{total_count}, Score: {score}")
        else:
            remaining_iterations = total_count

        # Run remaining iterations with best hyperparameters

        for i in range(remaining_iterations):
            print(f"\nTraining iteration {i+1}/{remaining_iterations}")
            clf, score, proba = self.evaluate_model()
            total_proba_list.append(proba)
            total_training_time.append(clf.training_time)
            print(f"Score: {score}")

            if isinstance(score, dict):
                if not total_score_dict:
                    total_score_dict = {key: 0 for key in score.keys()}
                for key, value in score.items():
                    total_score_dict[key] += value
            else:
                if not total_score_dict:
                    total_score_dict = {'score': 0}
                total_score_dict['score'] += score

        # Calculate averages
        avg_score_dict = {key: value / total_count for key, value in total_score_dict.items()}
        avg_proba_list = np.mean(total_proba_list, axis=0)
        avg_training_time = np.mean(total_training_time,axis=0)

        # Return appropriate format based on original score type
        if self.config['eval_metrics'] in ['f1', 'roc_auc']:
            return avg_score_dict['score'], avg_proba_list
        else:
            return avg_score_dict, avg_proba_list, self.best_hyperparams, avg_training_time
    

from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
import pandas as pd
import numpy as np

def extract_raw_feature(encoded_name):
    """
    Extract raw feature name from encoded feature name.
    Examples:
      - 'cat__workclass_Private' -> 'workclass' 
      - 'num__age' -> 'age'
    """
    if '__' in encoded_name:
        prefix, rest = encoded_name.split('__', 1)
        if prefix == 'cat':
            return rest.split('_', 1)[0]
        else:
            return rest
    return encoded_name

def aggregate_feature_importance_mean(encoded_importance_df):
    """
    Aggregate encoded feature importance back to raw feature level by taking mean.
    This removes cardinality bias.
    """
    # Add raw feature column
    encoded_importance_df = encoded_importance_df.copy()
    encoded_importance_df['raw_feature'] = encoded_importance_df['encode_feature_name'].apply(extract_raw_feature)
    
    # Group by raw feature and take mean
    aggregated = (
        encoded_importance_df
        .groupby('raw_feature', as_index=False)
        .agg({
            'encode_importance': 'mean'
        })
        .sort_values('encode_importance', ascending=False)
        .reset_index(drop=True)
    )

    aggregated = aggregated.rename(columns={'encode_importance': 'raw_importance'})
    return aggregated

def calculate_feature_importance(complete_data, method='random_forest', random_state=42):
    """
    Calculate feature importance using the complete_samples variant (complete rows only).
    
    Args:
        complete_data (dict): Dictionary containing complete_samples variant data with 'trnX', 'trnY', 'feature_names'
        method (str): Method to calculate feature importance ('random_forest' or 'permutation')
        random_state (int): Random seed for reproducibility
    
    Returns:
        tuple: (results_dict, encode_normalise_importance_df, raw_normalise_importance_df)
    """
    
    # Extract training data
    X_train = complete_data['trnX']
    y_train = complete_data['trnY']
    feature_names = complete_data['feature_names']
    
    print(f"Calculating feature importance from {len(X_train)} complete rows")
    print(f"Number of encoded features: {len(feature_names)}")
    
    # Train Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=random_state)
    rf.fit(X_train, y_train)
    
    # Calculate feature importance
    if method == 'random_forest':
        encoded_importance = rf.feature_importances_
    elif method == 'permutation':
        perm_importance = permutation_importance(rf, X_train, y_train, n_repeats=10, random_state=random_state)
        encoded_importance = perm_importance.importances_mean
    else:
        raise ValueError("Method must be 'random_forest' or 'permutation'")
    
    # Normalize encoded importance
    encoded_importance_normalized = encoded_importance / encoded_importance.sum()
    
    # Create encoded-level DataFrame
    encode_normalise_importance_df = pd.DataFrame({
        'encode_feature_name': feature_names,
        'encode_importance': encoded_importance_normalized
    }).sort_values('encode_importance', ascending=False).reset_index(drop=True)
    
    # Aggregate to raw feature level (mean aggregation)
    raw_normalise_importance_df = aggregate_feature_importance_mean(encode_normalise_importance_df)
    
    print(f"Number of raw features: {len(raw_normalise_importance_df)}")
    
    # Create results dict
    results_dict = {
        'encode_normalise_importance_dict': encode_normalise_importance_df.to_dict('records'),
        'raw_normalise_importance_dict': raw_normalise_importance_df.to_dict('records'),
        'method': method,
        'n_complete_rows': len(X_train)
    }
    
    return results_dict, encode_normalise_importance_df, raw_normalise_importance_df

def calculate_missing_FI_scores(incomplete_trnX_missing_indicators, 
                                                        raw_feature_names, 
                                                        raw_feature_importance_dict):
    """
    Calculate score for each data point based on missing important raw features.
    Handles order mismatch between raw_feature_names and feature_importance_dict.
    
    Args:
        incomplete_trnX_missing_indicators: numpy array from incomplete_with_indicators['trnX_missing_indicators']
        raw_feature_names: List of original feature names in the order they appear in missing indicators
        raw_feature_importance_dict: Dict with raw feature importance scores

    
    Returns:
        array: Missing feature importance score for each data point (higher = worse)
    """
    
    # Extract the missing indicators (last len(raw_feature_names) columns)
    n_original_features = len(raw_feature_names)
    
    print(f"Missing indicators shape: {incomplete_trnX_missing_indicators.shape}")
    print(f"Original feature names: {raw_feature_names}")
    
    # Get raw feature importance mapping from raw_feature_importance_dict
    calulated_raw_importance_dict = {item['raw_feature']: item['raw_importance'] 
                     for item in raw_feature_importance_dict}
    
    print(f"Available importance features: {list(calulated_raw_importance_dict.keys())}")
    
    # Create mapping from raw_feature_names to importance scores
    match_feature_importance_ordered = []
    for raw_feature in raw_feature_names:
        if raw_feature in calulated_raw_importance_dict:
            match_feature_importance_ordered.append(calulated_raw_importance_dict[raw_feature])
        else:
            print(f"Warning: Feature '{raw_feature}' not found in importance dict, using 0.0")
            match_feature_importance_ordered.append(0.0)
    
    match_feature_importance_ordered = np.array(match_feature_importance_ordered)
    
    print(f"Feature importance ordered: {match_feature_importance_ordered}")
    
    # Calculate penalties for each data point
    # For each row, multiply missing indicators by feature importance and sum
    raw_missing_FI = np.dot(incomplete_trnX_missing_indicators, match_feature_importance_ordered) 

    # Normalise missing FI to [0,1] range by the sum of all feature importance scores.
    # This is to answer what fraction of total FI is missing for each data point.
    sum_of_FI = np.sum(match_feature_importance_ordered)
    if sum_of_FI > 0:
        normalised_missing_FI_scores = raw_missing_FI / sum_of_FI
    else:
        normalised_missing_FI_scores = raw_missing_FI
    
    print(f"Missing FI score range: [{normalised_missing_FI_scores.min():.4f}, {normalised_missing_FI_scores.max():.4f}]")
    return normalised_missing_FI_scores


