import numpy as np
import random
import time
import torch
import torch.nn as nn
from pytorch_models import ModelEvaluator, SimplePytorchModel, PytorchModel
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.ensemble import RandomForestRegressor
from pytorch_models import ModelEvaluator, SimplePytorchModel, PytorchModel
from sklearn.metrics import pairwise_distances

class SVCalculator:
    def __init__(self, ml_task_config):
        # Default configuration for the predictor
        self.sv_alg_config = {
            # SV algorithm
            'sv_algorithm': 'gradient_sv',
            'max_iterations': 20,
            'convergence_rate': 1e-2,
            'lr': 0.01,
            'optimiser':'sgd',

            # Prediction configuration
            'init_coreset_ratio': 1e-5,
            'total_coreset_ratio': 0.005,
            'init_coreset_select_method': 'kmeans',
        }

        # ml_task_config: SV is evaluated on the same task as the original model
        self.model_name = ml_task_config['model_name']
        self.complete_trnX = ml_task_config['complete_trnX']
        self.complete_trnX_idx = ml_task_config['complete_trnX_idx']
        self.complete_trnY = ml_task_config['complete_trnY']
        self.complete_devX = ml_task_config['complete_devX']
        self.complete_devY = ml_task_config['complete_devY']
        self.epochs = ml_task_config.get('epochs', 100)
        self.device = ml_task_config.get('device', 'cpu')
        self.metrics = ml_task_config.get('metrics')
        self.hidden_layer_list = ml_task_config.get('hidden_layer_list', [256, 128, 64])
        
        self.sv_init_coreset = None
        self.sv_init_exec_time = None
        self.sv_init_iterations = None

    # update the predictor configuration with new values
    def update_sv_alg_config(self, new_config):
        self.sv_alg_config.update(new_config)
        self.sv_algorithm = self.sv_alg_config['sv_algorithm']
        self.max_iterations = self.sv_alg_config['max_iterations']
        self.lr = self.sv_alg_config['lr']
        self.convergence_rate = self.sv_alg_config['convergence_rate']
        self.optimiser = self.sv_alg_config['optimiser']

        self.init_coreset_ratio = self.sv_alg_config['init_coreset_ratio']
        self.total_coreset_ratio = self.sv_alg_config['total_coreset_ratio']
        self.init_coreset_select_method = self.sv_alg_config['init_coreset_select_method']

    def gradient_shapley(self, trnX, trnY, devX, devY):
        N = trnX.shape[0]
        Idx = list(range(N))
        val = np.zeros(N)
        convergence_rate = self.convergence_rate
        prev_val = np.zeros(N)
        max_iterations = self.max_iterations
        model, loss_function, num_classes = ModelEvaluator.generate_model_lossfn_numclass(
            self.model_name, trnX, trnY, self.hidden_layer_list)
        new_clf = SimplePytorchModel(model=model, num_classes=num_classes,
                                     epochs=1, device=self.device,loss_fn=nn.BCEWithLogitsLoss(), lr=self.lr,optimizer=self.optimiser)
        sv_cal_start_time = time.time()
        t = 1
        while True:
            random.shuffle(Idx)
            val_t = np.zeros(N + 1)

            new_clf.reinitialise_model()
            val_t[N] = new_clf.performance_score(devX, devY)

            for j in range(1, N + 1):
                X_point, y_point = trnX[Idx[j-1]:Idx[j-1]+1], trnY[Idx[j-1]:Idx[j-1]+1]
                new_clf.fit(X_point, y_point)
                val_t[j] = new_clf.performance_score(devX, devY)
            
            val[Idx] = ((t - 1) / t) * val[Idx] + (1 / t) * (val_t[1:] - val_t[:N])

            if t > 1:
                diffs = np.max(val - prev_val)
                if t >= max_iterations:
                    print(f'Maximum iterations reached. Convergence not achieved. Current diffs: {diffs}')
                    break
                if diffs < convergence_rate:
                    print(f'Converged at iteration {t}.')
                    break
            prev_val = val.copy()
            t += 1
        sv_cal_end_time = time.time()
        print(f"Time taken to calculate SV: {sv_cal_end_time - sv_cal_start_time:.2f} seconds.")
        sv_exec_time = sv_cal_end_time - sv_cal_start_time
        return val, sv_exec_time, t

    def sv_calculation(self):
        """
        Main pipeline:
        1) Select coreset from COMPLETE training data
        2) Compute ground truth SV on coreset using gradient_shapley
        
        Returns:
            sv_calculation_result_dict: Info about coreset selection
        """
        # 1) Select coreset from complete data
        coreset_result_dict = select_init_coreset(
            init_ratio=self.init_coreset_ratio,
            total_coreset_ratio=self.total_coreset_ratio,
            method=self.init_coreset_select_method,
            trnX_with_indicators=self.complete_trnX,
            trnX_idx=self.complete_trnX_idx,
            trnY=self.complete_trnY
        )
        coreset_np_selected_pos = coreset_result_dict['np_selected_pos']
        coreset_trnX_idx = coreset_result_dict['coreset_trnX_idx']
        coreset_trnY = self.complete_trnY[coreset_np_selected_pos]
        coreset_trnX = self.complete_trnX[coreset_np_selected_pos]

        # 2) Calculate ground truth SV for coreset
        sv_coreset, sv_exec_time, sv_iterations = self.gradient_shapley(
            trnX=coreset_trnX,
            trnY=coreset_trnY,
            devX=self.complete_devX,
            devY=self.complete_devY
        )

        sv_calculation_result_dict = {
            'np_selected_pos': coreset_np_selected_pos,
            'coreset_trnX_idx': coreset_trnX_idx,
            'coreset_sv': sv_coreset,
            'coreset_sv_exec_time': sv_exec_time,
            'coreset_sv_iterations': sv_iterations,
        }

        return sv_calculation_result_dict
        
"KMeans Coreset Selection"
# # K-means clustering to select centroids as core set
def kmeans_ds(features, ratio):
    num_selected_pts = int(ratio * features.shape[0])

    # Use MiniBatchKMeans instead of KMeans for better memory efficiency
    kmeans = MiniBatchKMeans(n_clusters=num_selected_pts, random_state=42, batch_size=1024)
    kmeans.fit(features)
    centroids = kmeans.cluster_centers_

    # Find the closest data point to each centroid in a memory-efficient way
    selected_idx = []
    for centroid in centroids:
        distances = euclidean_distances(centroid.reshape(1, -1), features)
        closest_point_idx = np.argmin(distances)
        selected_idx.append(closest_point_idx)

    return selected_idx

# Select coreset data points
def select_init_coreset(init_ratio, total_coreset_ratio, method,trnX_with_indicators, trnX_idx, trnY):
    selected_pos = []
    if method == 'random':
        # select random data points as anchor points
        selected_pos = random.sample(range(len(trnX_idx)), int(len(trnX_idx) * total_coreset_ratio))
    if method == 'kmeans':
        selected_pos = kmeans_ds(features=trnX_with_indicators, ratio=total_coreset_ratio)

    init_coreset_idx = [idx for pos, idx in enumerate(trnX_idx) if pos in selected_pos]
    # np_selected_pos need to be sorted, such that it match with train_idx
    result_dict = {'np_selected_pos': sorted(selected_pos), 'coreset_trnX_idx': init_coreset_idx}

    return result_dict

def get_centroids(features, targets):
    num_classes = len(np.unique(targets))
    centroids = np.zeros((num_classes, features.shape[-1]), dtype=features.dtype)

    for i in range(num_classes):
        class_features = features[targets == i]
        if class_features.size > 0:
            centroids[i] = np.mean(class_features, axis=0)
        else:
            print(f"Warning: Class {i} has no samples")
            centroids[i] = np.zeros(features.shape[-1])


    return centroids

def get_distance(features, targets):
    centroids = get_centroids(features, targets)

    num_classes = len(np.unique(targets))
    centroids_for_each_example = np.zeros(shape=(features.shape[0], centroids.shape[-1]))

    for i in range(num_classes):
        centroids_for_each_example[targets == i] = centroids[i]

    distance = np.linalg.norm(features - centroids_for_each_example, axis=1)

    return distance


class SVModel:
    def __init__(self, ml_task_config):
        # Default configuration for the predictor
        self.sv_predictor_config = {
            'complete_sv_pred_regressor': 'RF',
        }

        # ml_task_config: SV is evaluated on the same task as the original model
        self.model_name = ml_task_config['model_name']
        self.complete_trnX = ml_task_config['complete_trnX']
        self.complete_trnX_idx = ml_task_config['complete_trnX_idx']
        self.complete_trnY = ml_task_config['complete_trnY']
        self.complete_devX = ml_task_config['complete_devX']
        self.complete_devY = ml_task_config['complete_devY']
        self.epochs = ml_task_config.get('epochs', 100)
        self.device = ml_task_config.get('device', 'cpu')
        self.metrics = ml_task_config.get('metrics')
        self.hidden_layer_list = ml_task_config.get('hidden_layer_list', [256, 128, 64])
        
        # Surrogate model and scalers (will be set after training)
        self.surrogate_model = None

    def update_sv_predictor_config(self, new_config):
        self.sv_predictor_config.update(new_config)
        self.complete_sv_pred_regressor = self.sv_predictor_config['complete_sv_pred_regressor']
        self.gp_amplify_function = self.sv_predictor_config['gp_amplify_function']


    def _train_surrogate_model(self, coreset_features, coreset_sv):
        """
        Train a surrogate model on (coreset_features, coreset_sv).
        
        Stores the trained model in self.surrogate_model.
        
        Args:
            coreset_features: Feature vectors for coreset (either raw features or extracted features)
            coreset_sv: Ground truth Shapley values for coreset
            feature_selection_threshold: Length scale threshold for feature selection (default: 100.0)
                                        Features with length_scale < threshold are kept
        
        Returns:
            surrogate_model: Trained model
        """
        if self.complete_sv_pred_regressor == 'RF':
            surrogate_model = RandomForestRegressor(n_estimators=100, n_jobs=-1)
            surrogate_model.fit(coreset_features, coreset_sv)
            return surrogate_model
            
        else:
            raise ValueError(f"Unknown complete_sv_pred_regressor: {self.complete_sv_pred_regressor}.")

    def sv_fit(self, sv_calculation_result_dict):
        """
        Train surrogate model on (coreset_features, coreset_SV)
        Return the trained surrogate model and related info
        Returns:
            sv_calculation_result_dict: coreset sv, coreset_trnX_idx, np_selected_pos
            surrogate_result_dict: Contains trained surrogate model, scalers (for GP), 
                                  selected dimensions (for GP), and metadata
            sv_pred_exec_time: Execution time
        """
        # 1) Select coreset from complete data

        coreset_np_selected_pos = sv_calculation_result_dict['np_selected_pos']
        coreset_trnX_idx = sv_calculation_result_dict['coreset_trnX_idx']
        sv_coreset = sv_calculation_result_dict['coreset_sv']
        sv_exec_time = sv_calculation_result_dict['coreset_sv_exec_time']
        coreset_trnY = self.complete_trnY[coreset_np_selected_pos]
        coreset_trnX = self.complete_trnX[coreset_np_selected_pos]
        
        # 3) Prepare features for surrogate model
        sv_reg_start_time = time.time()
        coreset_features_all = coreset_trnX
        
        # 4) Train surrogate model on (coreset_features, coreset_SV)
        # Returns (model, selected_dims) where selected_dims is None for non-GP
        self.surrogate_model = self._train_surrogate_model(
            coreset_features_all, 
            sv_coreset
        )
        
        sv_reg_end_time = time.time()
        sv_pred_exec_time = sv_exec_time + (sv_reg_end_time - sv_reg_start_time)

        # 5) Prepare result dictionary
        surrogate_result_dict = {
            'surrogate_model': self.surrogate_model,
            'surrogate_type': self.complete_sv_pred_regressor,
            'coreset_trnX_idx': coreset_trnX_idx,
            'coreset_sv': sv_coreset,
            'coreset_features_shape': coreset_features_all.shape,
            'model_name': self.model_name,
        }

        surrogate_result_dict['X_train_scaled'] = coreset_features_all
        surrogate_result_dict['y_train_scaled'] = sv_coreset
        
        print(f'Surrogate model training completed. '
              f'Total time: {sv_pred_exec_time:.2f} seconds '
              f'({sv_exec_time:.2f}s SV computation, {(sv_reg_end_time - sv_reg_start_time):.2f}s surrogate training)')
        
        return surrogate_result_dict, sv_pred_exec_time

class SVPredict:
    def __init__(self, surrogate_result_dict, preprocessor, preprocessor_metadata, 
                 marginals, numerical_cols, categorical_cols, device='cpu'):
        """
        SVPredict: Predict Shapley values for incomplete data using stochastic imputation.

        Args:
            surrogate_result_dict: From SVModel.sv_fit(), contains:
            preprocessor: Frozen ColumnTransformer from complete_samples variant
            preprocessor_metadata: Dict with 'categorical_cols', 'numerical_cols', etc.
            marginals: Dict with structure:
                {'num': {col: {'mean': mu, 'std': sd}}, 
                 'cat': {col: {'cats': [...], 'probs': [...]}}}
            numerical_cols: List of numerical column names
            categorical_cols: List of categorical column names
            device: 'cpu' or 'cuda' (optional, will try to get from surrogate_result_dict if not provided)
        """
        # Surrogate model info
        self.surrogate_model = surrogate_result_dict['surrogate_model']
        self.surrogate_type = surrogate_result_dict['surrogate_type']
        self.model_name = surrogate_result_dict.get('model_name')
        
        # Preprocessor artifacts
        self.preprocessor = preprocessor
        self.preprocessor_metadata = preprocessor_metadata
        self.numerical_cols = numerical_cols
        self.categorical_cols = categorical_cols
        
        # Marginals for stochastic imputation
        self.marginals = marginals
        
        # Device
        self.device = surrogate_result_dict.get('device', device)
        
        
        # Extract column info from metadata
        self.categorical_cols = self.preprocessor_metadata['categorical_cols']
        self.numerical_cols = self.preprocessor_metadata['numerical_cols']
        self.cat_indices = self.preprocessor_metadata['cat_indices']
        self.num_indices = self.preprocessor_metadata['num_indices']
        self.all_cols = self.preprocessor_metadata['all_cols']
    
    def _sample_imputation(self, incomplete_row):
        """
        Sample one imputation for a single incomplete row from marginals.
        
        Args:
            incomplete_row: pd.Series with missing values (NaNs)
        Returns:
            imputed_row: pd.Series with imputed values
        """
        
        imputed_row = incomplete_row.copy()
        
        # Impute numerical features
        for col in self.numerical_cols:
            if pd.isna(imputed_row[col]):
                mu = self.marginals['num'][col]['mean']
                sd = self.marginals['num'][col]['std']
                imputed_row[col] = np.random.normal(mu, sd)
        
        # Impute categorical features
        for col in self.categorical_cols:
            if pd.isna(imputed_row[col]):
                cats = self.marginals['cat'][col]['cats']
                probs = self.marginals['cat'][col]['probs']
                imputed_row[col] = np.random.choice(cats, p=probs)

        return imputed_row
    
    def _predict_sv_single_imputation(self, encoded_X):
        """
        Predict SV for a single imputation using surrogate model.
        Args:
            encoded_X: Encoded features (after preprocessor)
        Returns: point prediction
        """

        if self.surrogate_type in ['RF']:
            # These models return point predictions
            pred = self.surrogate_model.predict(encoded_X)
            return pred
        else:
            raise ValueError(f"Unknown surrogate_type: {self.surrogate_type}")

    def predict_sv_incomplete(self, incomplete_X_df, n_imputations=10, random_seed=42):
        """
        Predict SV for incomplete data with multiple stochastic imputations.

        Args:
            incomplete_X_df: DataFrame with incomplete data (can have NaNs)
            n_imputations: Number of imputations to draw per row
            random_seed: Random seed for reproducibility

        Returns:
            result_dict: {
                'mean_sv': np.array of shape (n_samples,),
                'var_sv': np.array of shape (n_samples,),
                'std_sv': np.array of shape (n_samples,)
            }
        """
        n_samples = len(incomplete_X_df)

        all_predictions = []

        print(f"Predicting SV for {n_samples} incomplete rows with {n_imputations} imputations each...")
        print(f"Using surrogate model: {self.surrogate_type}")

        # Draw M imputations and predict for each
        for m in range(n_imputations):
            np.random.seed(random_seed + m)

            # Sample imputation for all rows
            imputed_rows = []
            for idx in incomplete_X_df.index:
                row = incomplete_X_df.loc[idx]
                imputed_row = self._sample_imputation(row)
                imputed_rows.append(imputed_row)

            # Convert to DataFrame
            imputed_df = pd.DataFrame(imputed_rows, index=incomplete_X_df.index)

            # Encode using frozen preprocessor
            encoded_X = self._encode_imputed_data(imputed_df)
            # Predict SV for this imputation
            pred_m = self._predict_sv_single_imputation(encoded_X)

            all_predictions.append(pred_m)

            if (m + 1) % 5 == 0:
                print(f"  Completed {m + 1}/{n_imputations} imputations...")
        # ------------------------------
        # Rubin-style aggregation of SV predictions
        # ------------------------------
        predictions_array = np.array(all_predictions)  # (n_imputations, n_samples)

        # Overall mean: average of predictions across imputations
        mean_final = np.mean(predictions_array, axis=0)

        # TotaL variance: variance of predictions across imputations
        # If all rows give identical predictions across imputations, this will be ~0.
        if n_imputations > 1:
            total_var = np.var(predictions_array, axis=0, ddof=1)
        else:
            total_var = np.zeros_like(mean_final)
        # Standard deviation
        std_final = np.sqrt(total_var)

        result_dict = {
            'mean_sv': mean_final,
            'var_sv': total_var,
            'std_sv': std_final,
        }

        print(f"Prediction completed. Mean SV range: [{mean_final.min():.4f}, {mean_final.max():.4f}]")
        print(f"Average uncertainty (std): {std_final.mean():.4f}")
        return result_dict
