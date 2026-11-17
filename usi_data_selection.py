import numpy as np
class USI_sampler:
    def __init__(self, dataset_config):
        self.incomplete_trnX = np.asarray(dataset_config['incomplete_trnX'])
        self.incomplete_trnY = np.asarray(dataset_config['incomplete_trnY'])
        self.incomplete_trnX_idx = dataset_config['incomplete_trnX_idx']
        self.total_incomplete_select_ratio = dataset_config['total_incomplete_select_ratio']
        self.normalised_missing_FI_scores = dataset_config.get('normalised_missing_FI_scores', None)

        self.sv_sampler_config = {
            'descending': True,
            'absolute': False,
            # weights for final_eval_score when sv_score_only=False
            'w_sv': 1.0,
            'w_unc': 1.0,
            'w_mfi': 1.0,
        }

    def update_sampler_config(self, new_config):
        self.sv_sampler_config.update(new_config)

        self.descending = self.sv_sampler_config['descending']
        self.absolute = self.sv_sampler_config['absolute']

        self.w_sv = self.sv_sampler_config.get('w_sv', 1.0)
        self.w_unc = self.sv_sampler_config.get('w_unc', 1.0)
        self.w_mfi = self.sv_sampler_config.get('w_mfi', 1.0)

    def _compute_final_eval_scores(self, result_dict):
        """
        If sv_score_only=True:
            return mean_sv (optionally |mean_sv|).

        Otherwise:
            return final_eval_score based on:
                w_sv  * norm(mean_sv)
              - w_unc * norm(std_sv)
              - w_mfi * normalised_missing_FI_scores
        """

        mean_sv = np.asarray(result_dict['mean_sv'])
        std_sv  = np.asarray(result_dict['std_sv'])   # <-- UNCERTAINTY HERE
        mfi     = np.asarray(self.normalised_missing_FI_scores)

        # 1) absolute, if requested
        if self.absolute:
            mean_sv = np.abs(mean_sv)

        print(
            "Using final eval score: "
            f"{self.w_sv} * norm(SV) "
            f"- {self.w_unc} * norm(std_SV) "
            f"- {self.w_mfi} * norm(missing_FI)"
        )

        # normalise mean_sv to [0,1]
        sv_min, sv_max = mean_sv.min(), mean_sv.max()
        if sv_max > sv_min:
            norm_sv = (mean_sv - sv_min) / (sv_max - sv_min)
        else:
            norm_sv = np.zeros_like(mean_sv)

        # normalise std_sv to [0,1]
        std_min, std_max = std_sv.min(), std_sv.max()
        if std_max > std_min:
            norm_std = (std_sv - std_min) / (std_max - std_min)
        else:
            norm_std = np.zeros_like(std_sv)

        # mfi assumed already in [0,1]; clip just in case
        mfi = np.clip(mfi, 0.0, 1.0)

        # print(f'Normalised SV   range: [{norm_sv.min():.4f}, {norm_sv.max():.4f}]')
        # print(f'Normalised std  range: [{norm_std.min():.4f}, {norm_std.max():.4f}]')
        # print(f'Normalised MFI  range: [{mfi.min():.4f}, {mfi.max():.4f}]')

        final_scores = (
            self.w_sv  * norm_sv
            - self.w_unc * norm_std
            - self.w_mfi * mfi
        )

        print(f'Final eval range:       [{final_scores.min():.4f}, {final_scores.max():.4f}]')
        return final_scores

    def sv_incomplete_sampler(self, new_config, result_dict):
        """
        result_dict:
            {
                'mean_sv': np.array(n_incomplete,),
                'var_sv':  np.array(n_incomplete,),  # unused here
                'std_sv':  np.array(n_incomplete,),
                ...
            }
        """
        self.update_sampler_config(new_config)
        print(f'ABS={self.absolute}, DESC={self.descending}')

        # 1) compute scores according to config
        scores = self._compute_final_eval_scores(result_dict)

        # 2) sort by scores
        sorted_pairs = sorted(
            zip(scores, self.incomplete_trnX_idx),
            key=lambda x: x[0],
            reverse=self.descending
        )
        sorted_scores, sorted_trnX_idx = zip(*sorted_pairs)

        # 3) select top-K according to total_incomplete_select_ratio
        tmp_select_num = int(len(self.incomplete_trnX_idx) * self.total_incomplete_select_ratio)

        selected_incomplete_trnX_idx = sorted_trnX_idx[:tmp_select_num]
        selected_incomplete_pos = [self.incomplete_trnX_idx.index(idx) for idx in selected_incomplete_trnX_idx]

        selected_incomplete_trnX = self.incomplete_trnX[selected_incomplete_pos]
        selected_incomplete_trnY = self.incomplete_trnY[selected_incomplete_pos]

        selected_incomplete_pos_idx_dict = {
            'selected_incomplete_pos': selected_incomplete_pos,
            'selected_incomplete_trnX_idx': selected_incomplete_trnX_idx,
            'sv_coreset_trnX_idx': result_dict.get('coreset_trnX_idx', None),
        }

        return selected_incomplete_trnX, selected_incomplete_trnY, selected_incomplete_pos_idx_dict
