import numpy as np

class GP:
    
    def __init__(self, config, sigma):

        self.config = config
        self.sigma = np.asarray(sigma)
        
        self.use_sparse = config.getboolean("SPARSE", "use_sparse", fallback=False)
        
        # Only import when needed
        if self.use_sparse:
            from src.multivar_gp.sparse_gp import SparseGP
            self.gp_impl = SparseGP(config, sigma)
            self.model_type = "sparse"
        else:
            from src.multivar_gp.dense_gp import DenseGP
            self.gp_impl = DenseGP(config, sigma)
            self.model_type = "dense"

    def __repr__(self):
        return f"GP(sparse: {self.use_sparse}), kernel: {self.config.get('KERNEL', 'type')}, gpu: {self.config.get('PARALLEL', 'use_gpu')}"
    
    def fit(self, X, y):
        if self.use_sparse:
            inducing_method = self.config.get("SPARSE", "inducing_method", fallback="random")
            self.gp_impl.fit(X, y, inducing_method=inducing_method)
        else:
            self.gp_impl.fit(X, y)
        
        return self
    
    def predict(self, X_test, return_std=False):
        return self.gp_impl.predict(X_test, return_std=return_std)
    
    def eval_fit(self, y_pred, y_true):
        return self.gp_impl.eval_fit(y_pred, y_true)
    
    def get_cache_stats(self):
        return self.gp_impl.get_cache_stats()
    
    def clear_cache(self):
        return self.gp_impl.clear_cache()
    
    def get_model_info(self):
        info = {
            'model_type': self.model_type,
            'use_sparse': self.use_sparse
        }
        
        if self.use_sparse:
            sparse_info = self.gp_impl.get_sparse_info()
            if sparse_info:
                info.update(sparse_info)
                info['inducing_method'] = self.config.get("SPARSE", "inducing_method", fallback="random")
        
        return info
    
    def get_model_complexity(self):
        if self.use_sparse:
            num_inducing = self.config.getint("SPARSE", "num_inducing", fallback=20)
            n_params = len(self.sigma) + 1  # sigmas + lambda
            if self.config.get("KERNEL", "type") == "RQ":
                n_params += 1  # alpha parameter
            n_params += num_inducing * len(self.sigma)
            return n_params
        else:
            if hasattr(self.gp_impl, 'get_model_complexity'):
                return self.gp_impl.get_model_complexity()
            else:
                n_params = len(self.sigma) + 1  # sigmas + lambda
                if self.config.get("KERNEL", "type") == "RQ":
                    n_params += 1  # alpha parameter
                return n_params 