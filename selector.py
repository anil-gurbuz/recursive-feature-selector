class recursive_feature_selector():
    def __init__(self, model, model_params, x:pd.DataFrame(), y, trial_features, splitter, splitter_params, eval_func, selected_features=[]):
        """
        Class for running recursive experiments to decide best set of features
        based on the evaluation on validation set.
        One of the trial features -- the one leading best performance-- 
        will be selected and added to selected features in each experiment
        until all the features are selected.
        
        Arguments:...
        
        """
        # Initiate the attributes
        self.model = model
        self.model_params = model_params
        self.x = x
        self.y = y
        self.trial_features = trial_features
        self.splitter = splitter
        self.splitter_params = splitter_params
        self.eval_func = eval_func
        
        self.selected_features = selected_features
        self.list_of_trackers = []
        

        
        
    def run_one_feature(self, x, y, feature_name, tracker):
        """
        Run one cycle of cross validation after adding one feature to selected list of features.
        Store the results in Tracker() class object
        
        Arguments:
        
        """
        current_features = self.selected_features + [feature_name]
        
        for train_idx, valid_idx in self.splitter.split(x, y, **self.splitter_params):
            
            train_x = x.iloc[train_idx,][current_features]
            train_y = y[train_idx]
            valid_x = x.iloc[valid_idx,][current_features]
            valid_y = y[valid_idx]
            
            model = self.model(**self.model_params)
            model.fit(train_x, train_y , eval_set=(valid_x, valid_y), verbose=500)
            

            objective_value = self.eval_func(y[valid_idx], model.predict(x.iloc[valid_idx,][current_features]))
            
            tracker.update_storage(feature_name, objective_value)
            
    
    
    def run_one_experiment(self, select_min=True, exp_no=1):
        """
        Run CV for all trial features and store results in a Tracker()
        
        Arguments:
        
        Return:
        
        """
        
        tracker = Experiment_tracker(self.selected_features, self.trial_features, select_min = select_min)
        
        for one_feature in self.trial_features:
            
            self.run_one_feature(self.x, self.y, one_feature, tracker)
        
        
        selected_feature, objective_value = tracker.end_experiment()
        
        print("\n")
        print("="*50)
        print(f"Experiment {exp_no} finished. \n")
        print(f"Initial set of features: {self.selected_features}" )
        print(f"Feature selected at this experiment: {selected_feature} with score: {objective_value:.2f}")
        print("Detailed results:")
        tracker.print_experiment_results()
        print("="*50)
        print("\n")
        
        self.selected_features.append(selected_feature)
        self.trial_features = [f for f in self.trial_features if f not in self.selected_features]
        self.list_of_trackers.append(tracker)
        
        

        
        
    
    
    def run_all_experiments(self, select_min=True, exp_no=0):
        """
        Run recursive experiments for changing initial set of features. 
        Store the validation results in a seperate Tracker() object for each experiment.
        
        Arguments:
        
        
        Return:
        
        """
        exp_no +=1
        
        if len(self.trial_features) == 0:
            return self.list_of_trackers
        
        self.run_one_experiment(select_min, exp_no)
        
        return self.run_all_experiments(select_min, exp_no)
        
            

class Experiment_tracker():
    """
    Class for keeping validation results of one initial set of predictors.
    
    """
    def __init__(self, initial_feature_set, trial_features, select_min=True):
        
        self.fold_storage = dict(zip(trial_features, len(trial_features)*[np.zeros(shape=0)]))
        self.trial_features = trial_features
        self.initial_feature_set = initial_feature_set
        self.exp_storage = {}
        self.select_min = select_min
        
        
    def update_storage(self, feature_name, value):
        self.fold_storage[feature_name] = np.append(self.fold_storage[feature_name], value)
        
        
        
    def end_experiment(self):
        
        for feature in self.trial_features:
            avg = np.mean(self.fold_storage[feature])
            self.exp_storage[feature] = avg
        
        
        selected_feature = min(self.exp_storage, key=self.exp_storage.get)
        objective_value = self.exp_storage[selected_feature]
        
        if not self.select_min:
            selected_feature = max(self.exp_storage, key=self.exp_storage.get)
            objective_value = self.exp_storage[selected_feature]
            
        return selected_feature, objective_value
    
    
    def print_experiment_results(self):
        view = [ (v,k) for k,v in self.exp_storage.items() ]
        view.sort(reverse= not self.select_min)
        
        for v,k in view:
            print(f"{k:15} :  {v:.2f}")
        

