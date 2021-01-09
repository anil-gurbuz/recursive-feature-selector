from sklearn.model_selection import train_test_split
import numpy as np

class recursive_feature_selector():
    def __init__(self, 
                 model, 
                 model_params:dict, 
                 x:pd.DataFrame, 
                 y:np.array, 
                 trial_features:list, 
                 splitter,  
                 eval_func,
                 splitter_params=None,
                 selected_features:list=[],
                 stop_exp_no=float("inf"),
                 valid_and_test=True ) -> None:
        
        """
        Class for running recursive experiments to decide best set of features
        based on the value of evaluation function on validation set.
        
        One of the trial features -- the one leading best performance-- 
        will be selected and added to selected features in each experiment
        until all the features are selected or reached to selected number of experiments.
        
        Arguments:
        model: A sklearn model object
        model_params: Dictionary to pass into fit method of the model
        trial_features: List of features to recursively add to model
        splitter: sklearn splitter object or an index generator for splitting the data into train and validation
        splitter_params: Parameters to pass into split method of splitter object.
        eval_func: A callable method for performance evaluation of features.
        selected_features: Initial set of features.
        stop_exp_no: Max number of experiments to run.
        valid_and_test: Set true if wanted to further split validation into test and validation. Useful for early stopping.
        
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
        
        self.stop_exp_no = stop_exp_no
        self.list_of_trackers = []
        self.valid_and_test = valid_and_test
        

        
        
    def run_one_feature(self, x, y, feature_name, tracker):
        """
        Helper function to run one cycle of cross validation after adding one feature to selected list of features.
        Store the results in Tracker() class object.
        """
        
        # Define set of features will be used in current CV loop
        current_features = self.selected_features + [feature_name]

        # For each fold ...
        for train_idx, valid_idx in self.splitter.split(x, y, **self.splitter_params):
            
            # ... Define training and validation sets
            train_x = x.iloc[train_idx,][current_features]
            train_y = y[train_idx]
            
            # ... Further split validation into test and validation
            if self.valid_and_test:
                valid_x, test_x, valid_y, test_y  = train_test_split(x.iloc[valid_idx,][current_features],y[valid_idx],  test_size=0.5)
            
            
            # ... Initiate model object and train
            model = self.model(**self.model_params)
            model.fit(train_x, train_y , eval_set=(valid_x, valid_y), verbose=False)
            
            # ... Calculate error/objective value
            objective_value = self.eval_func(test_y, model.predict(test_x))
            
            # ... Store the result in Tracker object
            tracker.update_storage(feature_name, objective_value)
            
    
    
    def run_one_experiment(self, select_min=True, exp_no=1):
        """
        Helper function to run CV for all trial features and store results in a Tracker() object.
        """
        
        # Initiate Tracker object
        tracker = Experiment_tracker(self.selected_features, self.trial_features, select_min = select_min)
        
        # For each feature...
        for one_feature in self.trial_features:
            
            # Calculate CV results and store in tracker
            self.run_one_feature(self.x, self.y, one_feature, tracker)
        
        # Identify the feature with best performance
        selected_feature = tracker.end_experiment()
        
        # Print out the results of current level of recursive experiments
        self.print_exp_results(tracker, selected_feature, exp_no)
        
        # Update lists of selected features and trial features
        self.selected_features.append(selected_feature)
        self.trial_features = [f for f in self.trial_features if f not in self.selected_features]
        
        # Store tracker object of current experiment in a list -- Each step of recursive experiments have individual trackers
        self.list_of_trackers.append(tracker)
        
        
    
    def print_exp_results(self, tracker, last_selected_feature, exp_no):
        """
        Helper function to print out results of a step of recursive experiments.
        """
        
        previous_exp_error=float("inf")
        
        # If not first experiment...
        if exp_no !=1:
            # Get the tracker object of previous experiment
            previous_tracker = self.list_of_trackers[-1]
            # Get the error value for comparison of previous experiment results and the current one
            previous_exp_error = previous_tracker.exp_storage[previous_tracker.selected_feature]
            
        
        print("="*50)
        print(f"Experiment {exp_no} results: \n")
        print(f"Initial set of features: {self.selected_features}" )
        print(f"Initial error/objective value: {previous_exp_error:.2f}" )
        print(f"Feature selected at this experiment: {last_selected_feature}")
        print("\n")
        tracker.print_experiment_results(previous_exp_error)
        print("="*50)
        print("*"*50)
            

    
    
    def run_all_experiments(self, select_min=True, exp_no=0):
        """
        Main method to run all the required operation at once. 
        Runs recursive experiments by changing initial set of features and trial features. 
        Stores the validation results in a  list of tracker objects.
        
        Arguments:
        select_min: Indicate if the evaluation metric needs to be minimized or maximized.
        exp_no: Experiment id no show in the print out.
        
        Return:
        Recursive call to until all trial features are passed to selected features.
        """
        # Increase experiment no by one
        exp_no +=1
        
        # If there is no more features to move to selected features...
        if (len(self.trial_features) == 0) or (exp_no > self.stop_exp_no):
            
            # ... Return list of trackers as experiment results
            return self.list_of_trackers
        
        # Otherwise, run experiments for new set of selected features
        self.run_one_experiment(select_min, exp_no)
        
        # Make a recursive call
        return self.run_all_experiments(select_min, exp_no)
        
            

class Experiment_tracker():
    def __init__(self, initial_feature_set, trial_features, select_min=True):
        """
        Class for keeping validation results of one initial set of predictors.
        
        Arguments:
        initial_feature_set: Base set of features to add new set features on.
        trial_features: Set of features to track validation errors.
        select_min: Indicate if the evaluation metric needs to be minimized or maximized.
        """
        
        # Initiate attributes
        self.fold_storage = dict(zip(trial_features, len(trial_features)*[np.zeros(shape=0)]))
        self.trial_features = trial_features
        self.initial_feature_set = initial_feature_set
        self.exp_storage = {}
        self.select_min = select_min
        self.selected_feature = None
        
        
    def update_storage(self, feature_name, value):
        """
        Append error of one fold for existing array storing validation errors for a feature.
        
        Arguments:
        feature_name: Name of the feature to store validation error for.
        value: Objective/error value of one fold of CV for the "feature_name"
        """
        
        self.fold_storage[feature_name] = np.append(self.fold_storage[feature_name], value)
        
        
        
    def end_experiment(self):
        """
        Method to determine feature leading to best performance.
        
        Return:
        Name of the feature giving best performance
        """
        
        # For all features...
        for feature in self.trial_features:
            # ...Calculate average CV error
            avg = np.mean(self.fold_storage[feature])
            # ...Store the average error
            self.exp_storage[feature] = avg
        
        # Select the feature name leading to smallest/biggest error/objective
        self.selected_feature = min(self.exp_storage, key=self.exp_storage.get)
        
        if not self.select_min:
            self.selected_feature = max(self.exp_storage, key=self.exp_storage.get)
            
            
        return self.selected_feature
    
    
    def print_experiment_results(self, previous_exp_error=0):
        """
        Method to print out average errors/objective values and
        amount of increase/decrease compared to previous experiment for each feature in a structured way.
        """
        # Create a list view of the dictionary storing average errors/objectives
        view = [ (v,k) for k,v in self.exp_storage.items() ]
        
        # Sort tuples of the list
        view.sort(reverse= not self.select_min)
        
        # Print out values and improvements compared to previous experiment
        print(" "*27, "Error", "    ","Improvement")
        for v,k in view:
            print(f"{k:25} :  {v:.2f}      {previous_exp_error - v :.2f}")
        

