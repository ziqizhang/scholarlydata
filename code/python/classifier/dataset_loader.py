

# GLOBAL VARIABLES
#DATA_ORG = "/home/zqz/Work/scholarlydata/data/train/training_org(expanded)_features_o.csv"
#TASK_NAME = "scholarlydata_org"
#DATA_COLS_START = 3  # inclusive
#DATA_COLS_END = 20  # exclusive 16
#DATA_COLS_FT_END = 16  # exclusive 12
#DATA_COLS_TRUTH = 16  # inclusive 12


def create_dataset_props(task, feature_file, feature_col_start, feature_col_end, feature_size, truth_col, appended_list):
    props=(task, feature_file, feature_col_start, feature_col_end, feature_size, truth_col)
    appended_list.append(props)

def load_exp_datasets():
    l=list()
    load_org_datasets(l)
    load_per_datasets(l)
    return l

def load_org_datasets(list):
    create_dataset_props("scholarlydata_org",
                         "/home/zqz/Work/scholarlydata/data/train/training_org(expanded)_features_o.csv",
                         3,20,16,16, list) #this is the original feature set WITHOUT non-presence feature or normalisation
    create_dataset_props("scholarlydata_org",
                         "/home/zqz/Work/scholarlydata/data/train/training_org(expanded)_features_o.csv",
                         3,24,20,20, list) #this is the original feature set WITH non-presence feature or normalisation

def load_per_datasets(list):
    create_dataset_props("scholarlydata_per",
                         "/home/zqz/Work/scholarlydata/data/train/training_per_features.csv",
                         3,36,32,32, list) #this is the original feature set WITHOUT non-presence feature or normalisation
    create_dataset_props("scholarlydata_per",
                         "/home/zqz/Work/scholarlydata/data/train/training_per_features.csv",
                         3,44,40,40, list) #this is the original feature set WITH non-presence feature or normalisation
