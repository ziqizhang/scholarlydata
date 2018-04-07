

# GLOBAL VARIABLES
#DATA_ORG = "/home/zqz/Work/scholarlydata/data/train/training_org(expanded)_features_o.csv"
#TASK_NAME = "scholarlydata_org"
#DATA_COLS_START = 3  # inclusive
#DATA_COLS_END = 20  # exclusive 16
#DATA_COLS_FT_END = 16  # exclusive 12
#DATA_COLS_TRUTH = 16  # inclusive 12


def create_dataset_props(task, identifier, feature_file, feature_col_start, feature_col_end, feature_size, truth_col, appended_list):
    props=(task, identifier, feature_file, feature_col_start, feature_col_end, feature_size, truth_col)
    appended_list.append(props)

def load_exp_datasets():
    l=list()
    #load_org_datasets(l)
    load_per_datasets(l)
    return l

def load_org_datasets(list):
    create_dataset_props("scholarlydata_org", "original ",
                         "/home/zqz/Work/scholarlydata/data/train/training_org(expanded)_features_o.csv",
                         3,20,16,16, list) #3,20,16,16
    #this is the original feature set WITHOUT non-presence feature or normalisation
    create_dataset_props("scholarlydata_org", "stringsim ",
                         "/home/zqz/Work/scholarlydata/data/train/training_org(expanded)_features_s.csv",
                         3,24,20,20, list)  #3,28,24,24
    #this is the original feature set WITH non-presence feature or normalisation
    create_dataset_props("scholarlydata_org", "normalized ",
                         "/home/zqz/Work/scholarlydata/data/train/training_org(expanded)_features_n.csv",
                         3,20,16,16, list) #3,20,16,16
    create_dataset_props("scholarlydata_org", "presence ",
                         "/home/zqz/Work/scholarlydata/data/train/training_org(expanded)_features_p.csv",
                         3,24,20,20, list)  #3,24,20,20
    create_dataset_props("scholarlydata_org", "presence_stringsim ",
                         "/home/zqz/Work/scholarlydata/data/train/training_org(expanded)_features_ps.csv",
                         3,28,24,24, list)
    #this is the original feature set WITH non-presence feature or normalisation
    create_dataset_props("scholarlydata_org", "presence_norm_stringsim ",
                         "/home/zqz/Work/scholarlydata/data/train/training_org(expanded)_features_pns.csv",
                         3,28,24,24, list) #3, 32, 28,28
    #this is the original feature set WITH non-presence feature or normalisation

def load_per_datasets(list):
    # create_dataset_props("scholarlydata_per","original ",
    #                      "/home/zqz/Work/scholarlydata/data/train/training_per_features_o.csv",
    #                      3,36,32,32, list) #3, 36,32,32
    # #this is the original feature set WITHOUT non-presence feature or normalisation
    # create_dataset_props("scholarlydata_per","stringsim  ",
    #                      "/home/zqz/Work/scholarlydata/data/train/training_per_features_s.csv",
    #                      3,43,39,39, list) #3,50,46,46
    # create_dataset_props("scholarlydata_per","norm ",
    #                      "/home/zqz/Work/scholarlydata/data/train/training_per_features_n.csv",
    #                      3,36,32,32, list) #3,36,32,32
    #this is the original feature set WITHOUT non-presence feature or normalisation
    create_dataset_props("scholarlydata_per","presence ",
                         "/home/zqz/Work/scholarlydata/data/train/training_per_features_p.csv",
                         3,44,40,40, list) #3,44,40,40
    #this is the original feature set WITH non-presence feature or normalisation
    create_dataset_props("scholarlydata_per","presence_stringsim ",
                         "/home/zqz/Work/scholarlydata/data/train/training_per_features_ps.csv",
                         3,51,47,47, list)
    create_dataset_props("scholarlydata_per","norm_presence_stringsim ",
                         "/home/zqz/Work/scholarlydata/data/train/training_per_features_pns.csv",
                         3,51,47,47, list) #3,44,40,40
    return list
