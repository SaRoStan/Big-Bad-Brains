import numpy as np

EXPERIMENTS = {
    'MOTOR'      : {'cond':['lf','rf','lh','rh','t','cue']},
    'WM'         : {'cond':['0bk_body','0bk_faces','0bk_places','0bk_tools','2bk_body','2bk_faces','2bk_places','2bk_tools']},
    'EMOTION'    : {'cond':['fear','neut']},
    'GAMBLING'   : {'cond':['loss','win']},
    'LANGUAGE'   : {'cond':['math','story']},
    'RELATIONAL' : {'cond':['match','relation']},
    'SOCIAL'     : {'cond':['ment','rnd']}
}

def sort_evals_descending(evals, evectors):

    index = np.flip(np.argsort(evals))
    evals = evals[index]
    evectors = evectors[:, index]
    if evals.shape[0] == 2:
        if np.arccos(np.matmul(evectors[:, 0],
                            1 / np.sqrt(2) * np.array([1, 1]))) > np.pi / 2:
            evectors[:, 0] = -evectors[:, 0]
        if np.arccos(np.matmul(evectors[:, 1],
                            1 / np.sqrt(2) * np.array([-1, 1]))) > np.pi / 2:
            evectors[:, 1] = -evectors[:, 1]
    return evals, evectors

def get_the_eigen_matrix(data):
    """
    get eigenvalues and vectors
    """
    mean_data = data - np.mean(data, 0)
    relation_pre_data = mean_data.T @ mean_data / mean_data.shape[0]
    # calculate the eigenvalue and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(relation_pre_data)
    eigenvalues, eigenvectors = sort_evals_descending(eigenvalues, eigenvectors)
    return eigenvalues, eigenvectors
    
def pca(data, eigenvectors, k=1024):
    """
    reduce the data dimensions
    (N,features) -> (N,k)
    """
    score = (data @ eigenvectors[:,:k]).real
    row_sums = score.sum(axis=1)
    score = score / row_sums[:, np.newaxis]
    return score

def get_data_with_window(data, end_pos, hop_size=5, win_size=10):
    """
    slicing the data 
        data - the input data matrix
        end_pos - the end position of the matrix for slicing
        win_size - window size
    """
    row = data.shape[0]
    num_data_items = (end_pos - win_size) // hop_size + 1
    res = np.empty((num_data_items,row*win_size))
    for i in range(num_data_items):
        res[i] = data[:,i*hop_size:i*hop_size+win_size].reshape((-1))
    return res

def load_flatten_data(task,sub_idx, row=360, col=232):
    """
    Load data for one subject and slice based on a window
        task - str, name of your task eg. RELATIONAL
        sub_idx - int
        (row, col) - the shape of your data matrix
    """
    # loading data
    HCP_DIR = "./hcp_task"
    subjects = np.loadtxt(os.path.join(HCP_DIR, 'subjects_list.txt'), dtype='str')
    sub = subjects[sub_idx]
    subj_data = np.empty((2,row,col))
    for i_run,run in enumerate(RUNS):

        path = os.path.join(HCP_DIR, 'subjects', sub, task, 'tfMRI_'+task+'_'+run, 'data.npy')
        temp = np.load(path)
        subj_data[i_run] = temp
    subj_data = np.concatenate((subj_data[0],subj_data[1]),axis=1)
    print(subj_data.shape)
    
    # slicing the data
    data = get_data_with_window(subj_data, col*2)
        
    return data

if __name__ == '__main__':
    data = load_flatten_data('RELATIONAL',0)
    eigenvalues, eigenvectors = get_the_eigen_matrix(data)
    pca(data, eigenvectors)


