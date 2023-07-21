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
    mean_data = data - np.mean(data, 0)
    relation_pre_data = mean_data.T @ mean_data / mean_data.shape[0]
    # calculate the eigenvalue and eigenvectors
    eigenvalues, eigenvectors = np.linalg.LA.eig(relation_pre_data)
    eigenvalues, eigenvectors = sort_evals_descending(eigenvalues, eigenvectors)
    return eigenvalues, eigenvectors
def pca(data, eigenvectors, k=4096):
    score = data @ eigenvectors[:,:k]
    return score

def load_flatten_data(task,num_conditions,num_subj = 100, size=360):
    data = np.empty((num_subj,num_conditions*size*size))
    for sub in range(num_subj):
        path = "../featureExtr/"+task+"/sub_"+sub+"_"+task+"_"+EXPERIMENTS[task]['cond'][0]+".npy"
        subj_data =  np.load(path).reshape((-1))
        
        for con in range(1,num_conditions):
            path = "../featureExtr/"+task+"/sub_"+sub+"_"+task+"_"+EXPERIMENTS[task]['cond'][con]+".npy"
            subj_data = np.concatenate(subj_data,np.load(path).reshape((-1)))
        data[sub] = subj_data
    return data

if __name__ == '__main__':
    data = load_flatten_data('RELATIONAL',2)
    eigenvalues, eigenvectors = get_the_eigen_matrix(data)
    pca(data, eigenvectors)


