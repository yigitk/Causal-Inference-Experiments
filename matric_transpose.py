import numpy as np

class gsl_block_struct(object):
    __slots__ = ['size', 'data']

gsl_block = gsl_block_struct

NULL_MATRIX_VIEW = [[0, 0, 0, 0, 0, 0]]
NULL_MATRIX = [0, 0, 0, 0, 0, 0]
GSL_SUCCESS = 1
MULTIPLICITY = 1

class gsl_matrix(object):
    __slots__ = ['size1',
                 'size2',
                 'tda',
                 'data',
                 'block',
                 'owner']

class _gsl_matrix_view(object):
     __slots__ = ['matrix']

gsl_matrix_view = _gsl_matrix_view

def gsl_matrix_view_array(array, n1, n2):
    view = _gsl_matrix_view()
    view.matrix = gsl_matrix()
    view.matrix.size1=0
    view.matrix.size2 = 0
    view.matrix.tda = 0
    view.matrix.data = 0
    view.matrix.block = 0
    view.matrix.owner = 0
    m = gsl_matrix()
    m.size1, m.size2, m.tda, m.data, m.block, m.owner = NULL_MATRIX
    m.data = array
    m.size1 = n1
    m.size2 = n2
    m.tda = n2
    m.block = 0
    m.owner = 0
    view.matrix = m
    return view

def gsl_matrix_transpose(m):
    
    size1 = m.size1
    size2 = m.size2

    if size1 != size2:
        print("matrix must be square to take transpose")
    
    for i in range(0, size1):
        for j in range(i + 1, size2):
            for k in range(0, MULTIPLICITY):
                e1 = (i *  m.tda + j) * MULTIPLICITY + k
                e2 = (j *  m.tda + i) * MULTIPLICITY + k
                tmp = m.data[e1]
                m.data[e1] = m.data[e2]
                m.data[e2] = tmp

    return GSL_SUCCESS

good_dict = {}
args0 = []

for i in range(0, 1000):
    a_data = np.random.uniform(low=0.0, high=99.9, size=(64,))
    args0.append(a_data)
    m = gsl_matrix_view_array(a_data, 8, 8)
    gsl_matrix_transpose(m.matrix)
    good_dict[i] = (m.matrix.data[0], m.matrix.data[1])


a_data = np.random.uniform(low=0.0, high=99.9, size=(64,))
m = gsl_matrix_view_array(a_data, 8, 8)
print(m.matrix.data)
gsl_matrix_transpose(m.matrix)

print(m.matrix.data)