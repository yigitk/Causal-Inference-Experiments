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
    view_matrix = gsl_matrix()
    view.matrix = view_matrix
    view_matrix_size1 = 0
    view.matrix.size1 = view_matrix_size1
    view_matrix_size2 = 0
    view.matrix.size2 = view_matrix_size2
    view_matrix_tda = 0
    view.matrix.tda = view_matrix_tda

    view_matrix_data = 0
    view.matrix.data = view_matrix_data
    view_matrix_block = 0
    view.matrix.block = view_matrix_block
    view_matrix_owner = 0
    view.matrix.owner = view_matrix_owner
    m = gsl_matrix()
    m.size1, m.size2, m.tda, m.data, m.block, m.owner = NULL_MATRIX
    m_data = array
    m.data = m_data
    m_size1 = n1
    m.size1 = m_size1
    m_size2 = n2
    m.size2 = m_size2
    m_tda = n2
    m.tda = m_tda

    m_block = 0
    m.block = m_block

    m_owner = 0
    m.owner = m_owner

    view_matrix = m
    view.matrix = view_matrix
    return view


def gsl_matrix_transpose(m):
    m_size1 = m.size1
    size1 = m_size1
    m_size2 = m.size2
    size2 = m_size2

    if size1 != size2:
        print("matrix must be square to take transpose")

    for i in range(0, size1):
        for j in range(i + 1, size2):
            for k in range(0, MULTIPLICITY):
                m_tda = m.tda
                e1 = (i * m_tda + j) * MULTIPLICITY + k

                m_tda = m.tda
                e2 = (j * m_tda + i) * MULTIPLICITY + k
                m_data_e1 = m.data[e1]
                tmp = m_data_e1
                m_data_e2 = m.data[e2]
                m_data_e1 = m_data_e2
                m.data[e1] = m_data_e1
                m_data_e2 = tmp
                m.data[e2] = m_data_e2
    return GSL_SUCCESS


good_dict = {}
args0 = []

for i in range(0, 1000):
    a_data = np.random.uniform(low=0.0, high=99.9, size=(64,))
    args0.append(a_data)
    m = gsl_matrix_view_array(a_data.copy(), 8, 8)
    gsl_matrix_transpose(m.matrix)
    good_dict[i] = (m.matrix.data[0], m.matrix.data[1],
                    m.matrix.data[8], m.matrix.data[63])
