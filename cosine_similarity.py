'''
Measure the correlation between the discovered directions.
'''
import matplotlib.pyplot as plt
import numpy as np
import torch
import argparse

from directions import ATTRIBUTES

#ATTRIBUTES_CELEBAHQ = {'pose' : '0-3', 'gender' : '0-1', 'age' : '5-7',
#                        'eyeglasses' : '0-1', 'smile' : '2-3'}
#ATTRIBUTES_FFHQ = {'pose' : '0-6', 'gender' : '2-4', 'age' : '2,4,5,6',
#                        'eyeglasses' : '0-2', 'smile' : '3'}


def cosine_similarity(args):
    attribute_names = ['pose', 'gender', 'age', 'eyeglasses', 'smile']
    attribute_vectors = ATTRIBUTES[args.model_name]
    
    # load the 5 attribute vectors according to the method name specified
    attr_vectors = []
    #layer_indices = []
    for attr_name in attribute_names:
        attr_vector = np.load(f'{args.semantics_dir}/{args.method_name}/'
                              f'{args.model_name}_{attr_name}.npy')
        if attr_vector.ndim == 2:
            attr_vector = np.squeeze(attr_vector, axis=0)
        #layer_idx = parse_indices(val, min_val=0, max_val=G.num_layers - 1)
        attr_vectors.append(attr_vector / np.linalg.norm(attr_vector))
        #layer_indices.append(layer_idx)

    results = []
    for idx, attr_vector in enumerate(attr_vectors):
        sims = []
        for attr_vector_d in attr_vectors:
            sims.append(attr_vector.dot(attr_vector_d))

        results.append(sims)

    attribute_names = [attr_name.capitalize() for attr_name in attribute_names]

    # table
    fig, ax = plt.subplots()

    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.table(cellText=results, rowLabels=attribute_names,
            colLabels=attribute_names, loc='center')

    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Cosine similarity.')
    parser.add_argument('model_name', type=str, help='Name of the'
                        ' pre-trained GAN model.')
    parser.add_argument('method_name', type=str, choices=['mddgan', 'interfacegan', 'sefa'],
                        help='Study the correlation between the directions'
                             ' discovered by the selected method.')
    parser.add_argument('--semantics_dir', type=str, default='semantics',
                        help='Path to the discovered directions directory'
                        ' (default: %(default)s).')
    args = parser.parse_args()
    cosine_similarity(args)
