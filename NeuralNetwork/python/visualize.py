
import pickle
import matplotlib.pyplot as plt


def plot(data, height, width):
    img = data.reshape(28, 28)
    plt.matshow(img, cmap='gray')

def plot_figures(figures, nrows = 1, ncols=1):
    fig, axeslist = plt.subplots(ncols=ncols, nrows=nrows)
    for ind,title in zip(range(len(figures)), figures):
        axeslist.ravel()[ind].imshow(figures[title], cmap=plt.gray())
        #axeslist.ravel()[ind].set_title(title)
        axeslist.ravel()[ind].set_axis_off()
    plt.savefig('params_2layers_3')


def main():
    pickle_path = 'nn_params.mat'
    pickle_file = open(pickle_path, 'rb')
    params = pickle.load(pickle_file)
    w = params[3]['w']
    figures = {}
    for i in range(w.shape[1]):
        img = w[:,i]
        figures['param'+str(i)] = img.reshape(28, 28)
    plot_figures(figures, 10, 10)
    pickle_file.close()
    return 0

if __name__ == '__main__':
  main()