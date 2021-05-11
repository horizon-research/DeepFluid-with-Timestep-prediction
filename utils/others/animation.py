import os
import numpy as np
from matplotlib import pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib import animation
import zstandard as zstd
from glob import glob
import msgpack
import msgpack_numpy
msgpack_numpy.patch()
import pickle


def get_timestep_data(result_dir):
    """
    generate a list of arrays of positions of animated points

    Args:
        result_dir : directory consisting of .npz and .ply files
    
    Returns:
        fluid: The list of arrays of positions of animated points
    """
    files = os.listdir(result_dir)
    files = list(filter(lambda f: '.npz' in f and 'fluid' in f, files))
    files.sort()
    print(files)
    fluid = []
    for f in files:
        data = np.load(os.path.join(result_dir, f))['pos']
        # the y value in data is actually the z-value in 3D visualization
        fluid.append([data[:, 0], data[:, 2], data[:, 1]])
    return fluid


def draw_animation_scatter(all_timesteps):
    """
    draw animation in scatter graph
    :param all_timesteps: pos data of all timesteps,
    basically, [timestep0, timestep1, ..., timestepn]
    each timestep is in the form of [[p1_x, p2_x, ..., pn_x], [p1_y, p2_y, ..., pn_y], [p1_z, p2_z, ..., pn_z]]
    :return:
    """
    x = all_timesteps[0][0]
    y = all_timesteps[0][1]
    z = all_timesteps[0][2]

    def update_graph(frame):
        title.set_text('timestep={:d}'.format(frame))
        new_x = all_timesteps[frame][0]
        new_y = all_timesteps[frame][1]
        new_z = all_timesteps[frame][2]
        graph._offsets3d = (new_x, new_y, new_z)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    title = ax.set_title('')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_xlim3d([-1, 1])
    ax.set_ylim3d([-1, 1])
    ax.set_zlim3d([0, 4])
    ax.view_init(elev=10, azim=30)

    graph = ax.scatter(x, y, z, 'o', alpha=0.2, edgecolors='k')

    ani = animation.FuncAnimation(fig, update_graph, frames=len(all_timesteps))
    plt.show()


def visualize_prediction(prediction_dir):
    all_timesteps = get_timestep_data(prediction_dir)
    draw_animation_scatter(all_timesteps)


def visualize_trainset(trainset_dir):
    (path, dir_name) = os.path.split(trainset_dir)
    cache = path + '.pkl'

    if os.path.exists(cache):
        print('loading fluid position data from cache...', end='')
        with open(cache, 'rb') as f:
            fluid = pickle.load(f)
        print('done')
    else:
        print('decoding fluid position data...', end='')
        fluid = []

        train_files = sorted(glob(os.path.join(trainset_dir, '*.zst')))
        files_idxs = np.arange(len(train_files))

        decompressor = zstd.ZstdDecompressor()

        for file_i in files_idxs:
            with open(train_files[file_i], 'rb') as f:
                data = msgpack.unpackb(decompressor.decompress(f.read()), raw=False)
                for each_frame in data:
                    position = each_frame['pos']
                    fluid.append([position[:, 0], position[:, 2], position[:, 1]])
        print('done')
        print('dump data into local cache...', end='')
        with open(cache, 'wb') as f:
            pickle.dump(fluid, f)
        print('done')

    fluid = [fluid[i] for i in range(0, len(fluid), 30)]
    draw_animation_scatter(fluid)


if __name__ == '__main__':
    # visualize_prediction('../visualization_data/model1_y2.7_7956p')
    # visualize_prediction('../visualization_data/model2_y2.7_7956p')
    # visualize_prediction('../visualization_data/pretrained_y2.7_7956p')
    # visualize_trainset('../../data_200FPS/train')
    visualize_prediction('../visualization_data/upper_out_1.7')
    # visualize_prediction('../visualization_data/upper_out_2.7')
    # visualize_prediction('../visualization_data/upper_out_3.7')