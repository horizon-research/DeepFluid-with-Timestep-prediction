import torch
import torch.nn.functional as F
import open3d.ml.torch as ml3d
import numpy as np


class TimestepPredictionNetwork(torch.nn.Module):

    def __init__(
        self,
        kernel_size=[4, 4, 4],
        radius_scale=1.5,
        coordinate_mapping='ball_to_cube_volume_preserving',
        interpolation='linear',
        use_window=True,
        particle_radius=0.025,
        other_feats_channels=0,
    ):
        super().__init__()
        self.layer_channels = [32, 64, 64, 1]
        self.kernel_size = kernel_size
        self.radius_scale = radius_scale
        self.coordinate_mapping = coordinate_mapping
        self.interpolation = interpolation
        self.use_window = use_window
        self.particle_radius = particle_radius
        self.filter_extent = np.float32(self.radius_scale * 6 *
                                        self.particle_radius)

        self._all_convs = []

        def window_poly6(r_sqr):
            return torch.clamp((1 - r_sqr)**3, 0, 1)

        def Conv(name, activation=None, **kwargs):
            conv_fn = ml3d.layers.ContinuousConv

            window_fn = None
            if self.use_window == True:
                window_fn = window_poly6

            conv = conv_fn(kernel_size=self.kernel_size,
                           activation=activation,
                           align_corners=True,
                           interpolation=self.interpolation,
                           coordinate_mapping=self.coordinate_mapping,
                           normalize=False,
                           window_function=window_fn,
                           radius_search_ignore_query_points=True,
                           **kwargs)

            self._all_convs.append((name, conv))
            return conv

        self.conv0_fluid = Conv(name="conv0_fluid",
                                in_channels=4 + other_feats_channels,
                                filters=self.layer_channels[0],
                                activation=None)
        self.conv0_obstacle = Conv(name="conv0_obstacle",
                                   in_channels=3,
                                   filters=self.layer_channels[0],
                                   activation=None)
        self.dense0_fluid = torch.nn.Linear(in_features=4 +
                                            other_feats_channels,
                                            out_features=self.layer_channels[0])
        torch.nn.init.xavier_uniform_(self.dense0_fluid.weight)
        torch.nn.init.zeros_(self.dense0_fluid.bias)

        self.convs = []
        self.denses = []
        for i in range(1, len(self.layer_channels)):
            in_ch = self.layer_channels[i - 1]
            if i == 1:
                in_ch *= 3
            out_ch = self.layer_channels[i]
            dense = torch.nn.Linear(in_features=in_ch, out_features=out_ch)
            torch.nn.init.xavier_uniform_(dense.weight)
            torch.nn.init.zeros_(dense.bias)
            setattr(self, 'dense{0}'.format(i), dense)
            conv = Conv(name='conv{0}'.format(i),
                        in_channels=in_ch,
                        filters=out_ch,
                        activation=None)
            setattr(self, 'conv{0}'.format(i), conv)
            self.denses.append(dense)
            self.convs.append(conv)

    def compute_timestep(self,
                           pos,
                           vel,
                           other_feats,
                           box,
                           box_feats,
                           fixed_radius_search_hash_table=None):

        # compute the extent of the filters (the diameter)
        filter_extent = torch.tensor(self.filter_extent)

        fluid_feats = [torch.ones_like(pos[:, 0:1]), vel]
        if not other_feats is None:
            fluid_feats.append(other_feats)
        fluid_feats = torch.cat(fluid_feats, axis=-1)

        self.ans_conv0_fluid = self.conv0_fluid(fluid_feats, pos, pos,
                                                filter_extent)
        self.ans_dense0_fluid = self.dense0_fluid(fluid_feats)
        self.ans_conv0_obstacle = self.conv0_obstacle(box_feats, box, pos,
                                                      filter_extent)

        feats = torch.cat([
            self.ans_conv0_obstacle, self.ans_conv0_fluid, self.ans_dense0_fluid
        ],
                          axis=-1)

        self.ans_convs = [feats]
        for conv, dense in zip(self.convs, self.denses):
            inp_feats = F.relu(self.ans_convs[-1])
            ans_conv = conv(inp_feats, pos, pos, filter_extent)
            ans_dense = dense(inp_feats)
            if ans_dense.shape[-1] == self.ans_convs[-1].shape[-1]:
                ans = ans_conv + ans_dense + self.ans_convs[-1]
            else:
                ans = ans_conv + ans_dense
            self.ans_convs.append(ans)

        # scale to better match the scale of the output distribution
        self.timestep = torch.mean(self.ans_convs[-1])
        return self.timestep

    def forward(self, inputs, fixed_radius_search_hash_table=None):
        """computes predicted timestep 
        inputs: list or tuple with (pos,vel,feats,box,box_feats)
          pos and vel are the positions and velocities of the fluid particles.
          feats is reserved for passing additional features, use None here.
          box are the positions of the static particles and box_feats are the
          normals of the static particles.
        """
        pos, vel, feats, box, box_feats = inputs

        timestep = self.compute_timestep(
            pos, vel, feats, box, box_feats, fixed_radius_search_hash_table)

        return timestep

    # def init(self, feats_shape=None):
    # """Runs the network with dummy data to initialize the shape of all variables"""
    # pos = np.zeros(shape=(1, 3), dtype=np.float32)
    # vel = np.zeros(shape=(1, 3), dtype=np.float32)
    # if feats_shape is None:
    # feats = None
    # else:
    # feats = np.zeros(shape=feats_shape, dtype=np.float32)
    # box = np.zeros(shape=(1, 3), dtype=np.float32)
    # box_feats = np.zeros(shape=(1, 3), dtype=np.float32)

    # _ = self.__call__((pos, vel, feats, box, box_feats))
