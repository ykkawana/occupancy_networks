import torch
from torch import nn
import math
from periodic_shapes.models import model_utils
from im2mesh.layers import (ResnetBlockFC, CResnetBlockConv1d, CBatchNorm1d,
                            CBatchNorm1d_legacy, ResnetBlockConv1d)


class SuperShapes(nn.Module):
    def __init__(
        self,
        max_m,
        n_primitives,
        latent_dim=100,
        rational=True,
        train_logits=True,
        train_linear_scale=True,
        quadrics=False,
        sphere=False,
        train_ab=True,
        use_paramnet=True,
        paramnet_dense=True,
        transition_range=1.,
        paramnet_class='ParamNet',
        paramnet_hidden_size=128,
        is_single_paramnet=False,
        layer_depth=0,
        skip_position=3,  # count start from input fc
        is_skip=True,
        supershape_freeze_rotation_scale=False,
        get_features_from=[],
        dim=2):
        """Initialize SuperShapes.

        Args:
            max_m: Max number of vertices in a primitive. If quadrics is True,
                then max_m must be larger than 4.
            n_primitives: Number of primitives to use.
            rational: Use rational (stable gradient) version of supershapes
            train_logits: train logits flag.
            train_linear_scale: train linear scale flag.
            quadrics: quadrics mode flag. In this mode, n1, n2, n3 are set
                to one and not trained, and logits is not trained and 5th 
                element of m_vector is always one and the rest are zero.
            dim: if dim == 2, then 2D mode. If 3, 3D mode.
        Raises:
            NotImplementedError: when dim is neither 2 nor 3.
        """
        super().__init__()
        self.n_primitives = n_primitives
        self.max_m = max_m + 1
        self.rational = rational
        self.quadrics = quadrics
        self.sphere = sphere
        self.train_logits = train_logits
        self.train_ab = train_ab
        self.dim = dim
        self.transition_range = transition_range
        self.latent_dim = latent_dim
        self.use_paramnet = use_paramnet
        self.is_single_paramnet = is_single_paramnet
        self.get_features_from = get_features_from

        if get_features_from:
            assert self.use_paramnet

        if not self.dim in [2, 3]:
            raise NotImplementedError('dim must be either 2 or 3.')
        if self.rational:
            self.n23scale = 10.
            self.n23bias = 1.
        else:
            self.n23scale = 1.
            self.n23bias = 0.

        if self.quadrics:
            self.n23scale = 1.
            self.n23bias = 1e-7

        if self.quadrics:
            assert max_m >= 4, 'super quadrics musth have m bigger than 4.'

        if self.quadrics and self.train_logits:
            raise ValueError(
                'If quadrics is true, then train_logits has to be False.')

        self.train_n1 = not self.quadrics

        self.rot_dim = 1 if self.dim == 2 else 4  # 1 for euler angle for 2D, 4 for quaternion for 3D
        if self.is_single_paramnet:
            assert not supershape_freeze_rotation_scale, 'supershape_freeze_rotation_scale doesnt support single paramnet'
            self.paramnet = paramnet_dict[paramnet_class](
                self.n_primitives,
                self.latent_dim,
                self.dim + self.rot_dim + self.dim + 1,
                hidden_size=paramnet_hidden_size,
                layer_depth=layer_depth,
                dense=paramnet_dense)
        else:
            self.transition_net = paramnet_dict[paramnet_class](
                self.n_primitives,
                self.latent_dim,
                self.dim,
                hidden_size=paramnet_hidden_size,
                layer_depth=layer_depth,
                dense=paramnet_dense)
            self.rotation_net = paramnet_dict[paramnet_class](
                self.n_primitives,
                self.latent_dim,
                self.rot_dim,
                hidden_size=paramnet_hidden_size,
                layer_depth=layer_depth,
                dense=paramnet_dense)
            self.scale_net = paramnet_dict[paramnet_class](
                self.n_primitives,
                self.latent_dim,
                self.dim,
                hidden_size=paramnet_hidden_size,
                layer_depth=layer_depth,
                dense=paramnet_dense)
            if supershape_freeze_rotation_scale:
                self.scale_net.requires_grad = False
                self.rotation_net.requires_grad = False

            self.prob_net = paramnet_dict[paramnet_class](
                self.n_primitives,
                self.latent_dim,
                1,
                hidden_size=paramnet_hidden_size,
                layer_depth=layer_depth,
                dense=paramnet_dense)

        # Pose params
        # B=1, n_primitives, 2
        self.transition = nn.Parameter(torch.Tensor(1, n_primitives, self.dim))

        self.rotation = nn.Parameter(
            torch.Tensor(1, n_primitives, self.rot_dim))

        # B=1, n_primitives, 2
        self.linear_scale = nn.Parameter(
            torch.Tensor(1, n_primitives, self.dim))
        self.linear_scale.requires_grad = train_linear_scale

        # B=1, n_primitives
        self.prob = nn.Parameter(torch.Tensor(1, n_primitives, 1))

        if self.use_paramnet:
            self.linear_scale.requires_grad = False
            self.transition.requires_grad = False
            self.rotation.requires_grad = False
            self.prob.requires_grad = False

        if not self.sphere:
            # output has to be rehaped to (B, n_primitives, max_m, dim - 1)
            self.logits_net = ParamNet(self.n_primitives,
                                       self.latent_dim,
                                       self.max_m * (dim - 1),
                                       dense=paramnet_dense)
            # 5 for a, b, n1, n2, n3
            self.abn1n2n3_dim = 5
            # output has to be rehaped to (B, n_primitives, abn1n2n3, dim - 1)
            self.supershape_param_net = ParamNet(
                self.n_primitives,
                self.latent_dim, (self.abn1n2n3_dim * (self.dim - 1)),
                dense=paramnet_dense)
            self.abn1n2n3_mask = torch.ones(self.abn1n2n3_dim).float().view(
                1, 1, self.abn1n2n3_dim, 1)
            self.no_train_ab_mask = torch.tensor(
                [0., 0., 1., 1., 1.]).float().view(1, 1, self.abn1n2n3_dim, 1)
            self.no_train_n1_mask = torch.tensor(
                [1., 1., 0., 1., 1.]).float().view(1, 1, self.abn1n2n3_dim, 1)

            # 1, n_primitives, max_m, dim-1
            logits_list = []
            for idx in range(self.dim - 1):
                if self.quadrics:
                    logits = torch.eye(self.max_m).view(
                        1, 1, self.max_m, self.max_m)[:, :, 4, :].repeat(
                            1, n_primitives, 1).float() * 10
                elif not self.train_logits:
                    logits = torch.eye(self.max_m).view(
                        1, 1, self.max_m,
                        self.max_m)[:, :, (self.max_m - 1), :].repeat(
                            1, n_primitives, 1).float() * 10
                else:
                    logits = torch.Tensor(1, n_primitives, self.max_m)
                logits_list.append(logits)
            self.logits = torch.stack(logits_list, axis=-1)
            assert [*self.logits.shape
                    ] == [1, n_primitives, self.max_m, self.dim - 1]
            if self.train_logits:
                self.logits = nn.Parameter(self.logits)

            # Shape params
            # B=1, n_primitives, 1, 1
            self.n1 = nn.Parameter(
                torch.Tensor(1, self.n_primitives, self.dim - 1))
            if not self.train_n1:
                self.n1.requires_grad = False
                self.abn1n2n3_mask = self.abn1n2n3_mask * self.no_train_n1_mask

            self.n2 = nn.Parameter(
                torch.Tensor(1, self.n_primitives, self.dim - 1))
            self.n3 = nn.Parameter(
                torch.Tensor(1, self.n_primitives, self.dim - 1))

            assert [*self.n1.shape] == [1, self.n_primitives, self.dim - 1]

            # B=1, n_primitives, 1, 1
            self.a = nn.Parameter(
                torch.Tensor(1, self.n_primitives, self.dim - 1))
            self.b = nn.Parameter(
                torch.Tensor(1, self.n_primitives, self.dim - 1))
            assert [*self.a.shape] == [1, self.n_primitives, self.dim - 1]
            if not self.train_ab:
                self.a.requires_grad = False
                self.b.requires_grad = False
                self.abn1n2n3_mask = self.abn1n2n3_mask * self.no_train_ab_mask

            if self.use_paramnet:
                self.a.requires_grad = False
                self.b.requires_grad = False
                self.n1.requires_grad = False
                self.n2.requires_grad = False
                self.n3.requires_grad = False

        self.weight_init()

    def weight_init(self):
        if not self.sphere:
            if self.train_logits:
                torch.nn.init.uniform_(self.logits, 0, 1)

            if self.train_n1:
                torch.nn.init.uniform_(self.n1, 0, 1)
            else:
                torch.nn.init.ones_(self.n1)

            torch.nn.init.uniform_(self.n2, 0, 1)
            torch.nn.init.uniform_(self.n3, 0, 1)

            if self.train_ab:
                torch.nn.init.uniform_(self.a, 0.9, 1.1)
                torch.nn.init.uniform_(self.b, 0.9, 1.1)
            else:
                torch.nn.init.ones_(self.a)
                torch.nn.init.ones_(self.b)

        torch.nn.init.uniform_(self.rotation, 0, 1)
        torch.nn.init.uniform_(self.linear_scale, 0, 1)
        torch.nn.init.uniform_(self.transition, -self.transition_range,
                               self.transition_range)
        torch.nn.init.uniform_(self.prob, 0, 1)

    def get_primitive_params(self, x):
        B = x.shape[0]

        params = {}

        if self.use_paramnet and self.is_single_paramnet:
            pose_param = self.paramnet(x)

            pointer = 0
            next_pointer = self.rot_dim
            rotation_param = pose_param[:, :, pointer:next_pointer]

            pointer = next_pointer
            next_pointer = pointer + self.dim
            transition_param = pose_param[:, :, pointer:next_pointer]

            pointer = next_pointer
            next_pointer = pointer + self.dim
            scale_param = pose_param[:, :, pointer:next_pointer]

            pointer = next_pointer
            next_pointer = pointer + 1
            prob_param = pose_param[:, :, pointer:next_pointer]

        if self.use_paramnet and not self.is_single_paramnet:
            rotation = self.rotation + self.rotation_net(x)
        elif self.use_paramnet and self.is_single_paramnet:
            rotation = self.rotation + rotation_param
        else:
            rotation = self.rotation.repeat(B, 1, 1)
        if self.dim == 2:
            rotation = torch.tanh(self.rotation) * math.pi
        else:
            rotation = nn.functional.normalize(rotation, dim=-1)

        if self.use_paramnet and not self.is_single_paramnet:
            transition = self.transition + self.transition_net(x)
        elif self.use_paramnet and self.is_single_paramnet:
            transition = self.transition + transition_param
        else:
            transition = self.transition.repeat(B, 1, 1)

        if self.use_paramnet and not self.is_single_paramnet:
            linear_scale = self.linear_scale + self.scale_net(x)
        elif self.use_paramnet and self.is_single_paramnet:
            linear_scale = self.linear_scale + scale_param
        else:
            linear_scale = self.linear_scale.repeat(B, 1, 1)
        linear_scale = torch.tanh(linear_scale) + 1.1

        if self.use_paramnet and not self.is_single_paramnet:
            prob = self.prob + self.prob_net(x)
        elif self.use_paramnet and self.is_single_paramnet:
            prob = self.prob + prob_param
        else:
            prob = self.prob.repeat(B, 1, 1)
        prob = torch.sigmoid(prob)

        params.update({
            'rotation': rotation,
            'transition': transition,
            'linear_scale': linear_scale,
            'prob': prob
        })

        features = {}
        for feature_name in self.get_features_from:
            features[feature_name + '_feature'] = getattr(
                self, feature_name + '_net').get_feature(x)
        params.update(features)

        if not self.sphere:
            # logits
            probabilistic = self.training and self.train_logits and not self.quadrics
            if self.use_paramnet and self.train_logits:
                logits = self.logits + self.logits_net(x).view(
                    B, self.n_primitives, self.max_m, self.dim - 1)
            else:
                logits = self.logits.repeat(B, 1, 1, 1)

            m_vector = model_utils.get_m_vector(logits,
                                                probabilistic=probabilistic)

            # supershape params a, b
            if self.use_paramnet:
                abn1n2n3 = self.supershape_param_net(x).view(
                    B, self.n_primitives, self.abn1n2n3_dim,
                    self.dim - 1) * self.abn1n2n3_mask
            if self.use_paramnet and self.train_ab:
                a = self.a + abn1n2n3[:, :, 0, :]
                b = self.b + abn1n2n3[:, :, 1, :]
            else:
                a = self.a.repeat(B, 1, 1)
                b = self.b.repeat(B, 1, 1)
            a = nn.functional.relu(a)
            b = nn.functional.relu(b)

            if self.use_paramnet and self.train_n1:
                n1 = self.n1 + abn1n2n3[:, :, 2, :]
            else:
                n1 = self.n1.repeat(B, 1, 1)
            n1 = nn.functional.relu(n1)

            if self.use_paramnet:
                n2 = self.n2 + abn1n2n3[:, :, 3, :]
                n3 = self.n3 + abn1n2n3[:, :, 4, :]
            else:
                n2 = self.n2.repeat(B, 1, 1)
                n3 = self.n3.repeat(B, 1, 1)
            n2 = nn.functional.relu(n2 * self.n23scale) + self.n23bias
            n3 = nn.functional.relu(n3 * self.n23scale) + self.n23bias

            params.update({
                'n1': n1,
                'n2': n2,
                'n3': n3,
                'a': a,
                'b': b,
                'm_vector': m_vector,
            })

        return params

    def forward(self, x):
        assert x.ndim == 2  # B, latent dim
        assert x.shape[-1] == self.latent_dim
        return self.get_primitive_params(x)

    def to(self, device):
        super().to(device)
        if not self.sphere:
            if not self.train_logits:
                self.logits = self.logits.to(device)
            self.no_train_ab_mask = self.no_train_ab_mask.to(device)
            self.no_train_n1_mask = self.no_train_n1_mask.to(device)
            self.abn1n2n3_mask = self.abn1n2n3_mask.to(device)

        return self


class ParamNet(nn.Module):
    def __init__(self,
                 n_primitives,
                 in_channel,
                 param_dim,
                 dense=True,
                 layer_depth=0,
                 hidden_size=128,
                 **kwargs):
        super().__init__()
        self.n_primitives = n_primitives
        self.param_dim = param_dim
        self.dense = dense
        if layer_depth == 0:
            self.conv1d = nn.Linear(in_channel, in_channel)
            self.out_conv1d = nn.Linear(in_channel, n_primitives * param_dim)
            self.convs = []
        else:
            self.conv1d = nn.Linear(in_channel, hidden_size)
            self.convs = nn.ModuleList([
                nn.Linear(hidden_size, hidden_size) for _ in range(layer_depth)
            ])
            self.out_conv1d = nn.Linear(hidden_size, n_primitives * param_dim)
        #self.conv1d = nn.Conv1d(in_channel, in_channel, 1)
        self.act = nn.LeakyReLU(0.2, True)
        #self.out_conv1d = nn.Conv1d(in_channel, n_primitives * param_dim, 1)

    def get_feature(self, x):
        if self.dense:
            x = self.act(self.conv1d(x))

        for conv in self.convs:
            x = self.act(conv(x))

        return x

    def forward(self, x):
        B = x.shape[0]
        x = self.get_feature(x)

        return self.out_conv1d(x).view(B, self.n_primitives, self.param_dim)


class ParamNetResNetBlock(nn.Module):
    def __init__(self,
                 n_primitives,
                 in_channel,
                 param_dim,
                 hidden_size=128,
                 **kwargs):
        super().__init__()
        self.n_primitives = n_primitives
        self.param_dim = param_dim
        self.conv1d = nn.Conv1d(in_channel, hidden_size, 1)
        self.act = nn.LeakyReLU(0.2, True)
        self.out_conv1d = nn.Conv1d(hidden_size, n_primitives * param_dim, 1)
        self.bn = nn.BatchNorm1d(hidden_size)

        self.block0 = ResnetBlockConv1d(hidden_size)
        self.block1 = ResnetBlockConv1d(hidden_size)
        self.block2 = ResnetBlockConv1d(hidden_size)

    def forward(self, x):
        B = x.shape[0]
        net = x.view(B, -1, 1)

        net = self.conv1d(net)
        net = self.block0(net)
        net = self.block1(net)
        net = self.block2(net)

        net = self.out_conv1d(self.act(self.bn(net))).view(
            B, self.n_primitives, self.param_dim)

        return net


class ParamNetMLP(nn.Module):
    def __init__(
        self,
        n_primitives,
        in_channel,
        param_dim,
        layer_depth=4,
        skip_position=3,  # count start from input fc
        is_skip=True,
        hidden_size=128,
        **kwargs):
        super().__init__()
        self.n_primitives = n_primitives
        self.param_dim = param_dim
        self.in_conv1d = nn.Linear(in_channel, hidden_size)
        self.act = nn.LeakyReLU(0.2, True)
        self.out_conv1d = nn.Linear(hidden_size, n_primitives * param_dim)
        self.layer_depth = layer_depth
        self.is_skip = is_skip
        self.skip_position = skip_position

        self.conv1d = nn.Linear(hidden_size, hidden_size)
        self.conv1d_for_skip = nn.Linear(hidden_size + in_channel, hidden_size)

    def forward(self, x):
        B = x.shape[0]
        reshaped_x = x.view(B, -1)

        net = self.in_conv1d(reshaped_x)

        for idx in range(self.layer_depth):
            if idx == self.skip_position - 1:
                net = torch.cat([net, reshaped_x], axis=1)
                net = self.conv1d_for_skip(net)
            else:
                net = self.conv1d(net)

        net = self.out_conv1d(self.act(net)).view(B, self.n_primitives,
                                                  self.param_dim)

        return net


class ParamNetResNetBlockWOBatchNorm(nn.Module):
    def __init__(self,
                 n_primitives,
                 in_channel,
                 param_dim,
                 hidden_size=128,
                 layer_depth=4,
                 **kwargs):
        super().__init__()
        self.n_primitives = n_primitives
        self.param_dim = param_dim
        self.in_conv1d = nn.Linear(in_channel, hidden_size)
        self.act = nn.LeakyReLU(0.2, True)
        self.out_conv1d = nn.Linear(hidden_size, n_primitives * param_dim)
        self.layer_depth = layer_depth

        self.block = ResnetBlockFC(hidden_size)

    def forward(self, x):
        B = x.shape[0]
        net = x.view(B, -1)

        net = self.in_conv1d(net)

        for _ in range(self.layer_depth):
            net = self.block(net)

        net = self.out_conv1d(self.act(net)).view(B, self.n_primitives,
                                                  self.param_dim)

        return net


paramnet_dict = {
    'ParamNet': ParamNet,
    'ParamNetResNetBlock': ParamNetResNetBlock,
    'ParamNetMLP': ParamNetMLP,
    'ParamNetResNetBlockWOBatchNorm': ParamNetResNetBlockWOBatchNorm
}
