import torch
import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import timm
import torchvision.transforms as TVF
import numpy as np

# Vlastný extraktor vlastností pre CNN
class CustomCnnNetwork(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim, which_model='vit'):
        super(CustomCnnNetwork, self).__init__(observation_space, features_dim=features_dim)

        self.model = load_model(which_model)

        self.norm_transform = TVF.Normalize(
            mean=tuple(list(self.model.default_cfg["mean"]) + [0.5, 0.5, 0.5]),
            std=tuple(list(self.model.default_cfg["std"]) + [0.5, 0.5, 0.5])
        )

        self.flatten = nn.Flatten()

        # Spoločná lineárna vrstva pre actor a critic hlavy
        self.shared_fc = nn.Linear(384 * (self.model.patch_embed.num_patches), features_dim)


    def forward(self, observations):
        observations = self.norm_transform(observations)
        
        features = self.model(observations)[:, 1:, :] 

        flattened_features = self.flatten(features)

        shared_features = self.shared_fc(flattened_features)

        return shared_features


class CustomMlpExtractor(nn.Module):
    def __init__(self, feature_dim, action_dim, net_arch=[64, 64], activation_fn=nn.ReLU):
        super(CustomMlpExtractor, self).__init__()


    def forward(self, features):
        output = features
        return output, output

    def forward_actor(self, features):
        output = features
        return output

    def forward_critic(self, features):
        output = features
        return output




class CustomSharedCnnPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, net_arch, **kwargs):
        super(CustomSharedCnnPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            features_extractor_class=CustomCnnNetwork,
            features_extractor_kwargs=dict(features_dim=net_arch[0]),
            share_features_extractor=True,
            **kwargs,
        )

        self.mlp_extractor = CustomMlpExtractor(net_arch[0], action_space.n, net_arch=net_arch)

        actor_layers = []
        critic_layers = []
        input_dim = net_arch[0]

        
        for layer_size in net_arch:
            # Actor network
            actor_layers.append(nn.Linear(input_dim, layer_size))
            actor_layers.append(nn.BatchNorm1d(layer_size))
            actor_layers.append(nn.ReLU())
            actor_layers.append(nn.Dropout(p=0.1))

            # Critic network
            critic_layers.append(nn.Linear(input_dim, layer_size))
            critic_layers.append(nn.BatchNorm1d(layer_size))
            critic_layers.append(nn.ReLU())
            critic_layers.append(nn.Dropout(p=0.1)) 

            input_dim = layer_size

        self.action_net = nn.Sequential(*actor_layers, nn.Linear(input_dim, action_space.n))
        self.value_net = nn.Sequential(*critic_layers, nn.Linear(input_dim, 1))
        

    def forward(self, observations, deterministic=False, eval=False):
        features = self.extract_features(observations)
        latent_pi = features
        latent_vf = features

        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)

        log_probs = distribution.log_prob(actions)
        values = self.value_net(latent_vf).squeeze(-1)

        if eval == False:
            return actions, values, log_probs
    
        return actions, values, log_probs, distribution

    def predict_values(self, observations):
        features = self.extract_features(observations)
        return self.value_net(features).squeeze(-1)




def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def load_model(which):
    if which == 'resnet':
        # Použitie ResNetu pre Space Invaders
        model = timm.create_model('resnet18', pretrained=True, features_only=True)
        model.fc = nn.Identity()

        for p in model.parameters():
            p.requires_grad = False

        model.conv1 = nn.Conv2d(in_channels=4, out_channels=model.conv1.out_channels, 
                                kernel_size=model.conv1.kernel_size, 
                                stride=model.conv1.stride, 
                                padding=model.conv1.padding, 
                                bias=model.conv1.bias)

        for param in model.conv1.parameters():
            param.requires_grad = True
        
    elif which == 'vit':
        model = timm.create_model('eva02_small_patch14_224.mim_in22k', pretrained=True)
        model.reset_classifier(0, "")
        model.default_cfg.update({'input_size': (6, 224, 224)})

        model.patch_embed = timm.layers.PatchEmbed(
            img_size=224,
            patch_size=14,
            in_chans=6,
            embed_dim=model.embed_dim
        )

        for p in model.parameters():
            p.requires_grad = False

        for param in model.patch_embed.parameters():
            param.requires_grad = True

    elif which == 'cnn':
        model = nn.Sequential(
            layer_init(nn.Conv2d(4, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU()
        )
    
    return model