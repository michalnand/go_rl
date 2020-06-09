import torch
import torch.nn as nn
import visualise

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class ResidualBlock(torch.nn.Module):
    def __init__(self, kernels_count):
        super(ResidualBlock, self).__init__()
        self.layers = [ 
                        nn.Conv2d(kernels_count, kernels_count, kernel_size=3, stride=1, padding=1),
                        nn.BatchNorm2d(kernels_count),
                        nn.ReLU(),
                        nn.Conv2d(kernels_count, kernels_count, kernel_size=3, stride=1, padding=1),
                        nn.BatchNorm2d(kernels_count)
                    ]
        
        self.activation = nn.ReLU()
        self.model = nn.Sequential(*self.layers)

    def forward(self, x):
        y = self.model(x)
        return self.activation(y + x)




class Model(torch.nn.Module):
    def __init__(self, input_shape, outputs_count, residual_blocks_count = 16, kernels_count = 256):
        super(Model, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.input_shape    = input_shape
        self.outputs_count  = outputs_count
        
        input_channels  = self.input_shape[0]
        input_height    = self.input_shape[1]
        input_width     = self.input_shape[2]
       

        layers_features = []
        
        layers_features.append(nn.Conv2d(input_channels, kernels_count, kernel_size=3, stride=1, padding=1))
        layers_features.append(nn.BatchNorm2d(kernels_count))
        layers_features.append(nn.ReLU())

        for i in range(residual_blocks_count):
            layers_features.append(ResidualBlock(kernels_count))

        self.model_features = self._layers_to_model(layers_features)



        layers_policy = [
                            nn.Conv2d(kernels_count, 2, kernel_size=1, stride=1, padding=0),
                            nn.BatchNorm2d(2),
                            nn.ReLU(),
                            Flatten(),
                            nn.Linear(2*input_height*input_width, outputs_count)
                        ]

        self.model_policy = self._layers_to_model(layers_policy)



        layers_value = [
                            nn.Conv2d(kernels_count, 1, kernel_size=1, stride=1, padding=0),
                            nn.BatchNorm2d(1),
                            nn.ReLU(),
                            Flatten(),

                            nn.Linear(input_height*input_width, 256),
                            nn.ReLU(),

                            nn.Linear(256, 1),
                            nn.Tanh()
                        ]

        self.model_value = self._layers_to_model(layers_value)


        print(self.model_features)
        print(self.model_policy)
        print(self.model_value)

    def forward(self, input):
        features    = self.model_features(input)

        policy      = self.model_policy(features)
        value       = self.model_value(features)
        return policy, value
   
    def save(self, path):
        torch.save(self.model_features.state_dict(), path + "trained/features.pt")
        torch.save(self.model_policy.state_dict(), path + "trained/policy.pt")
        torch.save(self.model_value.state_dict(), path + "trained/value.pt")

    def load(self, path):
        self.model_features.load_state_dict(torch.load(path + "trained/features.pt", map_location = self.device))
        self.model_features.eval() 

        self.model_policy.load_state_dict(torch.load(path + "trained/policy.pt", map_location = self.device))
        self.model_policy.eval() 

        self.model_value.load_state_dict(torch.load(path + "trained/value.pt", map_location = self.device))
        self.model_value.eval() 

    def _layers_to_model(self, layers):

        for i in range(len(layers)):
            if isinstance(layers[i], nn.Conv2d) or isinstance(layers[i], nn.Linear):
                torch.nn.init.xavier_uniform_(layers[i].weight)

        model = nn.Sequential(*layers)
        model.to(self.device)

        return model

if __name__ == "__main__":
    board_size      = 19
    input_shape     = (6, board_size, board_size)
    outputs_count   = board_size*board_size + 1

    model = Model(input_shape, outputs_count, residual_blocks_count=16, kernels_count=256)

    input = torch.rand((1,) + input_shape)

    policy, value = model.forward(input)
    print(policy.shape, value.shape)

    '''
    g = visualise.make_dot(policy, model.state_dict())
    g.view()
    g.render("model.png")
    '''